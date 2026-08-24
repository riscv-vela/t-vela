#!/usr/bin/env python3
"""Export an int8 torch.matmul model to bufferized Linalg MLIR.

Two result types are supported, matching Gemmini's two C move-out formats:

  i8  -- the accumulator is truncated on the way out (fullC = false)
  i32 -- the accumulator is moved out unchanged      (fullC = true)
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch_mlir import fx
from torch_mlir.ir import (
    FunctionType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Operation,
    Type,
    TypeAttr,
)
from torch_mlir.passmanager import PassManager


MATRIX_SIZE = 16


class Int8Matmul(nn.Module):
    def forward(self, lhs, rhs):
        return torch.matmul(lhs, rhs)


def make_i32_accumulation_explicit(module, result_dtype: str) -> None:
    """Retype torch.matmul's result to si32, and cast back to si8 on request.

    PyTorch defines int8 matmul with an int8 result, and torch-mlir lowers every
    integer matmul through an i64 accumulator followed by a truncation to the
    result dtype. Neither suits Gemmini:

    * The i64 accumulator does not match Gemmini's i32 acc_t.
    * Truncating to i8 inside the matmul conversion aborts with
      "for conversion to byte or char type dstOriginalDtype has to be passed to
      convertScalarToDtype", because a signless i8 cannot say whether the Torch
      dtype was Byte or Char. Snapshot wheels then crash on the null result.

    Retyping the matmul result to si32 sidesteps both: the accumulator becomes
    i32, and the i8 cast -- when one is wanted -- becomes a standalone
    prims.convert_element_type that lowers through its own pattern.
    """
    def operation_name(op) -> str:
        return op.operation.name if hasattr(op, "operation") else op.name

    with module.context:
        functions = [
            op for op in module.body.operations if operation_name(op) == "func.func"
        ]
        if len(functions) != 1:
            raise RuntimeError("expected exactly one exported function")

        function = functions[0]
        block = function.regions[0].blocks[0]
        matmuls = [
            op for op in block.operations if operation_name(op) == "torch.aten.matmul"
        ]
        returns = [
            op for op in block.operations if operation_name(op) == "func.return"
        ]
        if len(matmuls) != 1 or len(returns) != 1:
            raise RuntimeError("expected one torch.aten.matmul and one func.return")

        matmul = matmuls[0]
        return_op = returns[0]
        i8_result_type = matmul.results[0].type
        expected_i8_type = f"!torch.vtensor<[{MATRIX_SIZE},{MATRIX_SIZE}],si8>"
        if str(i8_result_type) != expected_i8_type:
            raise RuntimeError(f"unexpected torch.matmul type: {i8_result_type}")

        i32_result_type = Type.parse(
            f"!torch.vtensor<[{MATRIX_SIZE},{MATRIX_SIZE}],si32>"
        )
        matmul.results[0].set_type(i32_result_type)

        if result_dtype == "i32":
            # Move the i32 accumulator out directly; the function now returns it.
            return_op.operands[0] = matmul.results[0]
            old_type = FunctionType(TypeAttr(function.attributes["function_type"]).value)
            function.attributes["function_type"] = TypeAttr.get(
                FunctionType.get(list(old_type.inputs), [i32_result_type])
            )
            return

        with InsertionPoint(return_op):
            # Torch ScalarType::Char (signed int8) has numeric value 1.
            int8_dtype = Operation.create(
                "torch.constant.int",
                results=[Type.parse("!torch.int")],
                attributes={
                    "value": IntegerAttr.get(IntegerType.get_signless(64), 1)
                },
                loc=matmul.location,
            )
            result_cast = Operation.create(
                "torch.prims.convert_element_type",
                results=[i8_result_type],
                operands=[matmul.results[0], int8_dtype.results[0]],
                loc=matmul.location,
            )
        return_op.operands[0] = result_cast.results[0]


def write_module(path: Path, module) -> None:
    path.write_text(module.operation.get_asm(enable_debug_info=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build"),
        help="directory for generated MLIR files",
    )
    parser.add_argument(
        "--result-dtype",
        choices=["i8", "i32"],
        default="i8",
        help="element type of the matmul result",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.result_dtype

    model = Int8Matmul().eval()
    lhs = torch.ones((MATRIX_SIZE, MATRIX_SIZE), dtype=torch.int8)
    rhs = torch.full((MATRIX_SIZE, MATRIX_SIZE), 2, dtype=torch.int8)

    module = fx.export_and_import(
        model,
        lhs,
        rhs,
        output_type="raw",
        func_name="matmul",
    )
    make_i32_accumulation_explicit(module, args.result_dtype)
    module = fx._module_lowering(
        verbose=False,
        enable_ir_printing=False,
        output_type=fx.OutputType.LINALG_ON_TENSORS,
        torch_mod=module,
    )
    linalg_path = args.output_dir / f"matmul-{suffix}-linalg.mlir"
    write_module(linalg_path, module)

    # npu-opt's Gemmini conversion consumes memref-based Linalg operations.
    # Keep identity layouts because its tiling code expects statically shaped,
    # row-major memrefs.
    pipeline = (
        "builtin.module("
        "empty-tensor-to-alloc-tensor,"
        "one-shot-bufferize{bufferize-function-boundaries "
        "function-boundary-type-conversion=identity-layout-map},"
        "buffer-results-to-out-params{modify-public-functions=true "
        "hoist-static-allocs=true},"
        "canonicalize"
        ")"
    )
    with module.context:
        PassManager.parse(pipeline).run(module.operation)

    bufferized_path = args.output_dir / f"matmul-{suffix}-bufferized.mlir"
    write_module(bufferized_path, module)

    bufferized_asm = module.operation.get_asm(enable_debug_info=False)
    if "linalg.matmul" not in bufferized_asm:
        raise RuntimeError(
            "torch.matmul did not lower to linalg.matmul; inspect "
            f"{linalg_path} for unsupported Torch operations"
        )
    if "tensor<" in bufferized_asm:
        raise RuntimeError(
            "bufferization left tensor values in the module; inspect "
            f"{bufferized_path}"
        )

    print(f"wrote {linalg_path}")
    print(f"wrote {bufferized_path}")


if __name__ == "__main__":
    main()
