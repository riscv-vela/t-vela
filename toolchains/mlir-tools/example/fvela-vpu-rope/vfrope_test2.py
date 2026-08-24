import argparse
from pathlib import Path

import torch


ROPE_ELEMENTS = 8
MAX_CONFIG_FIELD = 0xFFFF
SELF_TEST_M = 65
SELF_TEST_IDX = 0
# Bit-exact coefficients for the 65/0 self-test. Exported custom operations
# preserve their own m/idx attributes instead of lowering this Python body.
COS_Q15 = (-18512, 31788, 1475, -6641)
SIN_Q15 = (27084, -8421, -32756, -32036)
INPUT_Q15 = (
    0x2000,
    0x4000,
    0x6000,
    0x7FFF,
    -0x2000,
    -0x4000,
    -0x6000,
    -0x8000,
)
EXPECTED_Q15 = (-18171, -2486, 32263, 25470, -16748, 7451, -27055, 30668)


def _validate(x: torch.Tensor, m: int, idx: int) -> None:
    if x.dtype != torch.int16 or tuple(x.shape) != (ROPE_ELEMENTS,):
        raise ValueError("vfrope requires a tensor<8xi16>")
    if not 0 <= m <= MAX_CONFIG_FIELD or not 0 <= idx <= MAX_CONFIG_FIELD:
        raise ValueError("vfrope requires m and idx in the range 0..65535")


def _mul_q15(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    product = a * b
    rounding = torch.where(product >= 0, 1 << 14, -(1 << 14))
    return (product + rounding) >> 15


@torch.library.custom_op("tvela::vfrope_q15_chunk", mutates_args=())
def vfrope_q15_chunk(x: torch.Tensor, m: int, idx: int) -> torch.Tensor:
    _validate(x, m, idx)
    pairs = x.to(torch.int64).reshape(4, 2)
    cos_q15 = torch.tensor(COS_Q15, dtype=torch.int64, device=x.device)
    sin_q15 = torch.tensor(SIN_Q15, dtype=torch.int64, device=x.device)

    even = pairs[:, 0]
    odd = pairs[:, 1]
    output_even = _mul_q15(even, cos_q15) - _mul_q15(odd, sin_q15)
    output_odd = _mul_q15(odd, cos_q15) + _mul_q15(even, sin_q15)
    output = torch.stack((output_even, output_odd), dim=1).reshape(ROPE_ELEMENTS)
    return output.clamp(-32768, 32767).to(torch.int16)


@vfrope_q15_chunk.register_fake
def _vfrope_q15_chunk_fake(x: torch.Tensor, m: int, idx: int) -> torch.Tensor:
    _validate(x, m, idx)
    return torch.empty_like(x)


class VFRopeSelfTest(torch.nn.Module):
    def forward(
        self, x: torch.Tensor, expected: torch.Tensor
    ) -> torch.Tensor:
        actual = vfrope_q15_chunk(x, SELF_TEST_M, SELF_TEST_IDX)
        mismatches = (actual != expected).to(torch.int32)
        return torch.sum(mismatches, dtype=torch.int32)


def check_reference() -> None:
    input_tensor = torch.tensor(INPUT_Q15, dtype=torch.int16)
    expected = torch.tensor(EXPECTED_Q15, dtype=torch.int16)
    mismatches = int(VFRopeSelfTest()(input_tensor, expected).item())
    if mismatches != 0:
        raise RuntimeError(f"reference mismatch count: {mismatches}")


def _add_bare_metal_main(module) -> None:
    import numpy as np
    from torch_mlir.dialects import arith, func, tensor
    from torch_mlir.ir import (
        DenseElementsAttr,
        InsertionPoint,
        IntegerType,
        Location,
        RankedTensorType,
    )

    with module.context, Location.unknown():
        i16 = IntegerType.get_signless(16)
        i32 = IntegerType.get_signless(32)
        vector_type = RankedTensorType.get((ROPE_ELEMENTS,), i16)
        status_type = RankedTensorType.get((), i32)

        def dense_i16(values: tuple[int, ...]):
            array = np.asarray(values, dtype=np.int16)
            attribute = DenseElementsAttr.get(
                type=i16, array=array, shape=array.shape
            )
            return arith.ConstantOp(vector_type, attribute)

        with InsertionPoint(module.body):
            main = func.FuncOp("main", ([], [i32]))
            with InsertionPoint(main.add_entry_block()):
                input_value = dense_i16(INPUT_Q15)
                expected_value = dense_i16(EXPECTED_Q15)
                result = func.CallOp(
                    [status_type],
                    "vfrope_self_test",
                    [input_value.result, expected_value.result],
                )
                status = tensor.ExtractOp(
                    result.result, [], results=[i32]
                )
                func.ReturnOp([status.result])


def export_linalg_mlir(output: Path) -> None:
    from torch_mlir import fx

    model = VFRopeSelfTest().eval()
    input_example = torch.tensor(INPUT_Q15, dtype=torch.int16)
    expected_example = torch.tensor(EXPECTED_Q15, dtype=torch.int16)
    module = fx.export_and_import(
        model,
        input_example,
        expected_example,
        output_type=fx.OutputType.LINALG_ON_TENSORS,
        decomposition_table={},
        func_name="vfrope_self_test",
    )
    _add_bare_metal_main(module)
    module.operation.verify()
    output.write_text(str(module) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    check_reference()
    export_linalg_mlir(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
