import argparse
from pathlib import Path

import torch


MAT_DIM_I = 1
MAT_DIM_K = 64
MAT_DIM_J = 64
PACKED_DIM_J = MAT_DIM_J // 4
TERNARY_OFF = "OFF"
TERNARY_ON = "ON"


def _validate_lhs(lhs: torch.Tensor) -> None:
    if lhs.dtype != torch.int8 or tuple(lhs.shape) != (MAT_DIM_I, MAT_DIM_K):
        raise ValueError("ternary matmul requires lhs tensor<1x64xi8>")


def _validate_packed_rhs(packed_rhs: torch.Tensor) -> None:
    if packed_rhs.dtype != torch.int8 or tuple(packed_rhs.shape) != (
        MAT_DIM_K,
        PACKED_DIM_J,
    ):
        raise ValueError(
            "ternary matmul requires packed_rhs tensor<64x16xi8>"
        )


def pack_ternary_weights(weights: torch.Tensor) -> torch.Tensor:
    if weights.dtype != torch.int8 or tuple(weights.shape) != (
        MAT_DIM_K,
        MAT_DIM_J,
    ):
        raise ValueError("weights must be tensor<64x64xi8>")
    if bool(torch.any((weights < -1) | (weights > 1)).item()):
        raise ValueError("weights must contain only -1, 0, or 1")

    groups = weights.reshape(MAT_DIM_K, PACKED_DIM_J, 4).to(torch.int16)
    codes = torch.where(groups == -1, 3, groups)
    shifts = torch.tensor(
        (0, 2, 4, 6), dtype=torch.int16, device=weights.device
    )
    packed = torch.sum(codes << shifts, dim=2)
    return packed.to(torch.int8).contiguous()


def _unpack_ternary_weights(packed_rhs: torch.Tensor) -> torch.Tensor:
    _validate_packed_rhs(packed_rhs)
    packed = packed_rhs.to(torch.int16) & 0xFF
    shifts = torch.tensor(
        (0, 2, 4, 6), dtype=torch.int16, device=packed_rhs.device
    )
    codes = (packed.unsqueeze(-1) >> shifts) & 0x03
    if bool(torch.any(codes == 2).item()):
        raise ValueError("packed_rhs contains the invalid ternary code 0b10")
    weights = torch.where(codes == 3, -1, codes)
    return weights.reshape(MAT_DIM_K, MAT_DIM_J).to(torch.int8)


@torch.library.custom_op("tvela::ternary_matmul", mutates_args=())
def ternary_matmul(
    lhs: torch.Tensor, packed_rhs: torch.Tensor
) -> torch.Tensor:
    _validate_lhs(lhs)
    weights = _unpack_ternary_weights(packed_rhs)
    accumulator = torch.matmul(lhs.to(torch.int32), weights.to(torch.int32))
    return accumulator.clamp(-128, 127).to(torch.int8)


@ternary_matmul.register_fake
def _ternary_matmul_fake(
    lhs: torch.Tensor, packed_rhs: torch.Tensor
) -> torch.Tensor:
    _validate_lhs(lhs)
    _validate_packed_rhs(packed_rhs)
    return torch.empty(
        (MAT_DIM_I, MAT_DIM_J), dtype=torch.int8, device=lhs.device
    )


def fvela_matmul(
    lhs: torch.Tensor, rhs: torch.Tensor, *, ternary: str = TERNARY_OFF
) -> torch.Tensor:
    if ternary == TERNARY_OFF:
        return torch.matmul(lhs, rhs)
    if ternary == TERNARY_ON:
        return ternary_matmul(lhs, rhs)
    raise ValueError('ternary must be "ON" or "OFF"')


class TernaryMatmulSelfTest(torch.nn.Module):
    def forward(
        self,
        lhs: torch.Tensor,
        packed_rhs: torch.Tensor,
        expected: torch.Tensor,
    ) -> torch.Tensor:
        actual = fvela_matmul(lhs, packed_rhs, ternary=TERNARY_ON)
        mismatches = (actual != expected).to(torch.int32)
        return torch.sum(mismatches, dtype=torch.int32)


def _reference_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lhs = torch.full((MAT_DIM_I, MAT_DIM_K), 4, dtype=torch.int8)
    column_values = (torch.arange(MAT_DIM_J, dtype=torch.int8) % 3) - 1
    weights = column_values.unsqueeze(0).repeat(MAT_DIM_K, 1)
    packed_rhs = pack_ternary_weights(weights)
    expected = torch.matmul(lhs.to(torch.int32), weights.to(torch.int32))
    expected = expected.clamp(-128, 127).to(torch.int8)
    return lhs, packed_rhs, expected


def check_reference() -> None:
    lhs, packed_rhs, expected = _reference_tensors()
    mismatches = int(
        TernaryMatmulSelfTest()(lhs, packed_rhs, expected).item()
    )
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

    lhs, packed_rhs, expected = _reference_tensors()

    with module.context, Location.unknown():
        i8 = IntegerType.get_signless(8)
        i32 = IntegerType.get_signless(32)
        lhs_type = RankedTensorType.get((MAT_DIM_I, MAT_DIM_K), i8)
        packed_rhs_type = RankedTensorType.get(
            (MAT_DIM_K, PACKED_DIM_J), i8
        )
        output_type = RankedTensorType.get((MAT_DIM_I, MAT_DIM_J), i8)
        status_type = RankedTensorType.get((), i32)

        def dense_i8(value: torch.Tensor, result_type):
            array = np.asarray(value.cpu().numpy(), dtype=np.int8)
            attribute = DenseElementsAttr.get(
                type=i8, array=array, shape=array.shape
            )
            return arith.ConstantOp(result_type, attribute)

        with InsertionPoint(module.body):
            main = func.FuncOp("main", ([], [i32]))
            with InsertionPoint(main.add_entry_block()):
                lhs_value = dense_i8(lhs, lhs_type)
                packed_rhs_value = dense_i8(packed_rhs, packed_rhs_type)
                expected_value = dense_i8(expected, output_type)
                result = func.CallOp(
                    [status_type],
                    "ternary_matmul_self_test",
                    [
                        lhs_value.result,
                        packed_rhs_value.result,
                        expected_value.result,
                    ],
                )
                status = tensor.ExtractOp(result.result, [], results=[i32])
                func.ReturnOp([status.result])


def export_linalg_mlir(output: Path) -> None:
    from torch_mlir import fx

    lhs, packed_rhs, expected = _reference_tensors()
    module = fx.export_and_import(
        TernaryMatmulSelfTest().eval(),
        lhs,
        packed_rhs,
        expected,
        output_type=fx.OutputType.LINALG_ON_TENSORS,
        decomposition_table={},
        func_name="ternary_matmul_self_test",
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
