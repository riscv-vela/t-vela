import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch_mlir import fx


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "toolchains/mlir-tools/example/fvela-npu-ternary"
sys.path.insert(0, str(EXAMPLE_DIR))

from ternary_matmul import (  # noqa: E402
    MAT_DIM_J,
    MAT_DIM_K,
    TernaryMatmulSelfTest,
    _reference_tensors,
    export_linalg_mlir,
    fvela_matmul,
    pack_ternary_weights,
)


class TernaryOn(torch.nn.Module):
    def forward(
        self, lhs: torch.Tensor, packed_rhs: torch.Tensor
    ) -> torch.Tensor:
        return fvela_matmul(lhs, packed_rhs, ternary="ON")


class TernaryOff(torch.nn.Module):
    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return fvela_matmul(lhs, rhs)


def export_linalg(
    model: torch.nn.Module, lhs: torch.Tensor, rhs: torch.Tensor
) -> str:
    module = fx.export_and_import(
        model.eval(),
        lhs,
        rhs,
        output_type=fx.OutputType.LINALG_ON_TENSORS,
        decomposition_table={},
        func_name="test",
    )
    return str(module)


class TernaryMatmulLinalgExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.lhs, self.packed_rhs, self.expected = _reference_tensors()

    def test_packing_order_and_encoding(self) -> None:
        weights = torch.zeros((MAT_DIM_K, MAT_DIM_J), dtype=torch.int8)
        weights[0, :4] = torch.tensor((-1, 0, 1, -1), dtype=torch.int8)
        packed = pack_ternary_weights(weights)
        self.assertEqual(int(packed[0, 0].item()) & 0xFF, 0xD3)

    def test_reference_and_saturation(self) -> None:
        mismatches = TernaryMatmulSelfTest()(
            self.lhs, self.packed_rhs, self.expected
        )
        self.assertEqual(int(mismatches.item()), 0)
        self.assertEqual(int(self.expected[0, 0].item()), -128)
        self.assertEqual(int(self.expected[0, 1].item()), 0)
        self.assertEqual(int(self.expected[0, 2].item()), 127)

    def test_on_exports_tvela_carrier(self) -> None:
        module = export_linalg(TernaryOn(), self.lhs, self.packed_rhs)
        self.assertEqual(module.count("tvela.ternary_matmul"), 1)
        self.assertNotIn("torch.", module)

    def test_off_uses_ordinary_matmul(self) -> None:
        lhs = self.lhs.to(torch.float32)
        rhs = torch.zeros((MAT_DIM_K, MAT_DIM_J), dtype=torch.float32)
        eager = fvela_matmul(lhs, rhs)
        self.assertTrue(torch.equal(eager, torch.matmul(lhs, rhs)))
        module = export_linalg(TernaryOff(), lhs, rhs)
        self.assertIn("linalg.matmul", module)
        self.assertNotIn("tvela.ternary_matmul", module)
        self.assertNotIn("torch.", module)

    def test_invalid_mode_is_rejected(self) -> None:
        rhs = torch.zeros((MAT_DIM_K, MAT_DIM_J), dtype=torch.int8)
        with self.assertRaisesRegex(ValueError, '"ON" or "OFF"'):
            fvela_matmul(self.lhs, rhs, ternary="on")

    def test_export_has_bare_metal_main(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ternary_matmul.mlir"
            export_linalg_mlir(output)
            module = output.read_text(encoding="utf-8")

        self.assertIn("func.func @ternary_matmul_self_test", module)
        self.assertIn("func.func @main() -> i32", module)
        self.assertEqual(module.count("arith.constant dense<"), 3)
        self.assertEqual(module.count("tvela.ternary_matmul"), 1)
        self.assertNotIn("torch.", module)


if __name__ == "__main__":
    unittest.main()
