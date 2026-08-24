import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch_mlir import fx


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "toolchains/mlir-tools/example/fvela-vpu-rope"
sys.path.insert(0, str(EXAMPLE_DIR))

from vfrope_test2 import (  # noqa: E402
    EXPECTED_Q15,
    INPUT_Q15,
    VFRopeSelfTest,
    export_linalg_mlir,
    vfrope_q15_chunk,
)


class SingleVFRope(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vfrope_q15_chunk(x, 65, 0)


class MixedVFRope(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        before = x + x
        rope = vfrope_q15_chunk(before, 65, 0)
        return rope + x


class OrdinaryAdd(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x


class AlternateConfig(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vfrope_q15_chunk(x, 64, 8)


class BoundaryConfig(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vfrope_q15_chunk(x, 65535, 65535)


class NegativeConfig(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vfrope_q15_chunk(x, -1, 0)


class OverflowConfig(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return vfrope_q15_chunk(x, 0, 65536)


@torch.library.custom_op("tvela::unsupported_chunk", mutates_args=())
def unsupported_chunk(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@unsupported_chunk.register_fake
def _unsupported_chunk_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class UnsupportedCustomOp(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unsupported_chunk(x)


def export_linalg(model: torch.nn.Module, example: torch.Tensor) -> str:
    module = fx.export_and_import(
        model.eval(),
        example,
        output_type=fx.OutputType.LINALG_ON_TENSORS,
        decomposition_table={},
        func_name="test",
    )
    return str(module)


class VFRopeLinalgExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.example = torch.tensor(INPUT_Q15, dtype=torch.int16)

    def test_single_vfrope(self) -> None:
        module = export_linalg(SingleVFRope(), self.example)
        self.assertEqual(module.count("tvela.vfrope_q15_chunk"), 1)
        self.assertNotIn("llvm.emit_c_interface", module)
        self.assertNotIn("torch.", module)

    def test_self_test_reference(self) -> None:
        expected = torch.tensor(EXPECTED_Q15, dtype=torch.int16)
        mismatches = VFRopeSelfTest()(self.example, expected)
        self.assertEqual(mismatches.dtype, torch.int32)
        self.assertEqual(int(mismatches.item()), 0)

    def test_self_test_counts_one_mismatch(self) -> None:
        expected = torch.tensor(EXPECTED_Q15, dtype=torch.int16)
        expected[0] += 1
        mismatches = VFRopeSelfTest()(self.example, expected)
        self.assertEqual(int(mismatches.item()), 1)

    def test_self_test_mlir_has_bare_metal_main(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "vfrope_self_test.mlir"
            export_linalg_mlir(output)
            module = output.read_text(encoding="utf-8")

        self.assertIn("func.func @vfrope_self_test", module)
        self.assertIn("func.func @main() -> i32", module)
        self.assertEqual(module.count("arith.constant dense<"), 2)
        self.assertNotIn("dense_resource", module)
        self.assertNotIn("llvm.emit_c_interface", module)
        self.assertNotIn("torch.", module)

    def test_mixed_linalg_and_vfrope(self) -> None:
        module = export_linalg(MixedVFRope(), self.example)
        self.assertEqual(module.count("tvela.vfrope_q15_chunk"), 1)
        self.assertEqual(module.count("linalg.generic"), 2)
        self.assertNotIn("torch.", module)

    def test_ordinary_linalg_is_unchanged(self) -> None:
        module = export_linalg(OrdinaryAdd(), self.example)
        self.assertEqual(module.count("linalg.generic"), 1)
        self.assertNotIn("tvela.", module)
        self.assertNotIn("torch.", module)

    def test_alternate_config_is_preserved(self) -> None:
        module = export_linalg(AlternateConfig(), self.example)
        self.assertIn("m = 64 : i64", module)
        self.assertIn("idx = 8 : i64", module)

    def test_config_boundary_is_accepted(self) -> None:
        module = export_linalg(BoundaryConfig(), self.example)
        self.assertIn("m = 65535 : i64", module)
        self.assertIn("idx = 65535 : i64", module)

    def test_out_of_range_config_is_rejected(self) -> None:
        for model in (NegativeConfig(), OverflowConfig()):
            with self.subTest(model=type(model).__name__):
                with self.assertRaisesRegex(Exception, "0..65535"):
                    export_linalg(model, self.example)

    def test_bad_shape_is_rejected(self) -> None:
        bad_shape = torch.zeros(4, dtype=torch.int16)
        with self.assertRaisesRegex(Exception, "tensor<8xi16>"):
            export_linalg(SingleVFRope(), bad_shape)

    def test_other_custom_op_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            export_linalg(UnsupportedCustomOp(), self.example)


if __name__ == "__main__":
    unittest.main()
