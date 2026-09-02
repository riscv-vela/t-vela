# F-Vela NPU ternary matmul

이 예제는 `f-vela/software/test/ternary_gemm.c`와 같은 shape으로 PyTorch에서
bare-metal RISC-V ELF까지 내린다.

```text
lhs         tensor<1x64xi8>
logical rhs tensor<64x64xi8>, values in {-1, 0, 1}
packed rhs  tensor<64x16xi8>, four 2-bit weights per byte
result      tensor<1x64xi8>, saturating move-out
```

2-bit encoding은 low bits부터 `00 = 0`, `01 = 1`, `11 = -1`이다.

## PyTorch API

```python
from ternary_matmul import fvela_matmul, pack_ternary_weights

ordinary = fvela_matmul(lhs, rhs)  # ternary="OFF"
packed_rhs = pack_ternary_weights(ternary_rhs)
ternary_result = fvela_matmul(lhs, packed_rhs, ternary="ON")
```

`OFF`는 `torch.matmul(lhs, rhs)`를 그대로 호출한다. `ON`만 opaque custom op로
export되어 `tvela.ternary_matmul` carrier를 거치며, 일반 matmul lowering에
`ternary` mode bit를 추가한다. 지원 문자열은 대문자 `ON`과 `OFF`뿐이다.

## Build and check

저장소 root에서 다음을 실행한다.

```sh
make -C toolchains/mlir-tools/example/fvela-npu-ternary
make -C toolchains/mlir-tools/example/fvela-npu-ternary check
```

첫 실행은 `toolchains/torch-mlir-fvela/setup.sh`를 통해 pinned frontend를
빌드한다. `check`는 Python reference, Torch-MLIR carrier, ternary mode bit,
RISC-V ELF와 custom instruction encoding을 확인한다.

생성물은 `build/ternary_matmul.elf`다. 연결된 F-Vela simulator 환경에서는
해당 ELF를 실행해 exit status 0인지 확인한다.
