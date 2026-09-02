# Torch-MLIR F-Vela frontend

이 디렉터리는 F-Vela 전용 PyTorch custom operation을 `LINALG_ON_TENSORS`까지 보존하는 build-time frontend다.

```text
torch.tvela.vfrope_q15_chunk -> tvela.vfrope_q15_chunk
torch.tvela.ternary_matmul   -> tvela.ternary_matmul
```

기본 torch-mlir는 사용자 정의 연산을 나타내는 `torch.operator`를 Linalg backend contract에서 허용하지 않는다. 이 패치는 위 두 operation만 검증해 textual T-Vela operation으로 바꾸고, 나머지 Torch operation은 정상적인 Linalg/Tensor/Arith 경로로 내린다.

## Textual MLIR 경계

patched torch-mlir가 출력한 textual mixed Linalg/T-Vela MLIR을 T-Vela compiler toolchain 입력으로 사용한다.

| Component | Pinned revision |
|---|---|
| PyTorch | `2.14.0.dev20260719` |
| torch-mlir | `69a686fa3c1c68bc4b1bfedb754eaebb4ce22407` |

정확한 버전은 `versions.lock`에 작성되어 있다.

## Setup

```sh
toolchains/torch-mlir-fvela/setup.sh
```

`setup.sh`는 `.deps/torch-mlir` checkout을 pinned revision으로 되돌린 뒤 `patches/0001-preserve-tvela-ops-in-linalg.patch`를 적용하고 빌드한다.
