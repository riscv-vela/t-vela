# Torch-MLIR T-Vela VFROPE frontend

이 디렉터리는 legacy bridge가 아니라 PyTorch custom VFROPE를
`LINALG_ON_TENSORS`까지 보존하기 위한 현재 build-time frontend다.

stock torch-mlir는 opaque `torch.tvela.vfrope_q15_chunk`를 Linalg backend
contract에서 허용하지 않는다. tracked patch는 이 operation만 검증하여 textual
`tvela.vfrope_q15_chunk`로 바꾸고, 나머지 Torch operation은 정상적으로
Linalg/Tensor/Arith로 내린다.

```text
torch.tvela.vfrope_q15_chunk(tensor<8xi16>, m, idx)
  -> tvela.vfrope_q15_chunk {m, idx}
```

허용되는 configuration은 compile-time 정수 `m`/`idx` `0..65535`다. 다른
opaque Torch operation, shape, dtype, dynamic configuration은 거부한다. 지원하는
frontend 경로는 patched `fx.OutputType.LINALG_ON_TENSORS` 하나뿐이며, 과거의
standalone RAW bridge는 사용하지 않는다.

## Textual MLIR 경계

Frontend가 사용하는 LLVM과 T-Vela compiler의 LLVM revision이 서로 다르므로 두
프로젝트를 한 process에 link하지 않는다. patched torch-mlir가 출력한 textual
mixed Linalg/T-Vela MLIR을 안정적인 경계로 사용한다.

| Component | Pinned revision |
|---|---|
| PyTorch | `2.14.0.dev20260719` |
| torch-mlir | `69a686fa3c1c68bc4b1bfedb754eaebb4ce22407` |
| frontend LLVM | `068c6c5c0c8a0555036a2ff09a99f486548e6e8d` |

정확한 값은 [versions.lock](versions.lock)이 source of truth다.

## 관리되는 파일

`setup.sh`는 `.deps/torch-mlir` checkout을 pinned revision으로 되돌린 뒤 tracked
patch를 다시 적용한다. `.deps/`와 `build/`는 생성물이며 commit 대상이 아니다.
해당 checkout에 수동 변경을 보관하면 setup 과정에서 덮어쓸 수 있다.

```sh
toolchains/torch-mlir-vfrope/setup.sh
```

생성된 ELF를 실행하는 데 이 frontend는 필요하지 않다. PyTorch source에서 MLIR과
ELF를 재생성하거나 frontend conversion을 검사할 때만 필요하다.

## 수치 reference 제한

Compiler carrier는 전체 16-bit configuration을 보존하지만 예제 Python eager
implementation의 `COS_Q15`/`SIN_Q15`는 `m=65`, `idx=0`용 고정 값이다. 다른
configuration의 export 가능 여부와 그 configuration의 수치 정확성은 별개의
계약이다. 현재 eager 코드의 이 동작은 유지하며, 65/0 외 결과는 reference나
hardware 검증 결과로 사용하지 않는다.
