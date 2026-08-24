# F-Vela VPU RoPE 예제

이 예제는 PyTorch custom operation을 self-checking bare-metal RISC-V ELF로
컴파일한다.

```text
vfrope_test2.py
  -> patched torch-mlir LINALG_ON_TENSORS
  -> Linalg + tvela.vfrope_q15_chunk
  -> fvela_rope_runtime
  -> build/vfrope_test2.elf
```

`toolchains/torch-mlir-vfrope`는 legacy가 아니라 이 ELF를 재생성할 때 필요한
build-time frontend다. 이미 생성된 ELF를 F-Vela simulator에서 실행할 때는
frontend source나 Python 환경을 전달할 필요가 없다.

## 준비 사항

- `toolchains/mlir-tools/build/bin` 아래에 빌드된 `npu-opt`,
  `npu-translate`, `npu-llc`
- `.conda-env/riscv-tools/bin` 아래의 HTIF bare-metal RISC-V toolchain
- 최초 frontend setup에 필요한 network 연결과 충분한 disk 공간

첫 frontend build는 시간이 오래 걸릴 수 있다.

## 빌드와 검사

저장소 root에서 다음 명령을 실행한다.

```sh
make -C toolchains/mlir-tools/example/fvela-vpu-rope \
  saturn-vfrope-test2-check
```

개별 단계는 다음 target으로 실행한다.

| Target | 생성 결과 |
|---|---|
| `saturn-vfrope-test2-frontend-setup` | pinned patched torch-mlir frontend |
| `saturn-vfrope-test2-export` | `build/vfrope_test2_linalg.mlir` |
| `saturn-vfrope-test2-lower` | `build/vfrope_test2_llvm.mlir` |
| `saturn-vfrope-test2-translate` | `build/vfrope_test2.ll` |
| `saturn-vfrope-test2-compile` | `build/vfrope_test2.elf` |
| `saturn-vfrope-test2-check` | frontend, MLIR, ELF 정적 검사 |

plain `make`, `make all`, `make saturn-vfrope-test2`는 모두 최종 ELF를 만든다.
`make check`와 `make frontend-setup`도 호환 alias로 제공한다. 실제 simulator
실행 경로가 저장소에 없으므로 `-run` target은 없다.

mixed Linalg/T-Vela MLIR만 생성하려면 다음과 같이 실행한다.

```sh
PYTHONPATH="$PWD/toolchains/torch-mlir-vfrope/build/tools/torch-mlir/python_packages/torch_mlir" \
  "$PWD/toolchains/torch-mlir-vfrope/.deps/venv/bin/python" \
  toolchains/mlir-tools/example/fvela-vpu-rope/vfrope_test2.py \
  --output ./vfrope_test2_linalg.mlir
```

## ELF self-test 계약

Python exporter가 실제 MLIR `main() -> i32`와 입력·기댓값을 생성한다. 두
배열은 최종 ELF의 `.rodata`에 들어가므로 별도 C harness는 사용하지 않는다.

```text
exit 0     eight outputs all match
exit 1..8  number of mismatched outputs
```

기댓값의 16-bit bit pattern은 다음과 같다.

```text
b905 f64a 7e07 637e be94 1d1b 9651 77cc
```

`saturn-vfrope-test2-check`는 ELF의 symbol, 입력·기댓값, vector instruction,
raw VFROPE opcode를 검사하지만 simulator를 실행하지는 않는다. 동일한 F-Vela
RTL과 LUT를 사용한 simulator에서 ELF 종료 상태가 `0`인지 확인하는 단계가 최종
hardware acceptance test다.

## 지원 범위

Compiler carrier는 contiguous `tensor<8xi16>`과 compile-time `m`/`idx`
`0..65535`를 운반한다. 하지만 현재 Python reference와 F-Vela hardware/LUT로
bit-exact 검증된 설정은 `m=65`, `idx=0`뿐이다. `e16, m1`에서 8개 lane을
사용하려면 `VLEN >= 128`이어야 한다.

현재 RTL은 `m`의 하위 12bit만 사용하며 `VL=8` tail 처리 때문에 `idx >= 8`에서
회전 lane이 활성화되지 않는다. Compiler가 값을 받아들인다는 사실은 그 설정의
수치 결과가 검증됐다는 뜻이 아니다.

전체 구조, Q1.15, `m`/`idx`, LUT, assembly 명령과 알려진 제한은
[DESIGN.md](DESIGN.md)에 설명돼 있다.
