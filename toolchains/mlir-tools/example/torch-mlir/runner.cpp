#include <cstdint>
#include <cstdio>
#include <vector>

#include <npu/Core/Container.h>

// Gemmini moves C out either as elem_t (i8) or, with fullC, as acc_t (i32).
// RESULT_I32 selects the module compiled from --result-dtype=i32.
#ifdef RESULT_I32
using result_t = int32_t;
static constexpr const char *kResultName = "int32";
#else
using result_t = int8_t;
static constexpr const char *kResultName = "int8";
#endif

// Override to exercise accumulators that do not fit in int8. The i8 result then
// tells apart a truncating move-out from a saturating one.
#ifndef LHS_VALUE
#define LHS_VALUE 1
#endif
#ifndef RHS_VALUE
#define RHS_VALUE 2
#endif

extern "C" void _mlir_ciface_matmul(MemRef<int8_t, 2> *lhs,
                                    MemRef<int8_t, 2> *rhs,
                                    MemRef<result_t, 2> *output);

int main() {
  constexpr size_t kMatrixSize = 16;
  constexpr int8_t kLhs = LHS_VALUE;
  constexpr int8_t kRhs = RHS_VALUE;
  // Gemmini accumulates in i32; an i8 result is then narrowed on move-out.
  constexpr int32_t kAccumulated =
      static_cast<int32_t>(kMatrixSize) * kLhs * kRhs;
#if defined(RESULT_I32)
  constexpr result_t kExpected = kAccumulated;
  static constexpr const char *kNarrowing = "none";
#elif defined(SATURATING_RESULT)
  // Gemmini clamps when it narrows the accumulator on move-out.
  constexpr result_t kExpected =
      kAccumulated > 127 ? 127 : (kAccumulated < -128 ? -128 : kAccumulated);
  static constexpr const char *kNarrowing = "saturating";
#else
  // A separate arith.trunci loop keeps the low 8 bits.
  constexpr result_t kExpected = static_cast<result_t>(kAccumulated);
  static constexpr const char *kNarrowing = "wrapping";
#endif
  const std::vector<size_t> shape = {kMatrixSize, kMatrixSize};

  MemRef<int8_t, 2> lhs(shape, kLhs);
  MemRef<int8_t, 2> rhs(shape, kRhs);
  MemRef<result_t, 2> output(shape, result_t{0});

  _mlir_ciface_matmul(&lhs, &rhs, &output);

  for (size_t row = 0; row < kMatrixSize; ++row) {
    for (size_t col = 0; col < kMatrixSize; ++col) {
      const result_t actual = output.getData()[row * kMatrixSize + col];
      if (actual != kExpected) {
        std::printf("FAIL: %s/%s output[%zu, %zu] = %d, expected %d "
                    "(i32 accumulator = %d)\n",
                    kResultName, kNarrowing, row, col, static_cast<int>(actual),
                    static_cast<int>(kExpected), kAccumulated);
        return 1;
      }
    }
  }

  std::printf("PASS: torch.matmul int8 16x16 -> %s (%s) produced %d in every "
              "cell, i32 accumulator = %d\n",
              kResultName, kNarrowing, static_cast<int>(kExpected),
              kAccumulated);
  return 0;
}
