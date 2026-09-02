//===-- TVela.td - T-Vela dialect operation definitions -------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TVELA_OPS
#define TVELA_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TVela_Dialect : Dialect {
  let name = "tvela";
  let cppNamespace = "::npu::tvela";
  let summary = "Operations implemented by the T-Vela target";
  let description = [{
    The T-Vela dialect preserves target-specific operations that must not be
    decomposed into generic tensor arithmetic before instruction selection.
  }];
}

class TVela_Op<string mnemonic, list<Trait> traits = []>
    : Op<TVela_Dialect, mnemonic, traits>;

def VFRopeQ15ChunkOp : TVela_Op<"vfrope_q15_chunk", [Pure]> {
  let summary = "Apply one F-Vela Q1.15 RoPE chunk";
  let description = [{
    Applies the F-Vela RoPE operation to four adjacent pairs in a tensor of
    eight signed Q1.15 bit patterns. The m and idx configuration values are
    packed into unsigned 16-bit fields for the assembly runtime.
  }];

  let arguments = (ins AnyRankedTensor:$input,
                       I64Attr:$m,
                       I64Attr:$idx);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
  let hasVerifier = 1;
}

def VFRopeQ15ChunkBufferOp : TVela_Op<"vfrope_q15_chunk_buffer"> {
  let summary = "Buffer form of the F-Vela Q1.15 RoPE chunk";
  let arguments = (ins
      Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$input,
      Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$output,
      I64Attr:$m,
      I64Attr:$idx);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
  let hasVerifier = 1;
}

def TernaryMatmulOp : TVela_Op<"ternary_matmul", [Pure]> {
  let summary = "Multiply int8 activations by packed ternary weights";
  let description = [{
    Multiplies a tensor<1x64xi8> activation row by tensor<64x16xi8>
    weights. Each packed weight byte contains four signed 2-bit values and
    produces four adjacent int8 result columns. The result is saturated to
    the int8 range during the Gemmini move-out.
  }];

  let arguments = (ins AnyRankedTensor:$lhs,
                       AnyRankedTensor:$packedRhs);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$lhs `,` $packedRhs attr-dict `:` type($lhs) `,` type($packedRhs) `->` type($result)";
  let hasVerifier = 1;
}

def TernaryMatmulBufferOp : TVela_Op<"ternary_matmul_buffer"> {
  let summary = "Buffer form of the F-Vela ternary matmul";
  let arguments = (ins
      Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$lhs,
      Arg<AnyMemRef, "", [MemReadAt<1, FullEffect>]>:$packedRhs,
      Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$output);
  let assemblyFormat = "$lhs `,` $packedRhs `,` $output attr-dict `:` type($lhs) `,` type($packedRhs) `,` type($output)";
  let hasVerifier = 1;
}

#endif // TVELA_OPS
