#!/bin/bash
npu_OPT := ../build/bin/npu-opt
npu_TRANSLATE := ../build/bin/npu-translate
npu_LLC := ../build/bin/npu-llc
INTERFACES:= ../midend/Interfaces

mvin-mvout-lower:
	@${npu_OPT} ./mvin-mvout.mlir \
		-lower-gemmini \
		-o log.mlir

mvin-mvout-translate:
	@${npu_OPT} ./mvin-mvout.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

mvin-mvout-asm:
	@${npu_OPT} ./mvin-mvout.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

mvin-mvout-run:
	@${npu_OPT} ./mvin-mvout.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matrix-add-lower:
	@${npu_OPT} ./matrix-add.mlir \
		-lower-gemmini \
		-o log.mlir

matrix-add-translate:
	@${npu_OPT} ./matrix-add.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

matrix-add-asm:
	@${npu_OPT} ./matrix-add.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

matrix-add-run:
	@${npu_OPT} ./matrix-add.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matrix-add-scale-lower:
	@${npu_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini \
		-o log.mlir

matrix-add-scale-translate:
	@${npu_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

matrix-add-scale-asm:
	@${npu_OPT} ./matrix-add-scale.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

matrix-add-scale-run:
	@${npu_OPT} ./matrix-add-scale.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

transpose-lower:
	@${npu_OPT} ./transpose.mlir \
		-lower-gemmini \
		-o log.mlir

transpose-translate:
	@${npu_OPT} ./transpose.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

transpose-asm:
	@${npu_OPT} ./transpose.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

transpose-run:
	@${npu_OPT} ./transpose.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matmul-os-lower:
	@${npu_OPT} ./matmul-os.mlir \
		-lower-gemmini \
		-o log.mlir

matmul-os-translate:
	@${npu_OPT} ./matmul-os.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

matmul-os-asm:
	@${npu_OPT} ./matmul-os.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

matmul-os-run:
	@${npu_OPT} ./matmul-os.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

compute-accumulated-lower:
	@${npu_OPT} ./compute-accumulated.mlir \
		-lower-gemmini \
		-o log.mlir

compute-accumulated-translate:
	@${npu_OPT} ./compute-accumulated.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

compute-accumulated-asm:
	@${npu_OPT} ./compute-accumulated.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

compute-accumulated-run:
	@${npu_OPT} ./compute-accumulated.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

matmul-ws-lower:
	@${npu_OPT} ./matmul-ws.mlir \
		-lower-gemmini \
		-o log.mlir

matmul-ws-translate:
	@${npu_OPT} ./matmul-ws.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

matmul-ws-asm:
	@${npu_OPT} ./matmul-ws.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

matmul-ws-run:
	@${npu_OPT} ./matmul-ws.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-lower:
	@${npu_OPT} ./tile-matmul.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-translate:
	@${npu_OPT} ./tile-matmul.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-matmul-asm:
	@${npu_OPT} ./tile-matmul.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-matmul-run:
	@${npu_OPT} ./tile-matmul.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-os-lower:
	@${npu_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-os-translate:
	@${npu_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-matmul-os-asm:
	@${npu_OPT} ./tile-matmul-os.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-matmul-os-run:
	@${npu_OPT} ./tile-matmul-os.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-igelu-lower:
	@${npu_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-igelu-translate:
	@${npu_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-matmul-ws-igelu-asm:
	@${npu_OPT} ./tile-matmul-ws-igelu.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-igelu-run:
	@${npu_OPT} ./tile-matmul-ws-igelu.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-relu-lower:
	@${npu_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-relu-translate:
	@${npu_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-matmul-ws-relu-asm:
	@${npu_OPT} ./tile-matmul-ws-relu.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-relu-run:
	@${npu_OPT} ./tile-matmul-ws-relu.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-softmax-lower:
	@${npu_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-softmax-translate:
	@${npu_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-matmul-ws-softmax-asm:
	@${npu_OPT} ./tile-matmul-ws-softmax.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-softmax-run:
	@${npu_OPT} ./tile-matmul-ws-softmax.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-matmul-ws-layernorm-lower:
	@${npu_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini \
		-o log.mlir

tile-matmul-ws-layernorm-translate:
	@${npu_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-matmul-ws-layernorm-asm:
	@${npu_OPT} ./tile-matmul-ws-layernorm.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-matmul-ws-layernorm-run:
	@${npu_OPT} ./tile-matmul-ws-layernorm.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-lower:
	@${npu_OPT} ./tile-conv.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-translate:
	@${npu_OPT} ./tile-conv.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-conv-asm:
	@${npu_OPT} ./tile-conv.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-conv-run:
	@${npu_OPT} ./tile-conv.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-igelu-lower:
	@${npu_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-igelu-translate:
	@${npu_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-conv-igelu-asm:
	@${npu_OPT} ./tile-conv-igelu.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-conv-igelu-run:
	@${npu_OPT} ./tile-conv-igelu.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-softmax-lower:
	@${npu_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-softmax-translate:
	@${npu_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-conv-softmax-asm:
	@${npu_OPT} ./tile-conv-softmax.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-conv-softmax-run:
	@${npu_OPT} ./tile-conv-softmax.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-relu-lower:
	@${npu_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-relu-translate:
	@${npu_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-conv-relu-asm:
	@${npu_OPT} ./tile-conv-relu.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-conv-relu-run:
	@${npu_OPT} ./tile-conv-relu.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-conv-layernorm-lower:
	@${npu_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini \
		-o log.mlir

tile-conv-layernorm-translate:
	@${npu_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-conv-layernorm-asm:
	@${npu_OPT} ./tile-conv-layernorm.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-conv-layernorm-run:
	@${npu_OPT} ./tile-conv-layernorm.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

tile-rect-conv-lower:
	@${npu_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini \
		-o log.mlir

tile-rect-conv-translate:
	@${npu_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

tile-rect-conv-asm:
	@${npu_OPT} ./tile-rect-conv.mlir \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

tile-rect-conv-run:
	@${npu_OPT} ./tile-rect-conv.mlir -lower-gemmini | \
	${npu_TRANSLATE} --npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-matmul-lower:
	@${npu_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-matmul-translate:
	@${npu_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-linalg-matmul-asm:
	@${npu_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-matmul-run:
	@${npu_OPT} ./matmul.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nchw-fchw-i8-lower:
	@${npu_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-conv2d-nchw-fchw-i8-translate:
	@${npu_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nchw-fchw-i8-asm:
	@${npu_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nchw-fchw-i8-run:
	@${npu_OPT} ./conv_2d_nchw_fchw_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nchw-fchw-f32-lower:
	@${npu_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" \
		-o log.mlir

gemmini-linalg-conv2d-nchw-fchw-f32-translate:
	@${npu_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nchw-fchw-f32-asm:
	@${npu_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nchw-fchw-f32-run:
	@${npu_OPT} ./conv_2d_nchw_fchw_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nhwc-hwcf-i8-lower:
	@${npu_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-conv2d-nhwc-hwcf-i8-translate:
	@${npu_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nhwc-hwcf-i8-asm:
	@${npu_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nhwc-hwcf-i8-run:
	@${npu_OPT} ./conv_2d_nhwc_hwcf_i8.mlir \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-conv2d-nhwc-hwcf-f32-lower:
	@${npu_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" \
		-o log.mlir

gemmini-linalg-conv2d-nhwc-hwcf-f32-translate:
	@${npu_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-linalg-conv2d-nhwc-hwcf-f32-asm:
	@${npu_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-conv2d-nhwc-hwcf-f32-run:
	@${npu_OPT} ./conv_2d_nhwc_hwcf_f32.mlir \
		-convert-linalg-to-gemmini="acc_t=f32" \
		-convert-linalg-to-loops \
		-lower-gemmini="dim=4 acc_t=f32 elem_t=f32" | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

gemmini-linalg-batch-matmul-lower:
	@${npu_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini \
		-o log.mlir

gemmini-linalg-batch-matmul-translate:
	@${npu_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-linalg-batch-matmul-asm:
	@${npu_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-linalg-batch-matmul-run:
	@${npu_OPT} ./batch_matmul.mlir \
		-convert-linalg-to-gemmini \
		-expand-strided-metadata\
		-convert-linalg-to-loops \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-gcc log.o -O2 -static -o a.out
	@spike --extension=gemmini pk a.out

linalg-matmul-32x32-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini \
		-o log.mlir

linalg-matmul-32x32-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-32x32-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
		-llvm-request-c-wrappers \
		-convert-linalg-to-gemmini \
		-convert-linalg-to-loops  \
		-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-32x32-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-32x32-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=1 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-32x32-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-32x32-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-32x32-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-32x32-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-32x32-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-32x32-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-matmul-32x32-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-32x32-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=1 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-64x64-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-64x64-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-64x64-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-64x64-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-64x64-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=2 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-64x64-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-64x64-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-64x64-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-64x64-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-64x64-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-64x64-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-matmul-64x64-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-64x64-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=2 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-128x128-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-128x128-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-128x128-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-128x128-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-128x128-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=3 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-128x128-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-128x128-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-128x128-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-128x128-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-128x128-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-128x128-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-matmul-128x128-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-128x128-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=3 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-256x256-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-256x256-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-256x256-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-256x256-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-256x256-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=4 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-256x256-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-256x256-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-256x256-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-256x256-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-256x256-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-256x256-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-matmul-256x256-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-256x256-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=4 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-512x512-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-512x512-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-512x512-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-512x512-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-512x512-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=5 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-512x512-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-512x512-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-512x512-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-512x512-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-512x512-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-512x512-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-matmul-512x512-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-512x512-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=5 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-matmul-1024x1024-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-matmul-1024x1024-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-1024x1024-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-1024x1024-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 -DDIALECT=1 \
	performance-test.cpp -O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-matmul-1024x1024-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DMATMUL=6 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-matmul-1024x1024-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-matmul-1024x1024-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-matmul-1024x1024-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-matmul-1024x1024-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 \
	-DDIALECT=1 performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-matmul-1024x1024-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-matmul-1024x1024-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-matmul-1024x1024-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-matmul-1024x1024-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DMATMUL=6 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-3x3-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-3x3-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-3x3-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-3x3-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-3x3-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=1 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-3x3-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-3x3-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-3x3-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-3x3-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-3x3-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-3x3-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-conv-3x3-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-conv-3x3-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=1 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-5x5-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-5x5-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-5x5-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-5x5-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-5x5-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=2 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-5x5-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-5x5-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-5x5-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-5x5-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-5x5-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-5x5-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-conv-5x5-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-conv-5x5-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=2 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-7x7-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-7x7-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-7x7-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-7x7-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-7x7-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=3 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-7x7-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-7x7-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-7x7-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-7x7-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-7x7-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-7x7-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-conv-7x7-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-conv-7x7-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=3 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-9x9-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-9x9-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-9x9-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-9x9-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-9x9-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=4 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-9x9-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-9x9-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-9x9-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-9x9-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-9x9-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-9x9-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-conv-9x9-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-conv-9x9-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=4 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

linalg-conv-11x11-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-11x11-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-11x11-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-11x11-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-11x11-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=5 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-11x11-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-11x11-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-11x11-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-11x11-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-11x11-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-11x11-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-conv-11x11-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-conv-11x11-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=5 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out


linalg-conv-13x13-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini \
	-o log.mlir

linalg-conv-13x13-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-13x13-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-13x13-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-gemmini \
	-convert-linalg-to-loops  \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

c-conv-13x13-gemmini-run:
	@riscv64-unknown-linux-gnu-gcc performance-test.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests  \
	-DCONV=6 -O2 -static
	@spike --extension=gemmini pk a.out

linalg-conv-13x13-cpu-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts \
	-o log.mlir

linalg-conv-13x13-cpu-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

linalg-conv-13x13-cpu-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

linalg-conv-13x13-cpu-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-affine -convert-scf-to-cf \
	-convert-vector-to-llvm -finalize-memref-to-llvm \
	-convert-arith-to-llvm \
	-lower-gemmini \
	-convert-func-to-llvm -reconcile-unrealized-casts | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 -DDIALECT=1 \
	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

gemmini-conv-13x13-gemmini-lower:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini \
	-o log.mlir

gemmini-conv-13x13-gemmini-translate:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir \
		-o log.ll

gemmini-conv-13x13-gemmini-asm:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=asm -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.s

gemmini-conv-13x13-gemmini-run:
	@${npu_OPT} ./ciface.mlir \
	-llvm-request-c-wrappers \
	-convert-linalg-to-loops \
	-lower-gemmini | \
	${npu_TRANSLATE} -npu-to-llvmir | \
	${npu_LLC} -filetype=obj -mtriple=riscv64 \
		-mattr=+npuext,+D -float-abi=hard \
		-o log.o
	@riscv64-unknown-linux-gnu-g++ log.o -DCONV=6 \
	-DDIALECT=2	performance-test.cpp \
	-O2 -static -o a.out -I${INTERFACES}
	@spike --extension=gemmini pk a.out

exo-matmul-4-run:
	@riscv64-unknown-linux-gnu-gcc ./exo-matmul-4.c \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests/include \
	-I${RISCV}/../../generators/gemmini/software/gemmini-rocc-tests \
	-O2 -static -o a.out
	@spike --extension=gemmini pk a.out
