    .text
    .p2align 2
    .globl fvela_rope_runtime
    .type fvela_rope_runtime, @function

# void fvela_rope_runtime(const int16_t *input, int16_t *output,
#                         uint64_t config)
#
# a0: input pointer
# a1: output pointer
# a2: packed scalar configuration, (m << 16) | idx
fvela_rope_runtime:
    # Enable vector state for the bare-metal M-mode test environment.
    li      t0, 0x1e00
    csrs    mstatus, t0

    # Process eight 16-bit elements in one vector register group.
    li      t0, 8
    vsetvli zero, t0, e16, m1, ta, ma
    vle16.v v2, (a0)

    # The raw instruction encodes vfrope.fvx v3, v2, gp. Preserve gp because
    # it is a fixed register in the RISC-V ABI.
    mv      t4, gp
    mv      gp, a2
    .word   0x4a21c1d7
    mv      gp, t4

    vse16.v v3, (a1)
    ret

    .size fvela_rope_runtime, .-fvela_rope_runtime

    .section .note.GNU-stack,"",@progbits
