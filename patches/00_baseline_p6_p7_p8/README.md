# patches/00_baseline_p6_p7_p8

The three Python-side patches that survived the 17-patch cascade and form the **starting baseline** for the rebuild:

- **P6** — `UE8M0_PRECONVERT` (Triton fp8e4nv emulation gate)
- **P7** — `UE8M0_SCALE_SHIM` (block-scale propagation)
- **P8** — `W8A8_BF16_FALLBACK` (BF16 fallback path for unsupported W8A8 ops)

All three are deb-installable cluster patches at `/repo/INSTALLERS/` on the ZCCX cluster. The new kernels in `csrc_patches/` apply ON TOP of these.

Source patches will be vendored into this directory in CC2's first commit so the build is self-contained.

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
