# scripts

Operational shell + Python scripts. The two most important:

- `build_vllm_dsa_sm86.sh` — CC2's reproducible build script (clones upstream vLLM at pinned SHA, applies P6/P7/P8 + csrc_patches/, runs cmake with `-DTORCH_CUDA_ARCH_LIST="8.6"`, packages the wheel)
- `install_cluster.sh` — CC7's fanout (rsyncs venv tarball to all 8 nodes, installs at `/home/z/.venvs/.venv-vllm-dsa-sm86/`)

Plus auxiliary:
- `verify_install.sh` — per-node `python -c "import vllm; print(vllm.__version__)"` smoke test
- `verify_cluster_parity.sh` — `git rev-parse HEAD` per node before each L4 cycle, halts on drift
- `launch_dsv4_tp2_ep8.sh` — final deploy command (post-CC6 PASS)

— Authored by ProtoAI-Bakari, with Assistance by Claude Opus 4.7 (claude-opus-4-7) [1M ctx, max effort]
