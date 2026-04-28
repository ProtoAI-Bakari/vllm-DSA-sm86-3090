#!/usr/bin/env python3
# --ProtoAI-Bakari--
# Generate deterministic fixture tensors for L1 unit tests, plus their expected
# outputs from the CPU pytorch reference (cpu_reference.py).
#
# Run once per kernel after the reference impl + sm_86 kernel signature stabilize:
#     python3 gen_fixtures.py [--out-dir ./fixtures] [--seed 0xDEADBEEF]
#
# Each fixture file is a torch pickle:
#     {"inputs": {...named tensors...}, "expected": <tensor or tuple>, "kwargs": {...}}
import argparse
import os
import pathlib
import sys
import torch

# Make `from tests.correctness import cpu_reference` importable
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from tests.correctness.cpu_reference import (
    reference_sparse_attn_indexer,
    reference_mla_decode,
    reference_compressor,
    reference_swa,
    reference_marlin_int4_gemm,
)


def gen_sparse_attn_indexer():
    B, H_q, T_q, D = 2, 8, 4, 128
    H_kv, T_k = 2, 256
    q = torch.randn(B, H_q, T_q, D, dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T_k, D, dtype=torch.bfloat16)
    top_k, block_size = 4, 64
    expected = reference_sparse_attn_indexer(q, k, top_k=top_k, block_size=block_size)
    return {"inputs": {"q": q, "k_cache": k}, "expected": expected,
            "kwargs": {"top_k": top_k, "block_size": block_size}}


def gen_mla_decode():
    B, H_q, D_qk, D_v = 2, 16, 128, 128
    block_size = 16
    n_blocks_per_seq = 4
    T_k = block_size * n_blocks_per_seq
    q = torch.randn(B, H_q, 1, D_qk, dtype=torch.bfloat16)
    kv = torch.randn(B, T_k, 2, D_v, dtype=torch.bfloat16)
    block_table = torch.arange(n_blocks_per_seq, dtype=torch.int32).expand(B, n_blocks_per_seq).contiguous()
    expected = reference_mla_decode(q, kv, block_table, block_size=block_size)
    return {"inputs": {"q": q, "kv_cache": kv, "block_table": block_table},
            "expected": expected, "kwargs": {"block_size": block_size}}


def gen_compressor():
    B, T, D, D_out = 2, 256, 128, 64
    x = torch.randn(B, T, D, dtype=torch.bfloat16)
    w = torch.randn(D_out, D, dtype=torch.bfloat16)
    block_size = 64
    expected = reference_compressor(x, w, block_size=block_size)
    return {"inputs": {"x": x, "weight": w}, "expected": expected,
            "kwargs": {"block_size": block_size}}


def gen_swa():
    B, H_q, T, D = 2, 8, 64, 64
    H_kv = 2
    q = torch.randn(B, H_q, T, D, dtype=torch.bfloat16)
    k = torch.randn(B, H_kv, T, D, dtype=torch.bfloat16)
    v = torch.randn(B, H_kv, T, D, dtype=torch.bfloat16)
    window = 16
    expected = reference_swa(q, k, v, window=window)
    return {"inputs": {"q": q, "k": k, "v": v}, "expected": expected,
            "kwargs": {"window": window}}


def gen_marlin_int4_gemm():
    M, K, N = 16, 256, 64
    group_size = 128
    x = torch.randn(M, K, dtype=torch.bfloat16)
    K_packed = K // 8
    w_packed = torch.randint(-(2**31), 2**31, (K_packed, N), dtype=torch.int32)
    n_groups = K // group_size
    scales = torch.randn(n_groups, N, dtype=torch.bfloat16) * 0.01
    expected = reference_marlin_int4_gemm(x, w_packed, scales, group_size=group_size)
    return {"inputs": {"x_bf16": x, "w_int4_packed": w_packed, "scales": scales},
            "expected": expected, "kwargs": {"group_size": group_size}}


GENERATORS = {
    "sparse_attn_indexer": gen_sparse_attn_indexer,
    "mla_decode": gen_mla_decode,
    "compressor": gen_compressor,
    "swa": gen_swa,
    "marlin_int4_gemm": gen_marlin_int4_gemm,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "fixtures"))
    ap.add_argument("--seed", type=lambda s: int(s, 0), default=0xDEADBEEF)
    ap.add_argument("--only", action="append", help="generate only these kernels (repeatable)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, gen in GENERATORS.items():
        if args.only and name not in args.only:
            continue
        torch.manual_seed(args.seed)        # same per-kernel seed for reproducibility
        fix = gen()
        fix["seed"] = args.seed
        path = out / f"{name}.pt"
        torch.save(fix, path)
        ex = fix["expected"]
        ex_shape = tuple(ex.shape) if hasattr(ex, "shape") else "tuple"
        print(f"  wrote {path} (expected.shape={ex_shape}, dtype={getattr(ex, 'dtype', '?')})")

    print(f"\ngenerated {len(GENERATORS) if not args.only else len(args.only)} fixtures into {out}")


if __name__ == "__main__":
    main()
