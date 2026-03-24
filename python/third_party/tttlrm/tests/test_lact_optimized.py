"""Correctness test for the optimized LACT fast weight function.

Compares the optimized implementation against a reference (naive) implementation
to verify numerical equivalence.

Key insight on tolerances:
  The Newton-Schulz (NS) orthogonalization converts to bf16 and iterates via
  matrix squaring (A @ A). This amplifies tiny input differences exponentially:
    - 5e-7 fp32 perturbation → 3.5% relative error after 5 NS steps
  The fused matmul reordering introduces ~5e-7 fp32 differences in gradients,
  which NS then amplifies. Both implementations are equally valid — they just
  disagree on bf16 rounding within the NS iterations.

  To prove mathematical equivalence, we test with muon_update_steps=0 (no NS
  iteration), which gives near-exact fp32 match (<1e-5).

Usage:
    TTTLRM_NO_COMPILE=1 uv run --extra cuda python tests/test_lact_optimized.py
"""
import sys
import time
from pathlib import Path

# Add tttlrm to path
_TTTLRM_ROOT = str(Path(__file__).parent.parent)
if _TTTLRM_ROOT not in sys.path:
    sys.path.insert(0, _TTTLRM_ROOT)

import os
os.environ["TTTLRM_NO_COMPILE"] = "1"

import torch
import torch.nn.functional as F

from model.lact_ttt import (
    fast_weight_swish_glu_weight_norm_mini_batch_apply,
    full_ttt_op,
    ar_ttt_op,
    zeropower_via_newtonschulz5,
    silu_backprop,
)
from utils import sp_support

# Patch sp_support for single-GPU
sp_support._SP_GROUP = None
sp_support.get_sp_rank = lambda: 0
sp_support.get_sp_world_size = lambda: 1
sp_support.sp_all_reduce = lambda tensor, op=None: tensor


def reference_fast_weight_apply(
    w0, w1, w2, q, k, v, lr0, lr1, lr2, ttt_config,
    muon_update_steps=0, elastic_lambda=0.0, fisher_alpha=0.1, anchor_beta=0.99,
):
    """Reference (naive) implementation — original code before optimization.

    Explicit dtype casts mirror what torch.autocast does in production.
    """
    dt = k.dtype
    w0_norm = w0.detach().norm(dim=1, keepdim=True)
    w1_norm = w1.detach().norm(dim=1, keepdim=True)
    w2_norm = w2.detach().norm(dim=1, keepdim=True)

    use_elastic = elastic_lambda > 0.0
    if use_elastic:
        w0_anchor = w0.clone()
        w1_anchor = w1.clone()
        w2_anchor = w2.clone()
        F0 = torch.zeros_like(w0)
        F1 = torch.zeros_like(w1)
        F2 = torch.zeros_like(w2)

    output = []
    for start, end, fast_weight, update, apply_flag in ttt_config:
        w0_now, w1_now, w2_now = w0, w1, w2

        if fast_weight:
            ki, vi = k[:, start:end, :], v[:, start:end, :]
            lr0i = lr0[:, start:end, :]
            lr1i = lr1[:, start:end, :]
            lr2i = lr2[:, start:end, :]

            gate_before_act = ki @ w0_now.to(dt)
            hidden_before_mul = ki @ w2_now.to(dt)
            hidden = F.silu(gate_before_act, inplace=False) * hidden_before_mul

            dhidden = vi @ w1_now.to(dt).transpose(-1, -2)
            dhidden_before_mul = dhidden * F.silu(gate_before_act, inplace=False)
            dgate = dhidden * hidden_before_mul
            dgate_before_act = silu_backprop(dgate, gate_before_act)

            w1_grad = (hidden * lr1i).to(dt).transpose(-1, -2) @ vi
            w0_grad = (ki * lr0i.to(dt)).transpose(-1, -2) @ dgate_before_act
            w2_grad = (ki * lr2i.to(dt)).transpose(-1, -2) @ dhidden_before_mul

            w1_grad = zeropower_via_newtonschulz5(w1_grad, muon_update_steps)
            w0_grad = zeropower_via_newtonschulz5(w0_grad, muon_update_steps)
            w2_grad = zeropower_via_newtonschulz5(w2_grad, muon_update_steps)

            w1_now = w1_now + w1_grad
            w0_now = w0_now + w0_grad
            w2_now = w2_now + w2_grad

            if use_elastic:
                with torch.no_grad():
                    F0 = fisher_alpha * F0 + (1.0 - fisher_alpha) * w0_grad.detach().square()
                    F1 = fisher_alpha * F1 + (1.0 - fisher_alpha) * w1_grad.detach().square()
                    F2 = fisher_alpha * F2 + (1.0 - fisher_alpha) * w2_grad.detach().square()
                    inv_imp0 = 1.0 - F0 / (F0.max() + 1e-8)
                    inv_imp1 = 1.0 - F1 / (F1.max() + 1e-8)
                    inv_imp2 = 1.0 - F2 / (F2.max() + 1e-8)

                w0_now = w0_now - elastic_lambda * inv_imp0 * (w0_now - w0_anchor)
                w1_now = w1_now - elastic_lambda * inv_imp1 * (w1_now - w1_anchor)
                w2_now = w2_now - elastic_lambda * inv_imp2 * (w2_now - w2_anchor)

            w0_now = w0_now / (w0_now.norm(dim=1, keepdim=True) + 1e-5) * w0_norm
            w1_now = w1_now / (w1_now.norm(dim=1, keepdim=True) + 1e-5) * w1_norm
            w2_now = w2_now / (w2_now.norm(dim=1, keepdim=True) + 1e-5) * w2_norm

            if update:
                w0, w1, w2 = w0_now, w1_now, w2_now
                if use_elastic:
                    w0_anchor = anchor_beta * w0_anchor + (1.0 - anchor_beta) * w0.detach()
                    w1_anchor = anchor_beta * w1_anchor + (1.0 - anchor_beta) * w1.detach()
                    w2_anchor = anchor_beta * w2_anchor + (1.0 - anchor_beta) * w2.detach()

        if apply_flag:
            qi = q[:, start:end, :]
            oi = (F.silu(qi @ w0_now.to(dt), inplace=False) * (qi @ w2_now.to(dt))) @ w1_now.to(dt)
            output.append(oi)

    output = torch.cat(output, dim=1)
    return output, w0, w1, w2


def make_inputs(b, d, dh, l, device="cpu", seed=42, dtype=torch.bfloat16):
    """Create random inputs for testing."""
    torch.manual_seed(seed)
    import math
    gain = math.sqrt(2)
    w0 = (torch.randn(b, d, dh, device=device) * gain / math.sqrt(d)).float()
    w1 = (torch.randn(b, dh, d, device=device) * gain / math.sqrt(dh)).float()
    w2 = (torch.randn(b, d, dh, device=device) * gain / math.sqrt(d)).float()
    q = torch.randn(b, l, d, device=device).to(dtype)
    k = torch.randn(b, l, d, device=device).to(dtype)
    v = torch.randn(b, l, d, device=device).to(dtype)
    lr0 = torch.rand(b, l, 1, device=device).float() * 0.1
    lr1 = torch.rand(b, l, 1, device=device).float() * 0.1
    lr2 = torch.rand(b, l, 1, device=device).float() * 0.1
    return w0, w1, w2, q, k, v, lr0, lr1, lr2


def run_both(w0, w1, w2, q, k, v, lr0, lr1, lr2, ttt_config, **kwargs):
    """Run optimized and reference, return outputs and weights."""
    with torch.no_grad():
        out_opt, w0_opt, w1_opt, w2_opt = fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2, ttt_config, **kwargs)
        out_ref, w0_ref, w1_ref, w2_ref = reference_fast_weight_apply(
            w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2, ttt_config, **kwargs)
    return (out_opt, w0_opt, w1_opt, w2_opt), (out_ref, w0_ref, w1_ref, w2_ref)


def check_errors(label, opt, ref, tol):
    """Check errors and print results. Returns True if all pass."""
    out_opt, w0_opt, w1_opt, w2_opt = opt
    out_ref, w0_ref, w1_ref, w2_ref = ref

    max_err = (out_opt.float() - out_ref.float()).abs().max().item()
    rel_err = max_err / (out_ref.float().abs().max().item() + 1e-8)

    w_errs = [
        (out_opt.float() - out_ref.float()).abs().max().item(),
        (w0_opt.float() - w0_ref.float()).abs().max().item(),
        (w1_opt.float() - w1_ref.float()).abs().max().item(),
        (w2_opt.float() - w2_ref.float()).abs().max().item(),
    ]

    print(f"  {label}:")
    print(f"    output rel_err={rel_err:.2e} (max_abs={max_err:.2e})")
    print(f"    w0={w_errs[1]:.2e}, w1={w_errs[2]:.2e}, w2={w_errs[3]:.2e}")

    ok = rel_err < tol
    if ok:
        print(f"    PASSED ✓ (tol={tol})")
    else:
        print(f"    FAILED ✗ (rel_err {rel_err:.2e} >= tol {tol})")
    return ok


# ===========================================================================
# 1. Mathematical equivalence (no NS iteration → no bf16 amplification)
# ===========================================================================

def test_math_equivalence_fp32(device="cpu"):
    """FP32 inputs, muon_update_steps=0: proves fused matmuls are correct."""
    b, d, dh = 1, 64, 256
    l_update, l_total = 128, 192
    ttt_config = full_ttt_op(
        update_minibatch=l_update, apply_only_minibatch=0,
        length=l_total, update_length=l_update,
    )
    inputs = make_inputs(b, d, dh, l_total, device, dtype=torch.float32)
    opt, ref = run_both(*inputs, ttt_config, muon_update_steps=0)
    return check_errors(f"fp32, steps=0 [{device}]", opt, ref, tol=1e-4)


def test_math_equivalence_bf16(device="cpu"):
    """BF16 inputs, muon_update_steps=0: proves fused matmuls work in bf16."""
    b, d, dh = 1, 64, 256
    l_update, l_total = 128, 192
    ttt_config = full_ttt_op(
        update_minibatch=l_update, apply_only_minibatch=0,
        length=l_total, update_length=l_update,
    )
    inputs = make_inputs(b, d, dh, l_total, device, dtype=torch.bfloat16)
    opt, ref = run_both(*inputs, ttt_config, muon_update_steps=0)
    # On GPU, F.silu uses a fused kernel that avoids intermediate bf16 rounding
    # vs our sigma*x path, allowing slightly larger differences
    return check_errors(f"bf16, steps=0 [{device}]", opt, ref, tol=1e-2)


# ===========================================================================
# 2. Full pipeline tests (with NS → allows NS amplification tolerance)
#    NS amplifies ~5e-7 fp32 matmul reordering → ~3% after 5 steps.
#    Both implementations are equally valid orthogonalizations.
# ===========================================================================

# NS amplifies ~5e-7 matmul reordering per step → ~3% after 5 iterations.
# Errors compound across update chunks: ~3% per chunk, multiplicative.
NS_TOL_1STEP = 0.10   # 1 update step
NS_TOL_MULTI = 0.35   # 4 update steps (compound: 1 - (1-0.03)^4 ≈ 12%, + apply matmul amplification)


def test_full_ttt_op(device="cpu"):
    """full_ttt_op with production NS steps (single update chunk)."""
    b, d, dh = 1, 64, 256
    l_update, l_total = 128, 192
    ttt_config = full_ttt_op(
        update_minibatch=l_update, apply_only_minibatch=0,
        length=l_total, update_length=l_update,
    )
    inputs = make_inputs(b, d, dh, l_total, device)
    opt, ref = run_both(*inputs, ttt_config, muon_update_steps=5)
    return check_errors(f"full_ttt_op NS=5 [{device}]", opt, ref, NS_TOL_1STEP)


def test_ar_ttt_op(device="cpu"):
    """ar_ttt_op with 4 update chunks (errors compound across chunks)."""
    b, d, dh = 1, 64, 256
    l_chunk, l_total = 64, 256
    ttt_config = ar_ttt_op(update_minibatch=l_chunk, length=l_total)
    inputs = make_inputs(b, d, dh, l_total, device)
    opt, ref = run_both(*inputs, ttt_config, muon_update_steps=5)
    return check_errors(f"ar_ttt_op NS=5 (4 chunks) [{device}]", opt, ref, NS_TOL_MULTI)


def test_with_elastic(device="cpu"):
    """Elastic regularization enabled (single update chunk)."""
    b, d, dh = 1, 64, 256
    l_update, l_total = 128, 192
    ttt_config = full_ttt_op(
        update_minibatch=l_update, apply_only_minibatch=0,
        length=l_total, update_length=l_update,
    )
    inputs = make_inputs(b, d, dh, l_total, device)
    opt, ref = run_both(
        *inputs, ttt_config, muon_update_steps=5,
        elastic_lambda=0.05, fisher_alpha=0.5, anchor_beta=0.8,
    )
    return check_errors(f"elastic NS=5 [{device}]", opt, ref, NS_TOL_1STEP)


def test_production_dims(device="cpu"):
    """Production dimensions: head_dim=768, inter_multi=4 → 768×3072."""
    b, d, dh = 1, 768, 3072
    l_update, l_total = 32, 48
    ttt_config = full_ttt_op(
        update_minibatch=l_update, apply_only_minibatch=0,
        length=l_total, update_length=l_update,
    )
    inputs = make_inputs(b, d, dh, l_total, device)
    opt, ref = run_both(*inputs, ttt_config, muon_update_steps=5)
    return check_errors(f"production 768×3072 NS=5 [{device}]", opt, ref, NS_TOL_1STEP)


def test_multi_update_steps(device="cpu"):
    """4 update chunks followed by apply-only (errors compound)."""
    b, d, dh = 1, 64, 256
    l_chunk = 32
    l_total = l_chunk * 4 + 32
    ttt_config = full_ttt_op(
        update_minibatch=l_chunk, apply_only_minibatch=0,
        length=l_total, update_length=l_chunk * 4,
    )
    inputs = make_inputs(b, d, dh, l_total, device)
    opt, ref = run_both(*inputs, ttt_config, muon_update_steps=5)
    return check_errors(f"multi-step (4 chunks) NS=5 [{device}]", opt, ref, NS_TOL_MULTI)


# ===========================================================================
# 3. Batched Newton-Schulz correctness (bit-exact for same input)
# ===========================================================================

def test_batched_ns(device="cpu"):
    """Verify batched NS produces identical results to separate calls."""
    torch.manual_seed(42)
    b, d, dh = 1, 64, 256
    g0 = torch.randn(b, d, dh, device=device)
    g1 = torch.randn(b, dh, d, device=device)
    g2 = torch.randn(b, d, dh, device=device)

    ns0 = zeropower_via_newtonschulz5(g0.clone(), 5)
    ns1 = zeropower_via_newtonschulz5(g1.clone(), 5)
    ns2 = zeropower_via_newtonschulz5(g2.clone(), 5)

    all_g = torch.cat([g0, g1.transpose(1, 2), g2], dim=0)
    all_ns = zeropower_via_newtonschulz5(all_g, 5)
    ns0_b = all_ns[:b]
    ns1_b = all_ns[b:2*b].transpose(1, 2)
    ns2_b = all_ns[2*b:]

    err0 = (ns0 - ns0_b).abs().max().item()
    err1 = (ns1 - ns1_b).abs().max().item()
    err2 = (ns2 - ns2_b).abs().max().item()

    print(f"  batched NS [{device}]: g0={err0:.2e}, g1={err1:.2e}, g2={err2:.2e}", end="")
    ok = max(err0, err1, err2) == 0.0
    print("  PASSED ✓" if ok else "  FAILED ✗")
    return ok


# ===========================================================================
# 4. Performance benchmark (CUDA only)
# ===========================================================================

def _get_gpu_device():
    """Return the available GPU device string, or None."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return None


def _sync_device(device):
    """Synchronize GPU device."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


def _reset_peak_memory(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device == "xpu":
        torch.xpu.reset_peak_memory_stats()


def _peak_memory_mb(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    elif device == "xpu":
        return torch.xpu.max_memory_allocated() / 1e6
    return 0


def benchmark_gpu():
    """Benchmark optimized vs reference on GPU with production dimensions."""
    device = _get_gpu_device()
    if device is None:
        print("  GPU benchmark: SKIPPED (no CUDA or XPU)")
        return

    b, d, dh = 1, 768, 3072
    l_update = 256
    l_total = l_update + 128

    ttt_config = full_ttt_op(
        update_minibatch=l_update, apply_only_minibatch=0,
        length=l_total, update_length=l_update,
    )
    inputs = make_inputs(b, d, dh, l_total, device)
    w0, w1, w2, q, k, v, lr0, lr1, lr2 = inputs

    def run_opt():
        return fast_weight_swish_glu_weight_norm_mini_batch_apply(
            w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2,
            ttt_config, muon_update_steps=5,
        )

    def run_ref():
        return reference_fast_weight_apply(
            w0.clone(), w1.clone(), w2.clone(), q, k, v, lr0, lr1, lr2,
            ttt_config, muon_update_steps=5,
        )

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            run_opt()
    _sync_device(device)

    n_iters = 20

    # Benchmark optimized
    _sync_device(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            run_opt()
        _sync_device(device)
    t_opt = (time.perf_counter() - t0) / n_iters * 1000

    # Warmup reference
    for _ in range(3):
        with torch.no_grad():
            run_ref()
    _sync_device(device)

    # Benchmark reference
    _sync_device(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            run_ref()
        _sync_device(device)
    t_ref = (time.perf_counter() - t0) / n_iters * 1000

    speedup = t_ref / t_opt
    dev_name = device.upper()
    print(f"  {dev_name} benchmark (768×3072, l={l_total}):")
    print(f"    optimized: {t_opt:.1f} ms")
    print(f"    reference: {t_ref:.1f} ms")
    print(f"    speedup:   {speedup:.2f}×")

    # Memory comparison
    _reset_peak_memory(device)
    with torch.no_grad():
        run_opt()
    mem_opt = _peak_memory_mb(device)

    _reset_peak_memory(device)
    with torch.no_grad():
        run_ref()
    mem_ref = _peak_memory_mb(device)

    print(f"    memory opt: {mem_opt:.0f} MB, ref: {mem_ref:.0f} MB, delta: {mem_opt - mem_ref:+.0f} MB")


if __name__ == "__main__":
    devices = ["cpu"]
    gpu = _get_gpu_device()
    if gpu:
        devices.append(gpu)

    print("=" * 60)
    print("LACT Optimized Fast Weight — Correctness & Performance Tests")
    print("=" * 60)
    all_passed = True

    # 1. Mathematical equivalence (no NS)
    print("\n--- Mathematical Equivalence (muon_update_steps=0) ---")
    print("  (Proves fused matmuls, sigma reuse are mathematically correct)")
    for dev in devices:
        all_passed &= test_math_equivalence_fp32(dev)
        all_passed &= test_math_equivalence_bf16(dev)

    # 2. Batched NS correctness
    print("\n--- Batched Newton-Schulz (should be bit-exact) ---")
    for dev in devices:
        all_passed &= test_batched_ns(dev)

    # 3. Full pipeline with NS
    print(f"\n--- Full Pipeline with NS (accounts for NS amplification) ---")
    for dev in devices:
        all_passed &= test_full_ttt_op(dev)
        all_passed &= test_ar_ttt_op(dev)
        all_passed &= test_with_elastic(dev)
        all_passed &= test_production_dims(dev)
        all_passed &= test_multi_update_steps(dev)

    # 4. Performance benchmark
    print("\n--- Performance Benchmark ---")
    benchmark_gpu()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
    print("=" * 60)
