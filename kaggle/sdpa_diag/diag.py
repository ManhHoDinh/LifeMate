import warnings, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
warnings.simplefilter("always")
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability(), "arch list", torch.cuda.get_arch_list(), flush=True)
for dt in (torch.float16, torch.float32):
    for L in (512, 4096):
        q = torch.randn(1, 32, L, 128, device="cuda", dtype=dt, requires_grad=True)
        k = torch.randn(1, 32, L, 128, device="cuda", dtype=dt, requires_grad=True)
        v = torch.randn(1, 32, L, 128, device="cuda", dtype=dt, requires_grad=True)
        for be in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
            torch.cuda.reset_peak_memory_stats()
            with warnings.catch_warnings(record=True) as w:
                try:
                    with sdpa_kernel([be]):
                        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                        o.sum().backward()
                    res = f"OK peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB"
                except Exception as e:
                    res = f"FAIL {type(e).__name__}: {str(e)[:120]}"
            reasons = " | ".join(str(x.message)[:200] for x in w)
            print(f"{str(dt):14s} L={L:5d} {be.name:20s} {res}  {reasons}", flush=True)
