#!/usr/bin/env python3
"""
為 Apple Silicon (M4 Max 64GB/128GB 統一記憶體) 自動套用 ComfyUI-HyperFlow-Flash 無損加速補丁：
在第 1 步將 GGUF Q4_K 權重解壓為 BF16 後，直接快取於記憶體中，
讓第 2~6 步及後續分段直接執行純 Metal BF16 矩陣乘法，消除重複位元解壓開銷，100% 不損失任何畫質。
"""

from pathlib import Path


def patch_hyperflow_gguf(comfy_dir: Path) -> None:
    gguf_io_path = comfy_dir / "custom_nodes" / "ComfyUI-HyperFlow-Flash" / "gguf_io.py"
    if not gguf_io_path.exists():
        return

    content = gguf_io_path.read_text(encoding="utf-8")
    if "CachedGGMLLinear" in content:
        print("✅ Apple Silicon GGUF 權重快取無損加速補丁已啟用")
        return

    target = """    def operations(self):
        ops = self.ops.GGMLOps()
        ops.Linear.dequant_dtype = None
        ops.Linear.patch_dtype = None
        return ops"""

    replacement = """    def operations(self):
        ops = self.ops.GGMLOps()
        ops.Linear.dequant_dtype = None
        ops.Linear.patch_dtype = None

        # On high-RAM systems (e.g. Apple Silicon 64GB/128GB unified memory), cache dequantized BF16 weights
        # after the first forward step so subsequent steps run pure Metal BF16 GEMM without re-dequantizing Q4_K.
        import os
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        enable_cache = os.environ.get("HYPERFLOW_CACHE_DEQUANT", "1" if total_ram_gb >= 60 else "0") == "1"
        if enable_cache:
            base_linear = ops.Linear

            class CachedGGMLLinear(base_linear):
                def forward_ggml_cast_weights(self, input):
                    if not getattr(self.weight, "patches", None):
                        cache = getattr(self, "_cached_dequant_wb", None)
                        if cache is not None:
                            w, b = cache
                            if w.device == input.device and w.dtype == input.dtype:
                                return torch.nn.functional.linear(input, w, b)
                        weight, bias = self.cast_bias_weight(input)
                        self._cached_dequant_wb = (weight, bias)
                        return torch.nn.functional.linear(input, weight, bias)
                    weight, bias = self.cast_bias_weight(input)
                    return torch.nn.functional.linear(input, weight, bias)

            ops.Linear = CachedGGMLLinear
        return ops"""

    if target in content:
        gguf_io_path.write_text(content.replace(target, replacement), encoding="utf-8")
        print("✅ 已自動套用 Apple Silicon GGUF 權重快取無損加速補丁！")


def patch_h3_corrector_mps(comfy_dir: Path) -> None:
    corrector_init = comfy_dir / "custom_nodes" / "ComfyUI-H3-Corrector" / "__init__.py"
    if not corrector_init.exists():
        return
    mps_unsafe = 'sample_sigmas.to(device="cpu", dtype=torch.float64)'
    content = corrector_init.read_text(encoding="utf-8")
    if mps_unsafe not in content:
        return
    corrector_init.write_text(content.replace(mps_unsafe, "sample_sigmas.cpu().to(torch.float64)"), encoding="utf-8")
    print("✅ Patched H3 corrector for MPS (float64 cast moved to CPU)")


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent
    patch_hyperflow_gguf(root_dir / "ComfyUI")
    patch_h3_corrector_mps(root_dir / "ComfyUI")
