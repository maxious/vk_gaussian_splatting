SPCONV_ALGO = "auto"  # 'auto', 'implicit_gemm', 'native'
FLEX_GEMM_ALGO = "auto"  # 'auto', 'explicit_gemm', 'implicit_gemm', 'implicit_gemm_splitk', 'masked_implicit_gemm', 'masked_implicit_gemm_splitk', 'igemm_mma'
FLEX_GEMM_HASHMAP_RATIO = 2.0  # Ratio of hashmap size to input size
FLEX_GEMM_USE_MULTI_XPU = False  # Use MultiXPUSpconv for multi-GPU inference
