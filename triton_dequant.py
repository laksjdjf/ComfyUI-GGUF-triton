try:
    import triton
    import triton.language as tl
    import torch
    use_triton = True
except:
    Warning("triton is not installed. ")
    use_triton = False

if use_triton:
    def split_block_dims(blocks, *args):
        n_max = blocks.shape[1]
        dims = list(args) + [n_max - sum(args)]
        return torch.split(blocks, dims, dim=1)

    @triton.jit
    def dequant_q4_0_kernel(
        scale_ptr,
        blocks_ptr,          # uint8*
        out_ptr,             # OUT_DTYPE*
        n_blocks,            # int32
        BLOCK_SIZE: tl.constexpr,      # 32
        OUT_DTYPE: tl.constexpr,       # tl.float16 / tl.bfloat16 / tl.float32
    ):
        pid = tl.program_id(0)
        if pid >= n_blocks:
            return

        scale = tl.load(scale_ptr + pid)
        qs_u8 = tl.load(blocks_ptr + pid * BLOCK_SIZE // 2 + tl.arange(0, BLOCK_SIZE // 2))

        low = (qs_u8 & 0x0F).to(tl.int8) - 8   # index 0,2,4,...
        high = ((qs_u8 >> 4) & 0x0F).to(tl.int8) - 8  # index 1,3,5,...

        tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE // 2), scale * low.to(OUT_DTYPE))
        tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(BLOCK_SIZE // 2, BLOCK_SIZE), scale * high.to(OUT_DTYPE))

    def dequantize_blocks_q4_0_triton(
        blocks: torch.ByteTensor,
        block_size: int = 32,
        type_size=None,
        dtype=torch.float16
    ):
        """
        blocks: (n_blocks, 18)  uint8
        return: (n_blocks, 32)  dtype
        """
        assert block_size == 32, "Q4_0 は固定 32 元"
        assert blocks.stride(-1) == 1 and blocks.dtype == torch.uint8
        n_blocks = blocks.shape[0]

        out = torch.empty((n_blocks, block_size),
                          dtype=dtype, device=blocks.device)
        grid = (n_blocks,)                  # = launch 1 kernel per block

        if dtype is torch.float16:
            out_dtype = tl.float16
        elif dtype is torch.float32:
            out_dtype = tl.float32
        elif dtype is torch.bfloat16:
            out_dtype = tl.bfloat16
        elif dtype is None:
            out_dtype = tl.float16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        d, qs = split_block_dims(blocks, 2)
        d  = d.view(torch.float16).to(dtype)

        dequant_q4_0_kernel[grid](
            scale_ptr=d.contiguous(),
            blocks_ptr=qs.contiguous(),
            out_ptr=out,
            n_blocks=n_blocks,
            BLOCK_SIZE=block_size,
            OUT_DTYPE=out_dtype,
        )

        return out
