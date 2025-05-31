try:
    import triton
    import triton.language as tl
    import torch
    use_triton = True
except:
    Warning("triton is not installed. ")
    use_triton = False

if use_triton:
    TORCH_DTYPES_TO_TL_DTYPES = {
        torch.float16: tl.float16,
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
    }

    def split_block_dims(blocks, *args):
        n_max = blocks.shape[1]
        dims = list(args) + [n_max - sum(args)]
        return torch.split(blocks, dims, dim=1)

    @triton.jit
    def dequant_Q4_0_kernel(
        scale_ptr,
        blocks_ptr,
        out_ptr,
        n_blocks,
        BLOCK_SIZE: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= n_blocks:
            return

        scale = tl.load(scale_ptr + pid)
        qs_u8 = tl.load(blocks_ptr + pid * BLOCK_SIZE //
                        2 + tl.arange(0, BLOCK_SIZE // 2))

        low = (qs_u8 & 0x0F).to(tl.int8) - 8
        high = ((qs_u8 >> 4) & 0x0F).to(tl.int8) - 8

        tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(0,
                 BLOCK_SIZE // 2), scale * low.to(OUT_DTYPE))
        tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(BLOCK_SIZE //
                 2, BLOCK_SIZE), scale * high.to(OUT_DTYPE))

    def dequantize_blocks_Q4_0_triton(
        blocks: torch.ByteTensor,
        block_size: int = 32,
        type_size=None,
        dtype=torch.float16
    ):
        n_blocks = blocks.shape[0]

        out = torch.empty((n_blocks, block_size),
                          dtype=dtype, device=blocks.device)
        grid = (n_blocks,)                  # = launch 1 kernel per block

        out_dtype = TORCH_DTYPES_TO_TL_DTYPES.get(dtype, tl.float16)

        d, qs = split_block_dims(blocks, 2)
        d = d.view(torch.float16).to(dtype)

        dequant_Q4_0_kernel[grid](
            scale_ptr=d.contiguous(),
            blocks_ptr=qs.contiguous(),
            out_ptr=out,
            n_blocks=n_blocks,
            BLOCK_SIZE=block_size,
            OUT_DTYPE=out_dtype,
        )

        return out

    @triton.jit
    def dequant_Q4_1_kernel(
        scale_ptr,
        mins_ptr,
        blocks_ptr,
        out_ptr,
        n_blocks,
        BLOCK_SIZE: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= n_blocks:
            return

        scale = tl.load(scale_ptr + pid)
        min = tl.load(mins_ptr + pid)
        qs_u8 = tl.load(blocks_ptr + pid * BLOCK_SIZE //
                        2 + tl.arange(0, BLOCK_SIZE // 2))

        low = (qs_u8 & 0x0F)
        high = ((qs_u8 >> 4) & 0x0F)

        tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(0,
                 BLOCK_SIZE // 2), scale * low.to(OUT_DTYPE) + min)
        tl.store(out_ptr + pid * BLOCK_SIZE + tl.arange(BLOCK_SIZE //
                 2, BLOCK_SIZE), scale * high.to(OUT_DTYPE) + min)

    def dequantize_blocks_Q4_1_triton(
        blocks: torch.ByteTensor,
        block_size: int = 32,
        type_size=None,
        dtype=torch.float16
    ):
        n_blocks = blocks.shape[0]

        out = torch.empty((n_blocks, block_size),
                          dtype=dtype, device=blocks.device)

        grid = (n_blocks,)

        out_dtype = TORCH_DTYPES_TO_TL_DTYPES.get(dtype, tl.float16)

        d, m, qs = split_block_dims(blocks, 2, 2)
        d = d.view(torch.float16).to(dtype)
        m = m.view(torch.float16).to(dtype)

        dequant_Q4_1_kernel[grid](
            scale_ptr=d.contiguous(),
            mins_ptr=m.contiguous(),
            blocks_ptr=qs.contiguous(),
            out_ptr=out,
            n_blocks=n_blocks,
            BLOCK_SIZE=block_size,
            OUT_DTYPE=out_dtype,
        )

        return out

    @triton.jit
    def get_scale_min(
        d_ptr,
        dmin_ptr,
        scales_ptr,
        out_d_ptr,
        out_dmin_ptr,
        n_blocks,
    ):
        pid = tl.program_id(0)
        if pid >= n_blocks:
            return

        d = tl.load(d_ptr + pid)
        dmin = tl.load(dmin_ptr + pid)

        scale_a = tl.load(scales_ptr + pid * 12 + tl.arange(0, 4))
        scale_b = tl.load(scales_ptr + pid * 12 + 4 + tl.arange(0, 4))
        scale_c = tl.load(scales_ptr + pid * 12 + 8 + tl.arange(0, 4))

        scale_1 = (scale_a & 0x3F).to(tl.float16) * d
        scale_2 = ((scale_c & 0x0F) | (
            (scale_a >> 2) & 0x30)).to(tl.float16) * d
        min_1 = (scale_b & 0x3F).to(tl.float16) * dmin
        min_2 = ((scale_c >> 4) | ((scale_b >> 2) & 0x30)).to(
            tl.float16) * dmin

        tl.store(out_d_ptr + pid * 8 + tl.arange(0, 4), scale_1)
        tl.store(out_d_ptr + pid * 8 + 4 + tl.arange(0, 4), scale_2)
        tl.store(out_dmin_ptr + pid * 8 + tl.arange(0, 4), min_1)
        tl.store(out_dmin_ptr + pid * 8 + 4 + tl.arange(0, 4), min_2)

    @triton.jit
    def dequant_Q4_K_kernel(
        scale_ptr,
        mins_ptr,
        blocks_ptr,
        out_ptr,
        n_blocks,
        BLOCK_SIZE: tl.constexpr,
        OUT_DTYPE: tl.constexpr,
    ):
        pid_x = tl.program_id(0)
        pid_y = tl.program_id(1)

        if pid_x >= n_blocks or pid_y >= 4:
            return

        scale_low = tl.load(scale_ptr + pid_x * 8 + pid_y * 2)
        scale_high = tl.load(scale_ptr + pid_x * 8 + pid_y * 2 + 1)
        min_low = tl.load(mins_ptr + pid_x * 8 + pid_y * 2)
        min_high = tl.load(mins_ptr + pid_x * 8 + pid_y * 2 + 1)

        offset = pid_x * 128 + pid_y * BLOCK_SIZE
        qs_u8 = tl.load(blocks_ptr + offset + tl.arange(0, BLOCK_SIZE))

        low = (qs_u8 & 0x0F)
        high = ((qs_u8 >> 4) & 0x0F)

        out_offset = pid_x * 256 + pid_y * BLOCK_SIZE * 2
        tl.store(out_ptr + out_offset + tl.arange(0, BLOCK_SIZE),
                 scale_low * low.to(OUT_DTYPE) - min_low)
        tl.store(out_ptr + out_offset + BLOCK_SIZE + tl.arange(0,
                 BLOCK_SIZE), scale_high * high.to(OUT_DTYPE) - min_high)

    def dequantize_blocks_Q4_K_triton(
        blocks: torch.ByteTensor,
        block_size: int = 32,
        type_size=None,
        dtype=torch.float16
    ):
        n_blocks = blocks.shape[0]
        block_size = 32

        out = torch.empty((n_blocks, block_size * 8),
                          dtype=dtype, device=blocks.device)

        out_dtype = TORCH_DTYPES_TO_TL_DTYPES.get(dtype, tl.float16)

        d, dmin, scales, qs = split_block_dims(blocks, 2, 2, 12)
        d = d.view(torch.float16)
        dmin = dmin.view(torch.float16)
        scales = scales.view(torch.uint8)

        d_scales = torch.empty(
            (n_blocks, 8), dtype=torch.float16, device=blocks.device)
        d_mins = torch.empty(
            (n_blocks, 8), dtype=torch.float16, device=blocks.device)

        get_scale_min[(n_blocks, )](
            d_ptr=d.contiguous(),
            dmin_ptr=dmin.contiguous(),
            scales_ptr=scales.contiguous(),
            out_d_ptr=d_scales,
            out_dmin_ptr=d_mins,
            n_blocks=n_blocks,
        )

        dequant_Q4_K_kernel[(n_blocks, 4)](
            scale_ptr=d_scales,
            mins_ptr=d_mins,
            blocks_ptr=qs.contiguous(),
            out_ptr=out,
            n_blocks=n_blocks,
            BLOCK_SIZE=block_size,
            OUT_DTYPE=out_dtype,
        )

        return out

    @triton.jit
    def dequant_Q6_K_kernel(
        scale_ptr, d_ptr, ql_ptr, qh_ptr, out_ptr,
        n_blocks: tl.constexpr,
        BLOCK_SIZE: tl.constexpr = 256,
        OUT_DTYPE: tl.constexpr = tl.float16,
    ):
        pid = tl.program_id(0)
        if pid >= n_blocks:
            return
        BYTES_L = BLOCK_SIZE // 2
        BYTES_H = BLOCK_SIZE // 4

        ql_off  = pid * BYTES_L
        qh_off  = pid * BYTES_H
        out_off = pid * BLOCK_SIZE
        sc_off  = pid * 16

        idx256  = tl.arange(0, BLOCK_SIZE)
        row16   = idx256 // 16

        byte_ql = (idx256 // 128) * 64 + (idx256 % 64)
        sh_ql   = ((idx256 % 128) // 64) * 4
        ql_bytes = tl.load(ql_ptr + ql_off + byte_ql, cache_modifier='.cg')
        ql_nib   = (ql_bytes >> sh_ql) & 0x0F

        byte_qh = (idx256 // 128) * 32 + (idx256 % 32)
        sh_qh   = ((idx256 % 128) // 32) * 2
        qh_bytes = tl.load(qh_ptr + qh_off + byte_qh, cache_modifier='.cg')
        qh_bits  = (qh_bytes >> sh_qh) & 0x03

        q_i8 = (ql_nib | (qh_bits << 4)).to(tl.int8) - 32

        scale_row = tl.load(scale_ptr + sc_off + row16)
        d_scalar  = tl.load(d_ptr + pid)
        scale_f   = scale_row.to(OUT_DTYPE) * d_scalar.to(OUT_DTYPE)

        out = q_i8.to(OUT_DTYPE) * scale_f
        tl.store(out_ptr + out_off + idx256, out)

    def dequantize_blocks_Q6_K_triton(
        blocks: torch.ByteTensor,
        block_size: int = 32,
        type_size=None,
        dtype=torch.float16
    ):
        QK_K = 256
        n_blocks = blocks.shape[0]

        out = torch.empty((n_blocks, QK_K),
                          dtype=dtype, device=blocks.device)

        ql, qh, scales, d, = split_block_dims(
            blocks, QK_K // 2, QK_K // 4, QK_K // 16)
        scales = scales.view(torch.int8)
        d = d.view(torch.float16)

        out_dtype = TORCH_DTYPES_TO_TL_DTYPES.get(dtype, tl.float16)

        dequant_Q6_K_kernel[(n_blocks, )](
            scale_ptr=scales.contiguous(),
            d_ptr=d.contiguous(),
            ql_ptr=ql.contiguous(),
            qh_ptr=qh.contiguous(),
            out_ptr=out,
            n_blocks=n_blocks,
            BLOCK_SIZE=QK_K,
            OUT_DTYPE=out_dtype,
        )

        return out
