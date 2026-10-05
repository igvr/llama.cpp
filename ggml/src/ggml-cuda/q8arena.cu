#include "mmq.cuh"
#include "mmvq.cuh"
#include "q8arena.cuh"
#include "quantize.cuh"

struct arena_view {
    const uint16_t * table;
    const uint32_t * indices;
    const int8_t *   codes;
    int              bits, cols, rows;
};

static __device__ __forceinline__ uint16_t arena_scale(arena_view a, int b) {
    uint64_t bit = uint64_t(b) * a.bits;
    uint32_t v   = __funnelshift_r(a.indices[bit >> 5], a.indices[(bit >> 5) + 1], bit & 31);
    return a.table[v & ((1u << a.bits) - 1)];
}

static __global__ void arena_rows(arena_view a, const int32_t * ids, float * out, int count) {
    int k = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (k >= a.cols || row >= count) {
        return;
    }
    int src = ids[row];
    if (src < 0 || src >= a.rows) {
        return;
    }
    int b = src * (a.cols / 32) + k / 32;
    out[size_t(row) * a.cols + k] =
        __half2float(__ushort_as_half(arena_scale(a, b))) * a.codes[size_t(src) * a.cols + k];
}

void ggml_cuda_q8arena(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    auto a = dst->src[0];
    auto p = ggml_q8arena_find(a);
    GGML_ASSERT(p && a->extra == p);
    auto       base   = (const char *) a->data;
    arena_view view   = { (const uint16_t *) (base + 64),
                          (const uint32_t *) (base + p->indices_offset),
                          (const int8_t *) (base + p->codes_offset),
                          p->bits,
                          int(p->cols),
                          int(p->rows) };
    auto       stream = ctx.stream();
    if (dst->op == GGML_OP_GET_ROWS) {
        GGML_ASSERT(ggml_is_contiguous(dst->src[1]) && ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32);
        int count = ggml_nelements(dst->src[1]);
        arena_rows<<<dim3((p->cols + 255) / 256, count), 256, 0, stream>>>(view, (const int32_t *) dst->src[1]->data,
                                                                           (float *) dst->data, count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    GGML_ASSERT(dst->op == GGML_OP_MUL_MAT);
    auto x = dst->src[1];
    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == 4);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (ggml_cuda_should_use_mmvq(a->type, cc, x->ne[1])) {
        ggml_q8arena_account(4, 0, 0);
        ggml_cuda_mul_mat_vec_q(ctx, a, x, nullptr, dst);
    } else {
        GGML_ASSERT(ggml_cuda_should_use_mmq(a->type, cc, x->ne[1], 0));
        ggml_q8arena_account(5, 0, 0);
        ggml_cuda_mul_mat_q(ctx, a, x, nullptr, dst);
    }
}
