#pragma once

// Compact pointers are explicitly tagged by the host dispatch. Ordinary Q8
// buffers are at least 2-byte aligned. All addressing here uses logical block
// indices, so the native reduction and stream-K schedule remain unchanged.
static __device__ __forceinline__ bool q8_arena_tagged(const void * p) {
    return uintptr_t(p) & 1;
}

static __device__ const __forceinline__ char * q8_arena_base(const void * p) {
    return (const char *) (uintptr_t(p) & ~uintptr_t(31));
}

static __device__ __forceinline__ int q8_arena_word(const void * p, int block, int word) {
    auto               base   = q8_arena_base(p);
    constexpr uint64_t offset = 4352;
    return ((const int *) (base + offset))[size_t(block) * 8 + word];
}

static __device__ __forceinline__ half q8_arena_half(const void * p, int block) {
    auto     base   = q8_arena_base(p);
    auto     offset = ((const uint64_t *) base)[0];
    unsigned bits   = ((uintptr_t(p) >> 1) & 15) + 1;
    auto     idx    = (const uint32_t *) (base + offset);
    uint64_t bit    = uint64_t(block) * bits;
    uint32_t id     = __funnelshift_r(idx[bit >> 5], idx[(bit >> 5) + 1], bit & 31) & ((1u << bits) - 1);
    return __ushort_as_half(((const uint16_t *) (base + 64))[id]);
}
