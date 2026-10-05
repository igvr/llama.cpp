#pragma once
#include "common.cuh"
#include "ggml-q8arena.h"

inline bool ggml_cuda_is_q8arena(const ggml_tensor * t) {
    auto p = ggml_q8arena_find(t);
    return p && t->extra == p;
}

void ggml_cuda_q8arena(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

inline const void * ggml_cuda_q8arena_tag(const ggml_tensor * t) {
    auto p = ggml_q8arena_find(t);
    return (const void *) (uintptr_t(t->data) | ((p && t->extra == p) ? ((p->bits - 1) << 1) | 1 : 0));
}
