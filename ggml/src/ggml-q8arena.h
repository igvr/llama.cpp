#pragma once
#include "ggml.h"

#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
#    include <vector>

struct ggml_q8arena_plan {
    int                   bits;
    int64_t               rows, cols;
    size_t                indices_offset, codes_offset, allocation;
    std::vector<uint16_t> table;
};

extern "C" {
#else
struct ggml_q8arena_plan;
#endif
GGML_API bool                             ggml_q8arena_enabled(void);
GGML_API void                             ggml_q8arena_account(int kind, uint64_t logical, uint64_t physical);
GGML_API const struct ggml_q8arena_plan * ggml_q8arena_find(const struct ggml_tensor * t);
GGML_API size_t                           ggml_q8arena_size(const struct ggml_tensor * t);
GGML_API void     ggml_q8arena_pack(const struct ggml_q8arena_plan * p, const void * src, void * dst);
GGML_API void     ggml_q8arena_unpack(const struct ggml_q8arena_plan * p,
                                      const void *                     src,
                                      void *                           dst,
                                      size_t                           first_block,
                                      size_t                           blocks);
GGML_API uint16_t ggml_q8arena_scale(const struct ggml_q8arena_plan * p, const void * src, size_t block);
#ifdef __cplusplus
}
#endif
