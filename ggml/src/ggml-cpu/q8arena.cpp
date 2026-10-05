#include "ggml-cpu.h"
#include "ggml-q8arena.h"
#include "ggml-quants.h"
#include "llamafile/sgemm.h"
#include "ops.h"
#include "traits.h"
#include "vec.h"

#include <algorithm>
#include <cstring>
#if defined(__AVX2__)
#    include <immintrin.h>
#endif

static inline uint16_t get_scale(const ggml_q8arena_plan * p, const void * src, size_t b) {
    const auto data = (const uint8_t *) src;
    uint64_t   bit  = b * p->bits, word;
    memcpy(&word, data + p->indices_offset + (bit >> 5) * 4, 8);
    return ((const uint16_t *) (data + 64))[(word >> (bit & 31)) & ((1u << p->bits) - 1)];
}

static ggml_backend_buffer_type_t arena_buft();

static bool is_arena(const ggml_tensor * t) {
    return t && t->buffer && t->buffer->buft == arena_buft();
}

class arena_traits : public ggml::cpu::tensor_traits {
    bool work_size(int nth, const ggml_tensor * op, size_t & size) override {
        if (op->op == GGML_OP_GET_ROWS) {
            size = 0;
            return true;
        }
        size = ggml_row_size(GGML_TYPE_Q8_0, ggml_nelements(op->src[1]));
        GGML_UNUSED(nth);
        return true;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        auto a = op->src[0];
        auto p = ggml_q8arena_find(a);
        GGML_ASSERT(p);
        if (op->op == GGML_OP_GET_ROWS) {
            if (params->ith == 0) {
                ggml_q8arena_account(6, 0, 0);
            }
            auto ids = op->src[1];
            GGML_ASSERT(ids->type == GGML_TYPE_I32 && ggml_is_contiguous(ids) && op->type == GGML_TYPE_F32);
            for (int64_t r = params->ith; r < ggml_nelements(ids); r += params->nth) {
                int row = ((const int32_t *) ids->data)[r];
                GGML_ASSERT(row >= 0 && row < p->rows);
                auto out   = (float *) ((char *) op->data + r * op->nb[1]);
                auto codes = (const int8_t *) a->data + p->codes_offset + size_t(row) * p->cols;
                for (int k = 0; k < p->cols; k += 32) {
                    float d = GGML_CPU_FP16_TO_FP32(get_scale(p, a->data, size_t(row) * (p->cols / 32) + k / 32));
                    for (int j = 0; j < 32; j++) {
                        out[k + j] = d * codes[k + j];
                    }
                }
            }
            return true;
        }
        GGML_ASSERT(op->op == GGML_OP_MUL_MAT);
        auto b = op->src[1];
        GGML_ASSERT(b->type == GGML_TYPE_F32 && b->nb[0] == 4 && a->ne[2] == 1 && a->ne[3] == 1);
        int64_t columns = ggml_nrows(b);
        size_t  qrow    = ggml_row_size(GGML_TYPE_Q8_0, p->cols);
        if (params->ith == 0) {
            ggml_q8arena_account(columns > 1 ? 3 : 2, 0, 0);
        }
        for (int64_t col = params->ith; col < columns; col += params->nth) {
            int64_t i1 = col % b->ne[1], i2 = (col / b->ne[1]) % b->ne[2], i3 = col / (b->ne[1] * b->ne[2]);
            auto    x = (const float *) ((const char *) b->data + i1 * b->nb[1] + i2 * b->nb[2] + i3 * b->nb[3]);
            ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float(x, (char *) params->wdata + col * qrow, p->cols);
        }
        ggml_barrier(params->threadpool);
        GGML_ASSERT(ggml_is_contiguous(op));
        bool ok = llamafile_sgemm_q8arena(params, p->rows, columns, p->cols / 32, a->data, p->cols / 32, params->wdata,
                                          p->cols / 32, op->data, p->rows);
        GGML_ASSERT(ok && "Q8 arena CPU backend requires AVX2");
        return true;
    }
};

static arena_traits traits;

static ggml_status init(ggml_backend_buffer_t, ggml_tensor * t) {
    if (!ggml_q8arena_find(t)) {
        GGML_ABORT("arena init: name=%s type=%s op=%s dims=%lld,%lld,%lld,%lld view=%s offset=%zu", t->name,
                   ggml_type_name(t->type), ggml_op_name(t->op), (long long) t->ne[0], (long long) t->ne[1],
                   (long long) t->ne[2], (long long) t->ne[3], t->view_src ? t->view_src->name : "none", t->view_offs);
    }
    t->extra = &traits;
    return GGML_STATUS_SUCCESS;
}

static void set(ggml_backend_buffer_t, ggml_tensor * t, const void * src, size_t off, size_t size) {
    GGML_ASSERT(off == 0 && size == ggml_nbytes(t));
    ggml_q8arena_pack(ggml_q8arena_find(t), src, t->data);
    ggml_q8arena_account(0, size, ggml_q8arena_size(t));
}

static void get(ggml_backend_buffer_t, const ggml_tensor * t, void * dst, size_t off, size_t size) {
    GGML_ASSERT(off % 34 == 0 && size % 34 == 0);
    ggml_q8arena_unpack(ggml_q8arena_find(t), t->data, dst, off / 34, size / 34);
}

static ggml_backend_buffer_t alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto b = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (!b) {
        return nullptr;
    }
    b->buft              = buft;
    b->iface.init_tensor = init;
    b->iface.set_tensor  = set;
    b->iface.get_tensor  = get;
    b->iface.cpy_tensor  = nullptr;
    return b;
}

class arena_extra : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        if (!ggml_q8arena_find(op->src[0])) {
            return false;
        }
        if (op->src[0]->buffer && op->src[0]->buffer->buft != arena_buft()) {
            return false;
        }
        return (op->op == GGML_OP_MUL_MAT && op->src[1]->type == GGML_TYPE_F32) || op->op == GGML_OP_GET_ROWS;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        return is_arena(op->src[0]) ? &traits : nullptr;
    }
};

static ggml_backend_buffer_type_t arena_buft() {
    static ggml_backend_buffer_type b = {
        { [](ggml_backend_buffer_type_t) { return "CPU_Q8_ARENA"; }, alloc, nullptr,
         [](ggml_backend_buffer_type_t) { return size_t(64); }, nullptr,
         [](ggml_backend_buffer_type_t, const ggml_tensor * t) { return ggml_q8arena_size(t); }, nullptr, nullptr },
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        new arena_extra()
    };
    return &b;
}

ggml_backend_buffer_type_t ggml_backend_cpu_q8arena_buffer_type() {
    return ggml_q8arena_enabled() ? arena_buft() : nullptr;
}
