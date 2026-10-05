#include "ggml-q8arena.h"

#include "../../vendor/nlohmann/json.hpp"
#include "ggml-impl.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

struct arena_counters {
    std::atomic<uint64_t> count[8]{}, logical[8]{}, physical[8]{};

    ~arena_counters() {
        if (!ggml_q8arena_enabled()) {
            return;
        }
        for (int i = 0; i < 8; i++) {
            if (count[i]) {
                fprintf(stderr, "q8arena_stats kind=%d calls=%llu logical=%llu physical=%llu\n", i,
                        (unsigned long long) count[i].load(), (unsigned long long) logical[i].load(),
                        (unsigned long long) physical[i].load());
            }
        }
    }
};

void ggml_q8arena_account(int kind, uint64_t logical, uint64_t physical) {
    static arena_counters counters;
    GGML_ASSERT(kind >= 0 && kind < 8);
    counters.count[kind]++;
    counters.logical[kind] += logical;
    counters.physical[kind] += physical;
}

static size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

bool ggml_q8arena_enabled() {
    return std::getenv("GGML_Q8_ARENA") != nullptr;
}

static const std::map<std::string, ggml_q8arena_plan> & plans() {
    static const auto all = [] {
        std::map<std::string, ggml_q8arena_plan> result;
        const char *                             path = std::getenv("GGML_Q8_ARENA");
        if (!path) {
            return result;
        }
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            throw std::runtime_error("Q8 arena cannot open index file");
        }
        auto size = f.tellg();
        if (size < 24) {
            throw std::runtime_error("Q8 arena truncated");
        }
        uint64_t n;
        char     magic[8];
        f.seekg(size - std::streamoff(16));
        f.read((char *) &n, 8);
        f.read(magic, 8);
        if ((memcmp(magic, "Q8ARENA2", 8) && memcmp(magic, "Q8INDEX1", 8)) || n > uint64_t(size) - 24) {
            throw std::runtime_error("Q8 arena invalid footer");
        }
        f.seekg(size - std::streamoff(16 + n));
        std::string json(n, '\0');
        f.read(json.data(), n);
        auto root = nlohmann::json::parse(json);
        for (const auto & r : root.at("tensors")) {
            if (r.at("kind") != "q8_dictionary") {
                continue;
            }
            ggml_q8arena_plan p{};
            p.bits = r.at("bits");
            p.cols = r.at("shape").at(0);
            p.rows = r.at("shape").at(1);
            GGML_ASSERT(p.bits > 0 && p.bits <= 16 && p.cols % 32 == 0 && p.rows > 0);
            p.table.resize(r.at("table_count"));
            f.seekg(r.at("table_offset").get<uint64_t>());
            f.read((char *) p.table.data(), p.table.size() * 2);
            if (!f) {
                throw std::runtime_error("Q8 arena dictionary read failed");
            }
            // A fixed dictionary band makes the code address immediate in the
            // native CPU/CUDA inner loops. This target needs at most 11 bits.
            GGML_ASSERT(p.table.size() <= 2048 && p.bits <= 11);
            p.codes_offset       = 4352;
            p.indices_offset     = align_up(p.codes_offset + p.rows * p.cols, 64);
            size_t indices_bytes = align_up((p.rows * (p.cols / 32) * p.bits + 7) / 8, 4) + 4;
            p.allocation         = align_up(p.indices_offset + indices_bytes, 256);
            result.emplace(r.at("name").get<std::string>(), std::move(p));
        }
        GGML_LOG_INFO("q8arena: loaded %zu exact scale dictionaries from %s\n", result.size(), path);
        return result;
    }();
    return all;
}

const ggml_q8arena_plan * ggml_q8arena_find(const ggml_tensor * t) {
    if (!t || t->type != GGML_TYPE_Q8_0 || t->ne[2] != 1 || t->ne[3] != 1 || !ggml_q8arena_enabled()) {
        return nullptr;
    }
    if (t->view_src) {
        // The scheduler wraps immutable weights in zero-offset, same-layout
        // aliases before transfer. Their arena identity and backing are shared.
        if (t->view_offs == 0 && ggml_are_same_layout(t, t->view_src)) {
            return ggml_q8arena_find(t->view_src);
        }
        return nullptr;
    }
    auto &      m     = plans();
    std::string name  = t->name;
    // Scheduler copies retain the original identity between backend# and #copy.
    auto        first = name.find('#'), last = name.rfind('#');
    if (first != std::string::npos && last > first) {
        name = name.substr(first + 1, last - first - 1);
    }
    const std::string suffix = " (view)";
    while (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        name.resize(name.size() - suffix.size());
    }
    auto it = m.find(name);
    if (it == m.end()) {
        return nullptr;
    }
    const auto & p = it->second;
    GGML_ASSERT(p.rows == t->ne[1] && p.cols == t->ne[0]);
    return &p;
}

size_t ggml_q8arena_size(const ggml_tensor * t) {
    auto p = ggml_q8arena_find(t);
    return p ? p->allocation : ggml_nbytes(t);
}

void ggml_q8arena_pack(const ggml_q8arena_plan * p, const void * raw, void * out) {
    auto     src = (const uint8_t *) raw;
    auto     dst = (uint8_t *) out;
    uint32_t lookup[65536];
    std::fill(lookup, lookup + 65536, UINT32_MAX);
    for (size_t i = 0; i < p->table.size(); i++) {
        lookup[p->table[i]] = i;
    }
    memset(dst, 0, p->codes_offset);
    memset(dst + p->indices_offset, 0, p->allocation - p->indices_offset);
    memcpy(dst + 64, p->table.data(), p->table.size() * 2);
    uint64_t header[2] = { p->indices_offset, p->codes_offset };
    memcpy(dst, header, 16);
    uint32_t bits = p->bits;
    memcpy(dst + 16, &bits, 4);
    auto   indices = (uint32_t *) (dst + p->indices_offset);
    auto   codes   = dst + p->codes_offset;
    size_t count   = p->rows * (p->cols / 32);
    for (size_t b = 0; b < count; b++) {
        uint16_t d;
        memcpy(&d, src + 34 * b, 2);
        uint32_t id = lookup[d];
        GGML_ASSERT(id != UINT32_MAX);
        uint64_t bit = b * p->bits;
        indices[bit >> 5] |= id << (bit & 31);
        if ((bit & 31) + p->bits > 32) {
            indices[(bit >> 5) + 1] |= id >> (32 - (bit & 31));
        }
        memcpy(codes + 32 * b, src + 34 * b + 2, 32);
    }
}

uint16_t ggml_q8arena_scale(const ggml_q8arena_plan * p, const void * src, size_t b) {
    auto     data = (const uint8_t *) src;
    uint64_t bit  = b * p->bits, word;
    memcpy(&word, data + p->indices_offset + (bit >> 5) * 4, 8);
    uint32_t id = (word >> (bit & 31)) & ((1u << p->bits) - 1);
    return ((const uint16_t *) (data + 64))[id];
}

void ggml_q8arena_unpack(const ggml_q8arena_plan * p, const void * src, void * out, size_t first, size_t count) {
    GGML_ASSERT(first + count <= size_t(p->rows * (p->cols / 32)));
    auto data = (const uint8_t *) src;
    auto dst  = (uint8_t *) out;
    for (size_t b = 0; b < count; b++) {
        uint16_t d = ggml_q8arena_scale(p, src, first + b);
        memcpy(dst + b * 34, &d, 2);
        memcpy(dst + b * 34 + 2, data + p->codes_offset + (first + b) * 32, 32);
    }
}
