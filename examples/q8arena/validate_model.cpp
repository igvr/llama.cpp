#include "ggml-backend.h"
#include "json.hpp"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
using json       = nlohmann::json;
using clock_type = std::chrono::steady_clock;

static double seconds(clock_type::time_point t) {
    return std::chrono::duration<double>(clock_type::now() - t).count();
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: validate_model MODEL PREFIX record|replay TOKEN_JSON\n");
        return 2;
    }
    bool replay = std::string(argv[3]) == "replay";
    if (!replay && std::string(argv[3]) != "record") {
        return 2;
    }
    const int steps = 48;
    ggml_backend_load_all();
    llama_backend_init();
    auto mp         = llama_model_default_params();
    mp.n_gpu_layers = 32;
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode    = LLAMA_LAZY_MODE_OFF;
    auto model      = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        return 1;
    }
    auto cp            = llama_context_default_params();
    cp.n_ctx           = 2048;
    cp.n_batch         = 128;
    cp.n_ubatch        = 128;
    cp.n_threads       = 16;
    cp.n_threads_batch = 16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    auto ctx           = llama_init_from_model(model, cp);
    if (!ctx) {
        return 1;
    }
    auto                     vocab   = llama_model_get_vocab(model);
    uint32_t                 nv      = llama_vocab_n_tokens(vocab);
    std::vector<std::string> prompts = {
        "Calculate 37 * 48 and explain the arithmetic briefly.",
        "Write a Python function that merges two sorted lists without changing the inputs.",
        "Explain why the sky is blue in three concise sentences.", "Read this inventory and retain the quantities.\n"
    };
    for (int i = 0; i < 48; i++) {
        prompts[3] += "Bin " + std::to_string(i) + " contains " + std::to_string(i * 3 + 7) + " bolts.\n";
    }
    prompts[3] += "How many bolts are in bins 11 and 29 combined? Show the calculation.";
    json cases = json::array(), reference;
    if (replay) {
        std::ifstream f(argv[4]);
        f >> reference;
    }
    std::ofstream logits(std::string(argv[2]) + ".logits", std::ios::binary);
    uint32_t      header[4] = { 0x5141524c, uint32_t(prompts.size()), uint32_t(steps), nv };
    logits.write((char *) header, sizeof(header));
    double prefill_seconds = 0, decode_seconds = 0;
    int    prefill_tokens = 0;
    for (size_t c = 0; c < prompts.size(); c++) {
        llama_memory_clear(llama_get_memory(ctx), true);
        std::string              text = "<|im_start|>user\n" + prompts[c] + "<|im_end|>\n<|im_start|>assistant\n";
        int                      n    = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
        std::vector<llama_token> tokens(n);
        int actual = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
        if (actual != n) {
            throw std::runtime_error("tokenization failed");
        }
        if (replay && reference.at("cases").at(c).at("prompt_tokens") != tokens) {
            throw std::runtime_error("prompt mismatch");
        }
        auto start = clock_type::now();
        for (int off = 0; off < n; off += 128) {
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + off, std::min(128, n - off)))) {
                throw std::runtime_error("prefill failed");
            }
        }
        llama_synchronize(ctx);
        prefill_seconds += seconds(start);
        prefill_tokens += n;
        json item;
        item["prompt"]        = prompts[c];
        item["prompt_tokens"] = tokens;
        item["tokens"]        = json::array();
        std::string generated;
        for (int step = 0; step < steps; step++) {
            const float * out = llama_get_logits_ith(ctx, -1);
            if (!out) {
                throw std::runtime_error("no logits");
            }
            logits.write((const char *) out, nv * sizeof(float));
            llama_token next = std::max_element(out, out + nv) - out;
            item["greedy"].push_back(next);
            if (replay) {
                next = reference.at("cases").at(c).at("tokens").at(step).get<llama_token>();
            }
            item["tokens"].push_back(next);
            char buf[1024];
            int  len = llama_token_to_piece(vocab, next, buf, sizeof(buf), 0, true);
            if (len > 0) {
                generated.append(buf, len);
            }
            start = clock_type::now();
            if (llama_decode(ctx, llama_batch_get_one(&next, 1))) {
                throw std::runtime_error("decode failed");
            }
            llama_synchronize(ctx);
            decode_seconds += seconds(start);
        }
        item["generated"] = generated;
        cases.push_back(item);
        fprintf(stderr, "validated case %zu/%zu, %d prompt tokens\n", c + 1, prompts.size(), n);
    }
    logits.close();
    if (!logits) {
        throw std::runtime_error("logit write failed");
    }
    json result = {
        { "cases",                    cases                                   },
        { "vocab",                    nv                                      },
        { "steps_per_case",           steps                                   },
        { "prefill_tokens",           prefill_tokens                          },
        { "prefill_seconds",          prefill_seconds                         },
        { "decode_seconds",           decode_seconds                          },
        { "decode_tokens",            steps * prompts.size()                  },
        { "decode_tokens_per_second", steps * prompts.size() / decode_seconds },
        { "replay",                   replay                                  }
    };
    std::ofstream(std::string(argv[2]) + ".json") << result.dump(2) << "\n";
    if (!replay) {
        std::ofstream(argv[4]) << result.dump(2) << "\n";
    }
    printf("{\"decode_tokens_per_second\":%.6f,\"prefill_seconds\":%.6f,\"logit_vectors\":%zu}\n",
           steps * prompts.size() / decode_seconds, prefill_seconds, steps * prompts.size());
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
