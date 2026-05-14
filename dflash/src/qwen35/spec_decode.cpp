// spec_decode.cpp — Speculative decoding loop for qwen35 layer-split inference.

#include "spec_decode.h"

#include "internal.h"
#include "io_utils.h"
#include "graph_builders.h"
#include "draft_feature_mirror.h"
#include "feature_copy.h"
#include "peer_access.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace dflash27b {

bool run_target_layer_split_dflash_decode(
        std::vector<TargetLayerSplitShard> & shards,
        DraftWeights & draft_weights,
        ggml_backend_t draft_backend,
        int draft_gpu,
        DraftFeatureMirror & feature_ring,
        const std::vector<int32_t> & prompt,
        int n_gen,
        int last_tok,
        const char * out_path,
        int kq_stride_pad,
        int fa_window,
        int draft_ctx_max,
        int stream_fd,
        DFlashDraftIpcClient * remote_draft) {
    const bool use_remote_draft = remote_draft && remote_draft->active();
    if (shards.empty() || (!use_remote_draft && !feature_ring.target_feat)) return false;
    const int hidden = draft_weights.n_embd;
    const int q_len = draft_weights.block_size;
    const int output_gpu = shards.back().gpu;
    ggml_backend_t output_backend = shards.back().backend;

    StepGraph draft_sg;
    StepGraph proj_sg;
    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<int32_t> out_all = prompt;
    int committed = (int)prompt.size();
    int n_generated = 0;
    int n_draft_steps = 0;
    int n_accept_sum = 0;

    auto sync_all = [&]() {
        for (auto & shard : shards) ggml_backend_synchronize(shard.backend);
        if (!use_remote_draft && draft_backend) ggml_backend_synchronize(draft_backend);
    };

    auto t_dec0 = std::chrono::steady_clock::now();
    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = draft_weights.mask_token_id;
        if (!shards.front().weights.embedder.embed(noise_ids.data(), q_len,
                                                    noise_embed.data())) {
            std::fprintf(stderr, "target-split-dflash noise embed failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }

        constexpr int DRAFT_CTX_MAX_DEFAULT = 2048;
        const int ring_cap = use_remote_draft ? remote_draft->ring_cap() : feature_ring.cap;
        const int draft_ctx = std::min(committed, std::min(ring_cap,
            std::max(DRAFT_CTX_MAX_DEFAULT, draft_ctx_max)));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            !use_remote_draft &&
            draft_feature_mirror_can_view(feature_ring, committed, draft_ctx, mirror_slot0);
        std::vector<float> remote_hidden;
        if (use_remote_draft) {
            if (!remote_draft->propose(committed, draft_ctx, noise_embed, remote_hidden)) {
                std::fprintf(stderr, "target-split-dflash remote draft propose failed\n");
                step_graph_destroy(draft_sg);
                step_graph_destroy(proj_sg);
                return false;
            }
        } else {
            if (!build_draft_step(draft_sg, draft_weights, nullptr, draft_backend,
                                  draft_ctx, use_mirror_view ? &feature_ring : nullptr,
                                  committed)) {
                std::fprintf(stderr, "target-split-dflash draft build failed\n");
                step_graph_destroy(draft_sg);
                step_graph_destroy(proj_sg);
                return false;
            }
            if (!use_mirror_view &&
                !copy_feature_ring_range_to_tensor(feature_ring, draft_sg.target_hidden_cat,
                                                   draft_start, draft_ctx)) {
                std::fprintf(stderr, "target-split-dflash draft feature copy failed\n");
                step_graph_destroy(draft_sg);
                step_graph_destroy(proj_sg);
                return false;
            }
            ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                    sizeof(float) * noise_embed.size());
            pos_k.resize((size_t)draft_ctx + q_len);
            for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
            for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
            ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                    sizeof(int32_t) * pos_q.size());
            ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                    sizeof(int32_t) * pos_k.size());
            auto st = ggml_backend_graph_compute(draft_backend, draft_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "target-split-dflash draft compute %d\n", (int)st);
                step_graph_destroy(draft_sg);
                step_graph_destroy(proj_sg);
                return false;
            }
        }

        if (!proj_sg.gf || !proj_sg.hidden_input || proj_sg.hidden_input->ne[1] != q_len) {
            if (!build_lm_head_projection_step(proj_sg, shards.back().weights,
                                               output_backend, q_len)) {
                std::fprintf(stderr, "target-split-dflash projection build failed\n");
                step_graph_destroy(draft_sg);
                step_graph_destroy(proj_sg);
                return false;
            }
        }
        if (use_remote_draft) {
            ggml_backend_tensor_set(proj_sg.hidden_input, remote_hidden.data(), 0,
                                    remote_hidden.size() * sizeof(float));
        } else {
            const size_t hidden_bytes = ggml_nbytes(draft_sg.hidden_states);
            if (!copy_peer_async(proj_sg.hidden_input->data, output_gpu,
                                 draft_sg.hidden_states->data, draft_gpu,
                                 hidden_bytes)) {
                std::fprintf(stderr, "target-split-dflash hidden peer copy failed\n");
                step_graph_destroy(draft_sg);
                step_graph_destroy(proj_sg);
                return false;
            }
            cudaSetDevice(output_gpu);
            cudaDeviceSynchronize();
        }
        auto st = ggml_backend_graph_compute(output_backend, proj_sg.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "target-split-dflash projection compute %d\n", (int)st);
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        ggml_backend_tensor_get(proj_sg.argmax_tokens, draft_tok.data(), 0,
                                sizeof(int32_t) * q_len);
        draft_tok[0] = last_tok;

        for (auto & shard : shards) snapshot_ssm_state(shard.cache);

        int verify_last_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            draft_tok, committed, q_len,
                                            verify_last_tok, kq_stride_pad, fa_window,
                                            &feature_ring,
                                            &target_tok, nullptr, remote_draft)) {
            std::fprintf(stderr, "target-split-dflash verify failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }

        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        for (auto & shard : shards) restore_ssm_state(shard.cache);

        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }
        int replay_last_tok = -1;
        if (!run_target_layer_split_forward(shards, shards.front().weights,
                                            replay_tok, committed, commit_n,
                                            replay_last_tok, kq_stride_pad, fa_window,
                                            &feature_ring,
                                            nullptr, nullptr, remote_draft)) {
            std::fprintf(stderr, "target-split-dflash replay failed\n");
            step_graph_destroy(draft_sg);
            step_graph_destroy(proj_sg);
            return false;
        }
        last_tok = replay_last_tok;

        bool hit_eos = false;
        for (int i = 0; i < commit_n; i++) {
            out_all.push_back(replay_tok[i]);
            stream_emit_fd(stream_fd, replay_tok[i]);
            if (is_eos_tok(replay_tok[i], shards.front().weights)) hit_eos = true;
        }
        committed += commit_n;
        n_generated += commit_n;
        n_accept_sum += std::min(accept_n, commit_n);
        n_draft_steps++;
        if (hit_eos) break;
    }
    sync_all();
    auto t_dec1 = std::chrono::steady_clock::now();
    const double decode_s = std::chrono::duration<double>(t_dec1 - t_dec0).count();
    const int total_draft_pos = std::max(1, n_draft_steps * q_len);
    const double accept_pct = 100.0 * (double)n_accept_sum / (double)total_draft_pos;
    std::printf("[target-split-dflash] decode tokens=%d time=%.3f s speed=%.2f tok/s\n",
                n_generated, decode_s, n_generated > 0 ? n_generated / decode_s : 0.0);
    std::printf("[target-split-dflash] %d draft steps, accepted=%d/%d (%.1f%%), avg commit/step=%.2f\n",
                n_draft_steps, n_accept_sum, total_draft_pos, accept_pct,
                n_draft_steps > 0 ? (double)n_generated / (double)n_draft_steps : 0.0);
    if (out_path) write_int32_file(out_path, out_all);

    step_graph_destroy(draft_sg);
    step_graph_destroy(proj_sg);
    return true;
}

} // namespace dflash27b
