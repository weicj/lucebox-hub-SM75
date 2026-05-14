// draft_ipc.cpp — DFlash draft IPC client + daemon implementation.

#include "draft_ipc.h"
#include "internal.h"
#include "draft_feature_mirror.h"
#include "graph_builders.h"
#include "feature_copy.h"
#include "step_graph.h"
#include "io_utils.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace dflash27b {

// ── DFlashDraftIpcClient ────────────────────────────────────────────

bool DFlashDraftIpcClient::start(
        const std::string & bin,
        const std::string & draft_path,
        int draft_gpu,
        int ring_cap,
        const std::string & work_dir) {
#if defined(_WIN32)
    (void)bin; (void)draft_path; (void)draft_gpu; (void)ring_cap; (void)work_dir;
    std::fprintf(stderr, "DFlash draft IPC is only implemented on POSIX hosts\n");
    return false;
#else
    close();
    if (bin.empty() || draft_path.empty() || ring_cap <= 0) return false;
    if (!init_work_dir(work_dir)) return false;

    int cmd_pipe[2] = {-1, -1};
    int stream_pipe[2] = {-1, -1};
    if (::pipe(cmd_pipe) != 0 || ::pipe(stream_pipe) != 0) {
        std::fprintf(stderr, "draft-ipc pipe failed: %s\n", std::strerror(errno));
        if (cmd_pipe[0] >= 0) ::close(cmd_pipe[0]);
        if (cmd_pipe[1] >= 0) ::close(cmd_pipe[1]);
        if (stream_pipe[0] >= 0) ::close(stream_pipe[0]);
        if (stream_pipe[1] >= 0) ::close(stream_pipe[1]);
        return false;
    }

    pid_ = ::fork();
    if (pid_ < 0) {
        std::fprintf(stderr, "draft-ipc fork failed: %s\n", std::strerror(errno));
        ::close(cmd_pipe[0]); ::close(cmd_pipe[1]);
        ::close(stream_pipe[0]); ::close(stream_pipe[1]);
        pid_ = -1;
        return false;
    }
    if (pid_ == 0) {
        ::dup2(cmd_pipe[0], STDIN_FILENO);
        ::close(cmd_pipe[0]);
        ::close(cmd_pipe[1]);
        ::close(stream_pipe[0]);

        const std::string cap_arg = "--ring-cap=" + std::to_string(ring_cap);
        const std::string gpu_arg = "--draft-gpu=" + std::to_string(std::max(0, draft_gpu));
        const std::string fd_arg = "--stream-fd=" + std::to_string(stream_pipe[1]);
        ::execl(bin.c_str(), bin.c_str(),
                "--draft-ipc-daemon", draft_path.c_str(),
                cap_arg.c_str(), gpu_arg.c_str(), fd_arg.c_str(),
                (char *)nullptr);
        std::fprintf(stderr, "draft-ipc exec failed: %s: %s\n",
                     bin.c_str(), std::strerror(errno));
        _exit(127);
    }

    ::close(cmd_pipe[0]);
    ::close(stream_pipe[1]);
    stream_fd_ = stream_pipe[0];
    cmd_ = ::fdopen(cmd_pipe[1], "w");
    if (!cmd_) {
        std::fprintf(stderr, "draft-ipc fdopen failed: %s\n", std::strerror(errno));
        ::close(cmd_pipe[1]);
        close();
        return false;
    }
    int32_t status = -1;
    if (!read_exact_fd(stream_fd_, &status, sizeof(status)) || status != 0) {
        std::fprintf(stderr, "draft-ipc daemon did not become ready (status=%d)\n", status);
        close();
        return false;
    }
    ring_cap_ = ring_cap;
    active_ = true;
    std::printf("[draft-ipc] ready bin=%s gpu=%d ring_cap=%d work_dir=%s\n",
                bin.c_str(), draft_gpu, ring_cap, work_dir_.c_str());
    return true;
#endif
}

bool DFlashDraftIpcClient::send_feature_slice(
        int capture_idx,
        int start_pos,
        int n_tokens,
        const std::vector<float> & slice) {
#if defined(_WIN32)
    (void)capture_idx; (void)start_pos; (void)n_tokens; (void)slice;
    return false;
#else
    if (!active_ || !cmd_ || n_tokens <= 0) return false;
    const size_t expected = (size_t)n_tokens * hidden_size_;
    if (slice.size() != expected) return false;
    const std::string path = next_path("feature");
    if (!write_binary_file(path, slice.data(), slice.size() * sizeof(float))) {
        std::fprintf(stderr, "draft-ipc write feature failed: %s\n", path.c_str());
        return false;
    }
    std::fprintf(cmd_, "feature_slice %d %d %d %s\n",
                 capture_idx, start_pos, n_tokens, path.c_str());
    std::fflush(cmd_);
    int32_t status = -1;
    const bool ok = read_exact_fd(stream_fd_, &status, sizeof(status)) && status == 0;
    std::remove(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "draft-ipc feature_slice failed status=%d\n", status);
    }
    return ok;
#endif
}

bool DFlashDraftIpcClient::propose(
        int committed,
        int ctx_len,
        const std::vector<float> & noise_embed,
        std::vector<float> & hidden_out) {
#if defined(_WIN32)
    (void)committed; (void)ctx_len; (void)noise_embed; (void)hidden_out;
    return false;
#else
    if (!active_ || !cmd_ || ctx_len <= 0) return false;
    const size_t noise_expected =
        (size_t)hidden_size_ * block_size_;
    if (noise_embed.size() != noise_expected) return false;
    const std::string path = next_path("noise");
    if (!write_binary_file(path, noise_embed.data(), noise_embed.size() * sizeof(float))) {
        std::fprintf(stderr, "draft-ipc write noise failed: %s\n", path.c_str());
        return false;
    }
    std::fprintf(cmd_, "propose %d %d %s\n", committed, ctx_len, path.c_str());
    std::fflush(cmd_);
    int32_t status = -1;
    bool ok = read_exact_fd(stream_fd_, &status, sizeof(status)) && status == 0;
    if (ok) {
        hidden_out.assign(noise_expected, 0.0f);
        ok = read_exact_fd(stream_fd_, hidden_out.data(),
                           hidden_out.size() * sizeof(float));
    }
    std::remove(path.c_str());
    if (!ok) {
        std::fprintf(stderr, "draft-ipc propose failed status=%d\n", status);
    }
    return ok;
#endif
}

void DFlashDraftIpcClient::close() {
#if !defined(_WIN32)
    if (cmd_) {
        std::fclose(cmd_);
        cmd_ = nullptr;
    }
    if (stream_fd_ >= 0) {
        ::close(stream_fd_);
        stream_fd_ = -1;
    }
    if (pid_ > 0) {
        int status = 0;
        ::waitpid(pid_, &status, 0);
        pid_ = -1;
    }
    if (owns_work_dir_ && !work_dir_.empty()) {
        ::rmdir(work_dir_.c_str());
    }
#endif
    active_ = false;
    ring_cap_ = 0;
}

#if !defined(_WIN32)
bool DFlashDraftIpcClient::init_work_dir(const std::string & requested) {
    if (!requested.empty()) {
        work_dir_ = requested;
        owns_work_dir_ = false;
        if (::mkdir(work_dir_.c_str(), 0700) != 0 && errno != EEXIST) {
            std::fprintf(stderr, "draft-ipc mkdir failed: %s: %s\n",
                         work_dir_.c_str(), std::strerror(errno));
            return false;
        }
        return true;
    }
    const char * tmp = std::getenv("TMPDIR");
    std::string templ = std::string(tmp && *tmp ? tmp : "/tmp") +
                        "/dflash-draft-ipc-XXXXXX";
    std::vector<char> buf(templ.begin(), templ.end());
    buf.push_back('\0');
    char * dir = ::mkdtemp(buf.data());
    if (!dir) {
        std::fprintf(stderr, "draft-ipc mkdtemp failed: %s\n", std::strerror(errno));
        return false;
    }
    work_dir_ = dir;
    owns_work_dir_ = true;
    return true;
}

std::string DFlashDraftIpcClient::next_path(const char * prefix) {
    return work_dir_ + "/" + prefix + "_" + std::to_string(seq_++) + ".bin";
}
#endif

// ── Remote draft feature copy helper ────────────────────────────────

bool copy_capture_slice_to_remote_draft(
        DFlashDraftIpcClient & remote,
        int capture_idx,
        const ggml_tensor * act_out,
        ggml_backend_t src_backend,
        int chunk_start,
        int start_pos,
        int n_tokens) {
    if (!remote.active() || !act_out || capture_idx < 0 || n_tokens <= 0) return true;
    const int hidden = remote.hidden_size();
    const size_t row_bytes = (size_t)hidden * sizeof(float);
    const size_t src_stride = act_out->nb[1];
    std::vector<float> host((size_t)n_tokens * hidden);
    ggml_backend_synchronize(src_backend);
    if (src_stride == row_bytes) {
        ggml_backend_tensor_get(act_out, host.data(),
                                (size_t)chunk_start * src_stride,
                                row_bytes * (size_t)n_tokens);
    } else {
        for (int i = 0; i < n_tokens; i++) {
            ggml_backend_tensor_get(act_out,
                                    host.data() + (size_t)i * hidden,
                                    (size_t)(chunk_start + i) * src_stride,
                                    row_bytes);
        }
    }
    return remote.send_feature_slice(capture_idx, start_pos, n_tokens, host);
}

// ── IPC Daemon ──────────────────────────────────────────────────────

int run_dflash_draft_ipc_daemon(const char * draft_path,
                                int ring_cap,
                                int draft_gpu,
                                int stream_fd) {
#if defined(_WIN32)
    (void)draft_path; (void)ring_cap; (void)draft_gpu; (void)stream_fd;
    std::fprintf(stderr, "DFlash draft IPC daemon is only implemented on POSIX hosts\n");
    return 2;
#else
    if (!draft_path || ring_cap <= 0 || stream_fd < 0) {
        std::fprintf(stderr, "usage: test_dflash --draft-ipc-daemon <draft> --ring-cap=N --stream-fd=FD [--draft-gpu=N]\n");
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cuda_init(std::max(0, draft_gpu));
    if (!backend) {
        std::fprintf(stderr, "[draft-ipc-daemon] backend init failed gpu=%d\n", draft_gpu);
        stream_status(stream_fd, -1);
        return 1;
    }

    DraftWeights draft_weights;
    std::string dp(draft_path);
    bool draft_ok = false;
    if (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf") {
        draft_ok = load_draft_gguf(draft_path, backend, draft_weights);
    } else {
        draft_ok = load_draft_safetensors(draft_path, backend, draft_weights);
    }
    if (!draft_ok) {
        std::fprintf(stderr, "[draft-ipc-daemon] draft load failed: %s\n",
                     dflash27b_last_error());
        stream_status(stream_fd, -1);
        ggml_backend_free(backend);
        return 1;
    }

    DraftFeatureMirror feature_ring;
    if (!draft_feature_mirror_init(feature_ring, backend, draft_gpu, draft_gpu, ring_cap)) {
        std::fprintf(stderr, "[draft-ipc-daemon] feature ring init failed cap=%d gpu=%d\n",
                     ring_cap, draft_gpu);
        stream_status(stream_fd, -1);
        free_draft_weights(draft_weights);
        ggml_backend_free(backend);
        return 1;
    }

    std::fprintf(stderr, "[draft-ipc-daemon] ready gpu=%d ring_cap=%d\n",
                 draft_gpu, ring_cap);
    stream_status(stream_fd, 0);

    const int hidden = draft_weights.n_embd;
    const int q_len = draft_weights.block_size;
    const int n_tgt_layers = draft_weights.n_target_layers;
    StepGraph draft_sg;
    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<float> hidden_out((size_t)hidden * q_len);

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit") {
            break;
        }
        if (cmd == "feature_slice") {
            int capture_idx = -1;
            int start_pos = -1;
            int n_tokens = 0;
            iss >> capture_idx >> start_pos >> n_tokens;
            std::string path = read_line_tail(iss);
            if (capture_idx < 0 || capture_idx >= n_tgt_layers ||
                start_pos < 0 || n_tokens <= 0 || path.empty()) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad feature_slice: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            std::vector<float> slice((size_t)n_tokens * hidden);
            if (!read_binary_file_exact(path, slice.data(), slice.size() * sizeof(float))) {
                std::fprintf(stderr, "[draft-ipc-daemon] read feature_slice failed: %s\n",
                             path.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            const size_t dst_stride = feature_ring.target_feat->nb[1];
            const size_t slice_offset =
                (size_t)capture_idx * (size_t)hidden * sizeof(float);
            for (int i = 0; i < n_tokens; i++) {
                const int slot = (start_pos + i) % feature_ring.cap;
                const size_t dst_off = (size_t)slot * dst_stride + slice_offset;
                ggml_backend_tensor_set(feature_ring.target_feat,
                                        slice.data() + (size_t)i * hidden,
                                        dst_off,
                                        (size_t)hidden * sizeof(float));
            }
            ggml_backend_synchronize(backend);
            stream_status(stream_fd, 0);
            continue;
        }
        if (cmd == "propose") {
            int committed = -1;
            int ctx_len = 0;
            iss >> committed >> ctx_len;
            std::string path = read_line_tail(iss);
            if (committed < 0 || ctx_len <= 0 || ctx_len > feature_ring.cap || path.empty()) {
                std::fprintf(stderr, "[draft-ipc-daemon] bad propose: %s\n",
                             line.c_str());
                stream_status(stream_fd, -1);
                continue;
            }
            if (!read_binary_file_exact(path, noise_embed.data(),
                                        noise_embed.size() * sizeof(float))) {
                std::fprintf(stderr, "[draft-ipc-daemon] read noise failed: %s\n",
                             path.c_str());
                stream_status(stream_fd, -1);
                continue;
            }

            int mirror_slot0 = 0;
            const bool use_mirror_view =
                draft_feature_mirror_can_view(feature_ring, committed, ctx_len, mirror_slot0);
            if (!build_draft_step(draft_sg, draft_weights, nullptr, backend,
                                  ctx_len, use_mirror_view ? &feature_ring : nullptr,
                                  committed)) {
                std::fprintf(stderr, "[draft-ipc-daemon] draft build failed\n");
                stream_status(stream_fd, -1);
                continue;
            }
            if (!use_mirror_view &&
                !copy_feature_ring_range_to_tensor(feature_ring,
                                                   draft_sg.target_hidden_cat,
                                                   committed - ctx_len,
                                                   ctx_len)) {
                std::fprintf(stderr, "[draft-ipc-daemon] feature copy failed\n");
                stream_status(stream_fd, -1);
                continue;
            }
            ggml_backend_tensor_set(draft_sg.inp_embed, noise_embed.data(), 0,
                                    noise_embed.size() * sizeof(float));
            pos_k.resize((size_t)ctx_len + q_len);
            for (int i = 0; i < q_len; i++) pos_q[i] = ctx_len + i;
            for (int i = 0; i < ctx_len + q_len; i++) pos_k[i] = i;
            ggml_backend_tensor_set(draft_sg.positions, pos_q.data(), 0,
                                    pos_q.size() * sizeof(int32_t));
            ggml_backend_tensor_set(draft_sg.positions_k, pos_k.data(), 0,
                                    pos_k.size() * sizeof(int32_t));
            auto st = ggml_backend_graph_compute(backend, draft_sg.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "[draft-ipc-daemon] draft compute failed status=%d\n",
                             (int)st);
                stream_status(stream_fd, -1);
                continue;
            }
            ggml_backend_tensor_get(draft_sg.hidden_states, hidden_out.data(), 0,
                                    hidden_out.size() * sizeof(float));
            if (!stream_status(stream_fd, 0) ||
                !write_exact_fd(stream_fd, hidden_out.data(),
                                hidden_out.size() * sizeof(float))) {
                std::fprintf(stderr, "[draft-ipc-daemon] stream write failed\n");
                break;
            }
            continue;
        }
        std::fprintf(stderr, "[draft-ipc-daemon] unknown command: %s\n", line.c_str());
        stream_status(stream_fd, -1);
    }

    step_graph_destroy(draft_sg);
    draft_feature_mirror_free(feature_ring);
    free_draft_weights(draft_weights);
    ggml_backend_free(backend);
    std::fprintf(stderr, "[draft-ipc-daemon] stopped\n");
    return 0;
#endif
}

} // namespace dflash27b
