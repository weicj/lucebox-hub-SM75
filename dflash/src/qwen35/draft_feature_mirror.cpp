#include "draft_feature_mirror.h"
#include "peer_access.h"

#include "ggml.h"

// ggml_get_to_fp32_cuda is not in any public header — it lives in
// ggml-cuda/convert.cuh. Declare the typedef + extern here so that
// draft_feature_mirror.cpp (and any future src/ consumer) can link against it.
using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
extern "C++" to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

#include <algorithm>
#include <cstdio>

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace dflash27b {

// ── internal helpers ────────────────────────────────────────────

static bool ensure_bf16_staging(DraftFeatureMirror & mirror, size_t elems) {
    if (elems <= mirror.bf16_staging_elems) return true;
    cudaError_t err = cudaSetDevice(mirror.device);
    if (err != cudaSuccess) return false;
    if (mirror.bf16_staging) {
        cudaFree(mirror.bf16_staging);
        mirror.bf16_staging = nullptr;
        mirror.bf16_staging_elems = 0;
    }
    err = cudaMalloc(&mirror.bf16_staging, elems * sizeof(uint16_t));
    if (err != cudaSuccess) return false;
    mirror.bf16_staging_elems = elems;
    return true;
}

// ── public API ──────────────────────────────────────────────────

void draft_feature_mirror_free(DraftFeatureMirror & mirror) {
    if (mirror.bf16_staging) {
        cudaSetDevice(mirror.device);
        cudaFree(mirror.bf16_staging);
        mirror.bf16_staging = nullptr;
        mirror.bf16_staging_elems = 0;
    }
    if (mirror.buf) {
        ggml_backend_buffer_free(mirror.buf);
        mirror.buf = nullptr;
    }
    if (mirror.ctx) {
        ggml_free(mirror.ctx);
        mirror.ctx = nullptr;
    }
    mirror.target_feat = nullptr;
    mirror.device = 0;
    mirror.target_device = 0;
    mirror.cap = 0;
}

bool draft_feature_mirror_init(DraftFeatureMirror & mirror,
                               ggml_backend_t backend,
                               int device,
                               int target_device,
                               int cap,
                               int n_target_layers,
                               int hidden_size) {
    draft_feature_mirror_free(mirror);
    if (cap <= 0) return false;
    mirror.device = device;
    mirror.target_device = target_device;
    mirror.n_target_layers = n_target_layers;
    mirror.hidden_size = hidden_size;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 4 + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    mirror.ctx = ggml_init(ip);
    if (!mirror.ctx) return false;

    const int fc_in = n_target_layers * hidden_size;
    mirror.target_feat = ggml_new_tensor_2d(mirror.ctx, GGML_TYPE_F32, fc_in, cap);
    ggml_set_name(mirror.target_feat, "draft_target_feat_mirror");
    mirror.buf = ggml_backend_alloc_ctx_tensors(mirror.ctx, backend);
    if (!mirror.buf) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    const size_t bytes = (size_t)fc_in * (size_t)cap * sizeof(float);
    cudaSetDevice(device);
    cudaError_t err = cudaMemset(mirror.target_feat->data, 0, bytes);
    if (err != cudaSuccess) {
        draft_feature_mirror_free(mirror);
        return false;
    }
    mirror.cap = cap;
    return true;
}

bool draft_feature_mirror_can_view(const DraftFeatureMirror & mirror,
                                   int committed,
                                   int ctx_len,
                                   int & slot0) {
    if (!mirror.target_feat || mirror.cap <= 0) return false;
    if (ctx_len <= 0 || ctx_len > mirror.cap || committed < ctx_len) return false;
    const int start = committed - ctx_len;
    slot0 = start % mirror.cap;
    return slot0 + ctx_len <= mirror.cap;
}

bool draft_feature_mirror_sync_range(const TargetCache & cache,
                                     const DraftFeatureMirror & mirror,
                                     int start_pos,
                                     int n_tokens) {
    if (!cache.target_feat || !mirror.target_feat || mirror.cap <= 0) return false;
    if (n_tokens <= 0) return true;
    if (n_tokens > mirror.cap) return false;

    const int fc_in = mirror.n_target_layers * mirror.hidden_size;
    const int src_cap = cache.target_feat_cap;
    const size_t src_stride = cache.target_feat->nb[1];
    const size_t dst_stride = mirror.target_feat->nb[1];

    int done = 0;
    while (done < n_tokens) {
        const int src_slot = (start_pos + done) % src_cap;
        const int dst_slot = (start_pos + done) % mirror.cap;
        const int src_run = src_cap - src_slot;
        const int dst_run = mirror.cap - dst_slot;
        const int run = std::min(n_tokens - done, std::min(src_run, dst_run));
        const size_t elems = (size_t)run * (size_t)fc_in;
        const void * src =
            (const char *)cache.target_feat->data + (size_t)src_slot * src_stride;
        void * dst =
            (char *)mirror.target_feat->data + (size_t)dst_slot * dst_stride;
        auto bf16_to_f32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        if (mirror.device == mirror.target_device) {
            cudaSetDevice(mirror.device);
            bf16_to_f32(src, (float *)dst, (int64_t)elems, nullptr);
        } else {
            DraftFeatureMirror & mutable_mirror =
                const_cast<DraftFeatureMirror &>(mirror);
            if (!ensure_bf16_staging(mutable_mirror, elems)) return false;
            if (!copy_peer_async(mirror.bf16_staging, mirror.device,
                                 src, mirror.target_device,
                                 elems * sizeof(uint16_t))) {
                return false;
            }
            cudaSetDevice(mirror.device);
            bf16_to_f32(mirror.bf16_staging, (float *)dst, (int64_t)elems, nullptr);
        }
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return false;
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool draft_feature_mirror_sync_tail(const TargetCache & cache,
                                    const DraftFeatureMirror & mirror,
                                    int committed) {
    if (!mirror.target_feat || committed <= 0) return true;
    const int n = std::min(committed, mirror.cap);
    return draft_feature_mirror_sync_range(cache, mirror, committed - n, n);
}

}  // namespace dflash27b
