#include "feature_copy.h"
#include "peer_access.h"

#include <algorithm>

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace dflash27b {

int target_capture_index(const TargetWeights & w, int layer_idx) {
    for (int k = 0; k < w.n_capture_layers; k++) {
        if (w.capture_layer_ids[k] == layer_idx) return k;
    }
    return -1;
}

bool copy_capture_slice_to_draft_ring(
    DraftFeatureMirror & feature_ring,
    int capture_idx,
    const ggml_tensor * act_out,
    int src_device,
    int chunk_start,
    int start_pos,
    int n_tokens) {
    if (!feature_ring.target_feat || capture_idx < 0 || n_tokens <= 0) return true;
    if (feature_ring.cap <= 0) return false;
    const int hidden = feature_ring.hidden_size;
    const size_t dst_stride = feature_ring.target_feat->nb[1];
    const size_t src_stride = act_out->nb[1];
    const size_t row_bytes = (size_t)hidden * sizeof(float);
    for (int i = 0; i < n_tokens; i++) {
        const int slot = (start_pos + i) % feature_ring.cap;
        const void * src = (const char *)act_out->data +
            (size_t)(chunk_start + i) * src_stride;
        void * dst = (char *)feature_ring.target_feat->data +
            (size_t)slot * dst_stride +
            (size_t)capture_idx * (size_t)hidden * sizeof(float);
        if (!copy_peer_async(dst, feature_ring.device, src, src_device, row_bytes)) {
            return false;
        }
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

bool copy_feature_ring_range_to_tensor(
    const DraftFeatureMirror & feature_ring,
    ggml_tensor * dst,
    int start_pos,
    int n_tokens) {
    if (!feature_ring.target_feat || !dst || feature_ring.cap <= 0) return false;
    if (n_tokens <= 0 || n_tokens > feature_ring.cap) return false;

    const int fc_in = feature_ring.n_target_layers * feature_ring.hidden_size;
    const size_t row_bytes = (size_t)fc_in * sizeof(float);
    const size_t src_stride = feature_ring.target_feat->nb[1];
    const size_t dst_stride = dst->nb[1];
    int done = 0;
    while (done < n_tokens) {
        const int slot = (start_pos + done) % feature_ring.cap;
        const int run = std::min(n_tokens - done, feature_ring.cap - slot);
        const char * src_base =
            (const char *)feature_ring.target_feat->data + (size_t)slot * src_stride;
        char * dst_base = (char *)dst->data + (size_t)done * dst_stride;
        if (src_stride == row_bytes && dst_stride == row_bytes) {
            if (!copy_peer_async(dst_base, feature_ring.device,
                                 src_base, feature_ring.device,
                                 row_bytes * (size_t)run)) {
                return false;
            }
        } else {
            for (int i = 0; i < run; i++) {
                if (!copy_peer_async(dst_base + (size_t)i * dst_stride,
                                     feature_ring.device,
                                     src_base + (size_t)i * src_stride,
                                     feature_ring.device,
                                     row_bytes)) {
                    return false;
                }
            }
        }
        done += run;
    }
    return cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace dflash27b
