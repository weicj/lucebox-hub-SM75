// DraftFeatureMirror — mirrors target hidden-state features on the draft GPU.
//
// In speculative decoding the target model runs on one GPU while the draft
// model may run on a different GPU. The draft model's cross-attention layers
// need the target's intermediate hidden states. DraftFeatureMirror keeps an
// F32 ring buffer on the draft GPU and syncs ranges from the target's BF16
// feature cache, converting BF16→F32 on the fly.

#pragma once

#include "internal.h"  // TargetCache, DFLASH27B_* constants

#include "ggml.h"
#include "ggml-backend.h"

namespace dflash27b {

struct DraftFeatureMirror {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_tensor * target_feat = nullptr; // F32 [n_target_layers*hidden_size, cap]
    void * bf16_staging = nullptr;
    size_t bf16_staging_elems = 0;
    int device = 0;
    int target_device = 0;
    int cap = 0;
    int n_target_layers = DFLASH27B_DRAFT_N_TARGET_LAYERS;
    int hidden_size = DFLASH27B_TARGET_HIDDEN;
};

void draft_feature_mirror_free(DraftFeatureMirror & mirror);

bool draft_feature_mirror_init(DraftFeatureMirror & mirror,
                               ggml_backend_t backend,
                               int device,
                               int target_device,
                               int cap,
                               int n_target_layers = DFLASH27B_DRAFT_N_TARGET_LAYERS,
                               int hidden_size = DFLASH27B_TARGET_HIDDEN);

// Check whether the mirror ring buffer can provide a contiguous view
// of ctx_len slots ending at committed. Returns slot0 (the starting
// slot in the ring buffer) on success.
bool draft_feature_mirror_can_view(const DraftFeatureMirror & mirror,
                                   int committed,
                                   int ctx_len,
                                   int & slot0);

// Copy and convert BF16→F32 for n_tokens starting at start_pos from the
// target cache into the mirror ring buffer.
bool draft_feature_mirror_sync_range(const TargetCache & cache,
                                     const DraftFeatureMirror & mirror,
                                     int start_pos,
                                     int n_tokens);

// Convenience: sync the last `committed` tokens (or mirror.cap, whichever is smaller).
bool draft_feature_mirror_sync_tail(const TargetCache & cache,
                                    const DraftFeatureMirror & mirror,
                                    int committed);

}  // namespace dflash27b
