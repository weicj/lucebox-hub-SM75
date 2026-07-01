// Qwen35MoE target layer-split adapter.

#include "qwen35moe_layer_split_adapter.h"

#include <algorithm>
#include <cstdlib>

namespace dflash::common {

Qwen35MoeLayerSplitAdapter::Qwen35MoeLayerSplitAdapter(
        const Qwen35LayerSplitAdapterConfig & cfg)
    : Qwen35FamilyLayerSplitAdapter(cfg) {}

int Qwen35MoeLayerSplitAdapter::default_prefill_ubatch(int prompt_tokens) const {
    (void)prompt_tokens;
    return std::max(1, config().chunk > 0 ? config().chunk : 512);
}

const char * Qwen35MoeLayerSplitAdapter::prefill_ubatch_env() const {
    if (std::getenv("DFLASH_QWEN35MOE_PREFILL_UBATCH")) {
        return "DFLASH_QWEN35MOE_PREFILL_UBATCH";
    }
    return "DFLASH27B_PREFILL_UBATCH";
}

}  // namespace dflash::common
