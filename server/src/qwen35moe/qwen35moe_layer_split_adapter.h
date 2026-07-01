// Qwen35MoE target layer-split adapter.

#pragma once

#include "qwen35/qwen35_layer_split_adapter.h"

namespace dflash::common {

class Qwen35MoeLayerSplitAdapter final : public Qwen35FamilyLayerSplitAdapter {
public:
    explicit Qwen35MoeLayerSplitAdapter(const Qwen35LayerSplitAdapterConfig & cfg);

    const char * name() const override { return "qwen35moe"; }

protected:
    int default_prefill_ubatch(int prompt_tokens) const override;
    const char * prefill_ubatch_env() const override;
};

}  // namespace dflash::common
