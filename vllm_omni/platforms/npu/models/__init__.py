# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from typing import Any

logger = logging.getLogger(__name__)

_VOXCPM2_TALKER_ARCH = "VoxCPM2TalkerForConditionalGeneration"


def _is_voxcpm2_talker(model: object) -> bool:
    """Probe the loaded model structure instead of trusting model_config
    attributes, which may not expose the Omni stage architecture on every
    runner path."""
    wrapped_model = getattr(model, "module", None)
    talker = wrapped_model if wrapped_model is not None else model
    tts = getattr(talker, "tts", None)
    feat_decoder = getattr(tts, "feat_decoder", None)
    estimator = getattr(feat_decoder, "estimator", None)
    return estimator is not None


def apply_post_load_model_patches(model: object, model_config: Any) -> None:
    """Apply model-specific Ascend setup after weights are loaded."""
    arch = getattr(model_config, "model_arch", None)
    if arch is None:
        architectures = getattr(model_config, "architectures", None) or []
        arch = architectures[0] if architectures else None

    if arch == _VOXCPM2_TALKER_ARCH:
        if not _is_voxcpm2_talker(model):
            logger.warning(
                "VoxCPM2 talker architecture detected but the expected "
                "tts.feat_decoder.estimator seam is absent; skipping LocDiT "
                "NPUGraph setup"
            )
            return
        from vllm_omni.platforms.npu.models.voxcpm2_talker import (
            setup_voxcpm2_loc_dit_npu_graph,
        )

        setup_voxcpm2_loc_dit_npu_graph(model)
