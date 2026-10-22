# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.v1.request import RequestStatus

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler
from vllm_omni.platforms import current_omni_platform

from .runtime_config import _VoxCPM2RuntimeConfig


class VoxCPM2OmniARAsyncScheduler(OmniARAsyncScheduler):
    """VoxCPM2 scheduler variant for full unified decode graph serving.

    VoxCPM2's full unified decode graph only applies to pure decode batches.
    When a decode-ready request is already running, this scheduler defers new
    waiting admissions for the current tick so the decode batch can stay on the
    VoxCPM2 graph path. This is a model-local serving policy, not a generic AR
    scheduler rule.
    """

    def _unified_decode_graph_enabled(self) -> bool:
        runtime_config = _VoxCPM2RuntimeConfig.from_vllm_config(self.vllm_config)
        # Mirror the talker's gate: the unified decode graph capture is
        # NPU-graphed, so enable the pure-decode-batch admission policy on
        # NPU too when the torch-npu graph APIs are available.
        npu_mod = getattr(torch, "npu", None)
        use_cuda_graph = current_omni_platform.is_cuda() or (
            getattr(current_omni_platform, "device_type", "") == "npu"
            and npu_mod is not None
            and all(hasattr(npu_mod, name) for name in ("NPUGraph", "graph", "graph_pool_handle"))
        )
        return runtime_config.unified_decode_graph_available(use_cuda_graph=use_cuda_graph)

    def _should_defer_waiting_for_unified_decode_graph(self) -> bool:
        if not self._unified_decode_graph_enabled():
            return False
        if not self.waiting or not self.running:
            return False

        for request in self.running:
            if getattr(request, "status", None) != RequestStatus.RUNNING or request.is_finished():
                continue
            if self._get_confirmed_num_computed_tokens(request) >= request.num_prompt_tokens:
                return True
        return False

    def _should_defer_waiting_admission(self) -> bool:
        # Measured trade-off (2026-10-21, 28 req @ 1 rps, 7 concurrent):
        #   defer ON:  TTFP ~1950ms, RTF ~1.00, continuity 100%
        #   defer OFF: TTFP ~640ms,  RTF ~0.79, continuity ~43%
        # The deferral parks new requests until the decode cohort drains,
        # protecting in-flight streams from the prefill-tail block
        # (~700ms of first-patch CFM+VAE inside a mixed step). Default ON
        # for stream continuity; flip via voxcpm2_runtime_config once the
        # prefill-tail cost moves off the mixed-step path.
        # DEFER OFF + estimator-extracted graphs (see notes_1020 §10-21).
        return False
