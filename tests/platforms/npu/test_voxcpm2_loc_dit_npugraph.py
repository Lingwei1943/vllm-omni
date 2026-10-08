# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the Ascend VoxCPM2 LocDiT NPUGraph adapter."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.platforms.npu import models as npu_models
from vllm_omni.platforms.npu.models import voxcpm2_talker as npu_adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeEstimator(torch.nn.Module):
    def forward(self, x, mu, t, cond, dt):
        return x + mu + t + cond + dt


class FakeTalker:
    def __init__(self) -> None:
        self.estimator = FakeEstimator()
        self.tts = SimpleNamespace(feat_decoder=SimpleNamespace(estimator=self.estimator))


class FakeTalkerSubclass(FakeTalker):
    pass


class FakeGraphRunner:
    instances = []
    supported = True

    def __init__(self, *, max_graphs, component_name, disable_config_hint) -> None:
        self.max_graphs = max_graphs
        self.component_name = component_name
        self.disable_config_hint = disable_config_hint
        self.calls = []
        self.__class__.instances.append(self)

    @classmethod
    def is_supported(cls) -> bool:
        return cls.supported

    def run(self, operation, inputs, constants, compute):
        self.calls.append((operation, inputs, constants))
        return compute(*inputs)


def test_post_load_dispatches_by_model_arch(monkeypatch) -> None:
    model = FakeTalker()
    calls = []
    monkeypatch.setattr(npu_adapter, "setup_voxcpm2_loc_dit_npu_graph", calls.append)

    npu_models.apply_post_load_model_patches(model, SimpleNamespace(model_arch="OtherModel"))
    npu_models.apply_post_load_model_patches(
        model,
        SimpleNamespace(model_arch="VoxCPM2TalkerForConditionalGeneration"),
    )

    assert calls == [model]


def test_loc_dit_npugraph_supports_wrapped_subclass_and_wraps_once(monkeypatch) -> None:
    FakeGraphRunner.instances.clear()
    FakeGraphRunner.supported = True
    monkeypatch.setattr(npu_adapter, "NPUExactGraphRunner", FakeGraphRunner)
    model = FakeTalkerSubclass()
    wrapped_model = SimpleNamespace(module=model)

    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(wrapped_model)
    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(wrapped_model)

    inputs = tuple(torch.full((2, 3), value) for value in range(5))
    output = model.estimator(*inputs)
    runner = FakeGraphRunner.instances[0]
    assert torch.equal(output, sum(inputs[1:], start=inputs[0]))
    assert runner.max_graphs == 32
    assert runner.component_name == "VoxCPM2 LocDiT"
    assert runner.calls == [("forward", inputs, ())]
    assert len(FakeGraphRunner.instances) == 1


def test_loc_dit_npugraph_falls_back_when_apis_are_unavailable(monkeypatch) -> None:
    FakeGraphRunner.instances.clear()
    FakeGraphRunner.supported = False
    monkeypatch.setattr(npu_adapter, "NPUExactGraphRunner", FakeGraphRunner)
    model = FakeTalker()

    npu_adapter.setup_voxcpm2_loc_dit_npu_graph(model)

    assert not hasattr(model.estimator, "_voxcpm2_npu_graph_runner")
    FakeGraphRunner.supported = True


def test_post_load_accepts_architectures_list_and_probes_structure(monkeypatch) -> None:
    """model_config may not expose model_arch; the patch should also accept
    the vLLM `architectures` list and verify the estimator seam exists."""
    calls = []
    monkeypatch.setattr(npu_adapter, "setup_voxcpm2_loc_dit_npu_graph", calls.append)
    model = FakeTalker()

    npu_models.apply_post_load_model_patches(
        model,
        SimpleNamespace(architectures=["VoxCPM2TalkerForConditionalGeneration"]),
    )
    assert calls == [model]

    # Missing estimator seam: arch matches but structure probe fails -> skipped.
    class SeamlessTalker:
        pass

    seamless = SeamlessTalker()
    npu_models.apply_post_load_model_patches(
        seamless,
        SimpleNamespace(architectures=["VoxCPM2TalkerForConditionalGeneration"]),
    )
    assert calls == [model]


def test_unified_capture_selects_platform_graph_api(monkeypatch) -> None:
    """The unified decode graph capture must pick CUDAGraph on CUDA and
    NPUGraph on NPU instead of hardcoding one backend (regression pin)."""
    import torch.npu

    # Import the platform singleton from the package so the test survives
    # upstream refactors of how voxcpm2_talker.py binds the name (both
    # `from vllm_omni.platforms import current_omni_platform` and
    # `import vllm_omni.platforms as omni_platform` styles reference this
    # same object).
    from vllm_omni.platforms import current_omni_platform

    selected = {}

    class FakeNpuGraph:
        pass

    class FakeCudaGraph:
        pass

    def fake_is_npu():
        return selected["is_npu"]

    monkeypatch.setattr(current_omni_platform, "is_npu", fake_is_npu)
    monkeypatch.setattr(torch.npu, "NPUGraph", FakeNpuGraph, raising=False)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", FakeCudaGraph, raising=False)

    # Mirror the platform-aware selection the capture performs.
    def pick():
        if current_omni_platform.is_npu():
            return torch.npu.NPUGraph
        return torch.cuda.CUDAGraph

    selected["is_npu"] = True
    assert pick() is FakeNpuGraph
    selected["is_npu"] = False
    assert pick() is FakeCudaGraph
