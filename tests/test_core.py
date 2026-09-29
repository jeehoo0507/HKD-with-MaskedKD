import copy

import numpy as np
import pytest
import torch
import timm
from timm.models.vision_transformer import VisionTransformer

from hkd.core import (MaskableTeacher, SmallCNN, hybrid_loss, information_flow_loss,
                      relation_probabilities, select_patches)
from run_experiment import (BudgetExceeded, make_plan, optimizer_for, paired_interval,
                            scores, state_hash, train_step)


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(2)
    torch.manual_seed(17)


def reference_loss(student, teacher):
    """Independent scalar NumPy calculation of paper equations (5)-(8)."""
    arrays = [teacher.detach().numpy(), student.detach().numpy()]
    total = 0.0
    for kernel in ("cosine", "student_t"):
        distributions = []
        for x in arrays:
            similarities = np.zeros((len(x), len(x)))
            for i in range(len(x)):
                for j in range(len(x)):
                    if i != j:
                        if kernel == "cosine":
                            similarities[i, j] = max(1e-6, (1 + np.dot(x[i], x[j]) / (np.linalg.norm(x[i]) * np.linalg.norm(x[j]))) / 2)
                        else:
                            similarities[i, j] = 1 / (1 + np.sqrt(np.sum((x[i] - x[j]) ** 2) + 1e-6))
            distributions.append(similarities / similarities.sum(axis=1, keepdims=True))
        p, q = distributions
        total += np.mean((p - q) * (np.log(np.maximum(p, 1e-8)) - np.log(np.maximum(q, 1e-8))))
    return total


def test_hybrid_matches_equations_with_unequal_feature_dimensions():
    student = torch.randn(8, 7, requires_grad=True)
    teacher = torch.randn(8, 11, requires_grad=True)
    loss = hybrid_loss(student, teacher)
    assert loss.item() == pytest.approx(reference_loss(student, teacher), rel=2e-5, abs=1e-7)
    loss.backward()
    assert student.grad.abs().sum() > 0
    assert teacher.grad is None


def test_identity_zero_degenerate_finite_and_batch_guard():
    x = torch.randn(8, 4, requires_grad=True)
    assert hybrid_loss(x, x).item() == pytest.approx(0, abs=1e-8)
    zero = torch.zeros(8, 4, requires_grad=True)
    loss = hybrid_loss(zero, x)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(zero.grad).all()
    for n in (1, 2):
        with pytest.raises(ValueError, match="at least 3"):
            hybrid_loss(torch.randn(n, 4), torch.randn(n, 6))
    probabilities = relation_probabilities(x, "student_t")
    assert torch.equal(probabilities.diag(), torch.zeros(8))
    torch.testing.assert_close(probabilities.sum(1), torch.ones(8))


def test_critical_period_weights_and_layer_guard():
    s = [torch.randn(8, i + 3) for i in range(4)]
    t = [torch.randn(8, 2 * (i + 3)) for i in range(4)]
    base = [hybrid_loss(a, b) for a, b in zip(s, t)]
    for epoch in (0, 10, 30):
        torch.testing.assert_close(information_flow_loss(s, t, epoch),
                                   100 * 0.7**epoch * sum(base[:3]) + base[-1])
    with pytest.raises(ValueError):
        information_flow_loss(s[:2], t, 0)


def tiny_teacher():
    return MaskableTeacher(VisionTransformer(img_size=224, patch_size=16, embed_dim=48,
                                             depth=2, num_heads=3, num_classes=10))


def test_token_removal_preserves_full_forward_positions_and_cls():
    teacher = tiny_teacher()
    images = torch.randn(3, 3, 224, 224)
    original = teacher.model.forward_head(teacher.model.forward_features(images), pre_logits=True)
    torch.testing.assert_close(teacher(images), original)
    all_indices = torch.arange(196).expand(3, -1)
    torch.testing.assert_close(teacher(images, all_indices), original)
    captured = []
    hook = teacher.model.blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    saliency = torch.arange(196).float().expand(3, -1).requires_grad_(True)
    indices = select_patches(saliency, 98)
    result = teacher(images, indices)
    hook.remove()
    full = teacher.model._pos_embed(teacher.model.patch_embed(images))
    assert captured[0].shape == (3, 99, 48)
    torch.testing.assert_close(captured[0][:, 0], full[:, 0])
    torch.testing.assert_close(captured[0][:, 1:], full[:, 99:])
    assert not result.requires_grad and saliency.grad is None
    assert indices[0].tolist() == list(range(98, 196))
    torch.testing.assert_close(result, teacher.chunks(images, indices, 1), atol=1e-6, rtol=1e-5)


def test_actual_deit_base_architecture_forward():
    # Real architecture, random weights: tests timm compatibility, not accuracy.
    model = timm.create_model("deit_base_patch16_224", pretrained=False).eval()
    teacher = MaskableTeacher(model)
    images = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = model.forward_head(model.forward_features(images), pre_logits=True)
        torch.testing.assert_close(teacher(images), expected)
        masked = teacher(images, torch.arange(98).unsqueeze(0))
    assert masked.shape == (1, 768) and torch.isfinite(masked).all()


def test_matched_models_and_training_stages():
    teacher = tiny_teacher()
    aux = SmallCNN(2)
    student = SmallCNN(1)
    assert len(aux.stages) == len(student.stages) == 3
    assert [s[0].out_channels for s in aux.stages] == [16, 32, 64]
    assert [s[0].out_channels for s in student.stages] == [8, 16, 32]
    batch = (torch.randn(8, 3, 224, 224), torch.arange(8), torch.arange(8))
    device = torch.device("cpu")
    other = copy.deepcopy(aux)
    assert state_hash(aux) == state_hash(other)
    before = state_hash(aux)
    opt, scaler = optimizer_for(aux, device)
    stats = train_step(aux, opt, scaler, batch, device, "aux", teacher, keep=98, micro=2)
    assert np.isfinite(stats["loss"]) and stats["kd"] > 0
    assert stats["loss"] == stats["kd"]  # auxiliary transfer is PKT, not vanilla logit KD
    assert before != state_hash(aux)
    aux.eval().requires_grad_(False)
    frozen_hash = state_hash(aux)
    before = state_hash(student)
    opt, scaler = optimizer_for(student, device)
    stats = train_step(student, opt, scaler, batch, device, "student", aux)
    assert np.isfinite(stats["loss"]) and stats["kd"] > 0
    assert before != state_hash(student)
    assert frozen_hash == state_hash(aux)


def test_budget_rejects_infeasible_pair_and_does_not_shorten_critical_period():
    measurement = dict(train_step_s=0.1, validation_batch_s=0.02)
    measured = {name: measurement for name in ("warm", "aux_full", "aux_masked", "student")}
    plan = make_plan(measured, 3210, 530, 32, 4 * 3600)
    assert plan["aux_epochs"] == 20 and plan["student_epochs"] == 50
    with pytest.raises(BudgetExceeded):
        make_plan(measured, 3210, 530, 32, 10)


def test_metrics_macro_and_paired_bootstrap():
    labels = [0] * 9 + [1]
    result = scores(labels, [0] * 10)
    assert result["macro_accuracy"] == 50
    assert result["accuracy"] == 90
    full = dict(labels=list(range(10)), predictions=list(range(10)), ids=list(range(10)))
    assert paired_interval(full, full, 0) == [0, 0]
    wrong_order = {**full, "ids": list(reversed(full["ids"]))}
    with pytest.raises(ValueError, match="not paired"):
        paired_interval(full, wrong_order, 0)
