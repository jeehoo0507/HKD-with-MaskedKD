from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from hkd.core import SmallCNN
from hkd.data import CocoSingle, loader
from run_experiment import Budget, paired_stage, state_hash
from timm.models.vision_transformer import VisionTransformer
from hkd.core import MaskableTeacher


def synthetic_rows(root):
    # Tiny data are only passed to internal tests, never accepted as benchmark data.
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, :16] = [255, 30, 0]
    Image.fromarray(pixels).save(root / "image.png")
    return [dict(id=i, image="image.png", label=i % 10, split="train" if i < 16 else "val")
            for i in range(26)]


def test_identical_batch_order_and_augmentation(tmp_path):
    rows = synthetic_rows(tmp_path)
    a = loader(tmp_path, rows, "train", 8, seed=4, epoch=2)
    b = loader(tmp_path, rows, "train", 8, seed=4, epoch=2)
    for (xa, ya, ia), (xb, yb, ib) in zip(a, b):
        torch.testing.assert_close(xa, xb)
        assert torch.equal(ya, yb) and torch.equal(ia, ib)
    val1 = CocoSingle(tmp_path, rows, "val", 1, 1)
    val2 = CocoSingle(tmp_path, rows, "val", 7, 20)
    torch.testing.assert_close(val1[0][0], val2[0][0])
    assert set(r["id"] for r in a.dataset.rows).isdisjoint(r["id"] for r in val1.rows)


def test_paired_epoch_checkpoint_and_validation_selection(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rows = synthetic_rows(tmp_path)
    output = tmp_path / "results"
    output.mkdir()
    a = SmallCNN(1)
    b = SmallCNN(1)
    b.load_state_dict(a.state_dict())
    args = SimpleNamespace(data=tmp_path, seed=0, workers=0)
    result = paired_stage(args, rows, {"hkd": a, "hkd_masked": b},
                          {"hkd": None, "hkd_masked": None}, "warm", 2,
                          dict(batch_size=8, teacher_microbatch=1), torch.device("cpu"), Budget(1), output)
    assert result["initial_sha256"]["hkd"] == result["initial_sha256"]["hkd_masked"]
    # With no difference in the treatment both branches should stay identical.
    assert state_hash(a) == state_hash(b)
    checkpoint = torch.load(output / "warm_paired_progress.pt", weights_only=True)
    assert checkpoint["completed_paired_epochs"] == 2
    assert len(checkpoint["history"]) == 4
    assert all(1 <= e <= 2 for e in result["best_epoch"].values())
    for method in ("hkd", "hkd_masked"):
        assert (output / f"warm_{method}_best.pt").is_file()
        assert (output / f"warm_{method}_last.pt").is_file()


def test_budget_stops_before_reporting_a_pair(tmp_path):
    rows = synthetic_rows(tmp_path)
    a, b = SmallCNN(1), SmallCNN(1)
    b.load_state_dict(a.state_dict())
    args = SimpleNamespace(data=tmp_path, seed=0, workers=0)
    from run_experiment import BudgetExceeded
    with pytest.raises(BudgetExceeded):
        paired_stage(args, rows, {"hkd": a, "hkd_masked": b}, {"hkd": None, "hkd_masked": None},
                     "warm", 1, dict(batch_size=8, teacher_microbatch=1), torch.device("cpu"), Budget(0), tmp_path)
    assert not (tmp_path / "warm_paired_progress.pt").exists()


def test_both_distillation_stages_complete_with_paired_branches(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    rows = synthetic_rows(tmp_path)
    output = tmp_path / "two_stages"
    output.mkdir()
    teacher = MaskableTeacher(VisionTransformer(img_size=224, patch_size=16, embed_dim=48,
                                                depth=2, num_heads=3, num_classes=10))
    token_counts = []
    hook = teacher.model.blocks[0].register_forward_pre_hook(lambda m, args: token_counts.append(args[0].shape[1]))
    auxiliary = {name: SmallCNN(2) for name in ("hkd", "hkd_masked")}
    auxiliary["hkd_masked"].load_state_dict(auxiliary["hkd"].state_dict())
    args = SimpleNamespace(data=tmp_path, seed=0, workers=0)
    plan = dict(batch_size=8, teacher_microbatch=2)
    paired_stage(args, rows, auxiliary, {m: teacher for m in auxiliary}, "aux", 1,
                 plan, torch.device("cpu"), Budget(1), output)
    hook.remove()
    assert set(token_counts) == {99, 197}
    students = {name: SmallCNN(1) for name in auxiliary}
    students["hkd_masked"].load_state_dict(students["hkd"].state_dict())
    for m in auxiliary:
        auxiliary[m].load_state_dict(torch.load(output / f"aux_{m}_best.pt", weights_only=True))
        auxiliary[m].eval().requires_grad_(False)
    before = {m: state_hash(auxiliary[m]) for m in auxiliary}
    result = paired_stage(args, rows, students, auxiliary, "student", 1,
                          plan, torch.device("cpu"), Budget(1), output)
    assert before == {m: state_hash(auxiliary[m]) for m in auxiliary}
    assert result["best_epoch"] == {"hkd": 1, "hkd_masked": 1}
    assert all((output / f"student_{m}_best.pt").is_file() for m in students)
