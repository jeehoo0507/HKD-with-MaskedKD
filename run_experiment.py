"""Paired COCO pilot: HKD versus HKD with CNN-guided teacher token removal."""
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import timm
import torch
from torch.nn import functional as F

from hkd.core import SmallCNN, MaskableTeacher, hybrid_loss, information_flow_loss, select_patches
from hkd.data import CLASSES, loader, read_manifest, sha256_file

TEACHER_URL = "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth"
METHODS = ("hkd", "hkd_masked")


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()


def optimizer_for(model, device):
    return torch.optim.Adam(model.parameters(), lr=1e-3), torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, hours):
        self.started = time.monotonic()
        self.deadline = self.started + hours * 3600

    def check(self, reserve=0):
        if time.monotonic() + reserve >= self.deadline:
            raise BudgetExceeded("Time budget reached; no winner is reported from an unfinished pair.")

    def remaining(self):
        return self.deadline - time.monotonic()


def train_step(model, optimizer, scaler, batch, device, phase, teacher=None,
               keep=196, micro=1, epoch=0):
    images, labels, _ = batch
    images, labels = images.to(device), labels.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast(device):
        logits, features, saliency = model(images)
        if phase == "aux":
            indices = select_patches(saliency, keep) if keep < 196 else None
            targets = teacher.chunks(images, indices, micro)
        elif phase == "student":
            with torch.no_grad():
                _, targets, _ = teacher(images)
    ce = F.cross_entropy(logits.float(), labels)
    # Outside autocast: relation probabilities/logarithms must stay in FP32.
    if phase == "aux":
        kd = hybrid_loss(features[-1], targets)
    elif phase == "student":
        kd = information_flow_loss(features, targets, epoch)
    else:
        kd = ce.detach() * 0
    # Official HKD auxiliary transfer uses only the penultimate relation loss,
    # after supervised auxiliary initialization. The final classification task
    # adds CE, as described in the paper's Appendix A.3.
    loss = kd if phase == "aux" else ce + kd
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite training loss")
    old_scale = scaler.get_scale()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return {"loss": loss.item(), "ce": ce.item(), "kd": kd.item(),
            "skipped": int(scaler.get_scale() < old_scale)}


def scores(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    per_class = [float((p[y == c] == c).mean() * 100) if (y == c).any() else None for c in range(10)]
    present = [x for x in per_class if x is not None]
    return {"macro_accuracy": float(np.mean(present)), "accuracy": float((y == p).mean() * 100),
            "per_class": dict(zip(CLASSES, per_class))}


@torch.no_grad()
def evaluate(model, batches, device, budget=None):
    model.eval()
    labels, predictions, ids = [], [], []
    for images, y, sample_ids in batches:
        if budget:
            budget.check()
        with autocast(device):
            logits, _, _ = model(images.to(device))
        labels.extend(y.tolist())
        predictions.extend(logits.argmax(1).cpu().tolist())
        ids.extend(sample_ids.tolist())
    return {**scores(labels, predictions), "labels": labels, "predictions": predictions, "ids": ids}


def fetch_teacher():
    directory = Path(__file__).resolve().parent / "checkpoints"
    torch.hub.load_state_dict_from_url(TEACHER_URL, model_dir=str(directory), map_location="cpu",
                                        check_hash=True, weights_only=True)
    path = directory / TEACHER_URL.rsplit("/", 1)[-1]
    if not sha256_file(path).startswith("b5f2ef4d"):
        raise RuntimeError(f"Invalid teacher checksum: {path}")
    return path


def load_teacher(path, device):
    # Require an ImageNet 1000-class non-distilled checkpoint. No random fallback.
    net = timm.create_model("deit_base_patch16_224", pretrained=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return MaskableTeacher(net).to(device)


def memory_peak(device):
    if device.type != "cuda":
        return {"peak_allocated_gib": 0, "peak_reserved_gib": 0}
    return {"peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}


def reset_peak(device):
    if device.type == "cuda":
        # Do not charge a masked run for cached blocks left by the full teacher.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def measure(args, rows, teacher, device, batch_size, micro, phase, keep, budget):
    seed_all(args.seed)
    model = SmallCNN(width=1 if phase == "student" else 2).to(device)
    guide = SmallCNN(width=2).eval().requires_grad_(False).to(device) if phase == "student" else teacher
    optimizer, scaler = optimizer_for(model, device)
    batches = iter(loader(args.data, rows, "train", batch_size, args.seed, workers=args.workers))
    reset_peak(device)
    elapsed = []
    for i in range(5):
        budget.check(reserve=120)
        sync(device)
        start = time.monotonic()
        batch = next(batches)
        train_step(model, optimizer, scaler, batch, device, phase, guide, keep, micro)
        sync(device)
        if i >= 2:
            elapsed.append(time.monotonic() - start)
    val = iter(loader(args.data, rows, "val", batch_size, workers=args.workers))
    sync(device)
    start = time.monotonic()
    evaluate(model, [next(val), next(val)], device, budget)
    sync(device)
    return {"train_step_s": float(np.median(elapsed)), "validation_batch_s": (time.monotonic() - start) / 2,
            **memory_peak(device)}


def make_plan(measurements, train_size, val_size, batch, remaining):
    steps, val_steps = train_size // batch, math.ceil(val_size / batch)
    costs = {k: v["train_step_s"] * steps + v["validation_batch_s"] * val_steps
             for k, v in measurements.items()}
    # Keep the original epoch-indexed 100 * 0.7**epoch schedule. Do not compress it.
    # Pick solely using runtime, never validation/test performance.
    for aux_epochs, student_epochs in ((20, 50), (15, 50), (10, 50), (10, 30)):
        estimate = 1.4 * (5 * costs["warm"] + aux_epochs * (costs["aux_full"] + costs["aux_masked"])
                          + 2 * student_epochs * costs["student"]) + 180
        if estimate <= remaining:
            return {"warm_epochs": 5, "aux_epochs": aux_epochs, "student_epochs": student_epochs,
                    "estimated_remaining_s_with_margin": estimate, "epoch_costs_s": costs}
    raise BudgetExceeded(f"Minimum plan needs about {estimate / 3600:.2f} h including margin, "
                         f"but only {remaining / 3600:.2f} h remain. "
                         "Use a faster GPU or a larger time budget; the protocol was not silently shortened.")


def preflight(args, rows, teacher, device, budget):
    attempts = []
    for batch in (32, 16, 8):
        for micro in (4, 2, 1):
            try:
                measurements = {}
                for name, phase, keep in (("aux_full", "aux", 196), ("aux_masked", "aux", 98),
                                          ("student", "student", 196), ("warm", "warm", 196)):
                    measurements[name] = measure(args, rows, teacher, device, batch, micro, phase, keep, budget)
                plan = make_plan(measurements, sum(r["split"] == "train" for r in rows),
                                 sum(r["split"] == "val" for r in rows), batch, budget.remaining())
                return {**plan, "batch_size": batch, "teacher_microbatch": micro,
                        "measurements": measurements, "oom_attempts": attempts}
            except torch.cuda.OutOfMemoryError:
                attempts.append({"batch": batch, "teacher_microbatch": micro})
                # The failed measurement's local models/graphs are released on unwind.
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    raise RuntimeError("No configuration fits the allocator budget, even batch 8 / teacher microbatch 1.")


def train_epoch(args, rows, model, opt, scaler, device, phase, epoch, batch, micro, budget,
                teacher=None, keep=196):
    totals = dict(loss=0.0, ce=0.0, kd=0.0, skipped=0)
    batches = loader(args.data, rows, "train", batch, args.seed, epoch, args.workers)
    for samples in batches:
        budget.check(reserve=90)
        values = train_step(model, opt, scaler, samples, device, phase, teacher, keep, micro, epoch)
        for key, value in values.items():
            totals[key] += value
    return {key: value if key == "skipped" else value / len(batches) for key, value in totals.items()}


def paired_stage(args, rows, models, guides, phase, epochs, plan, device, budget, output):
    opts = {m: optimizer_for(models[m], device) for m in METHODS}
    best = {m: -1.0 for m in METHODS}
    best_epoch = {}
    costs = {m: 0.0 for m in METHODS}
    peaks = {m: dict(peak_allocated_gib=0, peak_reserved_gib=0) for m in METHODS}
    history = []
    # Assert identical initialization for the two experimental arms.
    hashes = {m: state_hash(models[m]) for m in METHODS}
    if len(set(hashes.values())) != 1:
        raise RuntimeError("Paired models have different initial weights")
    for epoch in range(epochs):
        for method in METHODS if epoch % 2 == 0 else METHODS[::-1]:
            sync(device)
            start = time.monotonic()
            reset_peak(device)
            opt, scaler = opts[method]
            train = train_epoch(args, rows, models[method], opt, scaler, device, phase, epoch,
                                plan["batch_size"], plan["teacher_microbatch"], budget,
                                guides[method], 98 if method == "hkd_masked" and phase == "aux" else 196)
            val = evaluate(models[method], loader(args.data, rows, "val", plan["batch_size"], workers=args.workers),
                           device, budget)
            sync(device)
            elapsed = time.monotonic() - start
            costs[method] += elapsed
            for key, value in memory_peak(device).items():
                peaks[method][key] = max(peaks[method][key], value)
            score = val["macro_accuracy"]
            if score > best[method]:
                best[method], best_epoch[method] = score, epoch + 1
                torch.save(cpu_state(models[method]), output / f"{phase}_{method}_best.pt")
            row = dict(phase=phase, method=method, epoch=epoch + 1, val_macro_accuracy=score,
                       epoch_seconds=elapsed, **train)
            history.append(row)
            print(json.dumps(row), flush=True)
        # Preserve complete paired epochs, including optimizer states, for recovery.
        torch.save({"phase": phase, "completed_paired_epochs": epoch + 1, "plan": plan,
                    "models": {m: cpu_state(models[m]) for m in METHODS},
                    "optimizers": {m: opts[m][0].state_dict() for m in METHODS},
                    "scalers": {m: opts[m][1].state_dict() for m in METHODS},
                    "best": best, "best_epoch": best_epoch, "history": history,
                    "seed": args.seed}, output / f"{phase}_paired_progress.pt")
        save_json(output / f"{phase}_history.json", history)
    for method in METHODS:
        torch.save(cpu_state(models[method]), output / f"{phase}_{method}_last.pt")
    return {"seconds": costs, "peaks": peaks, "best_epoch": best_epoch, "initial_sha256": hashes}


def paired_interval(full, masked, seed):
    if full["ids"] != masked["ids"] or full["labels"] != masked["labels"]:
        raise ValueError("Predictions are not paired")
    y = np.asarray(full["labels"])
    difference = (np.asarray(masked["predictions"]) == y).astype(float) - (np.asarray(full["predictions"]) == y)
    groups = [difference[y == c] for c in range(10)]
    if any(len(g) == 0 for g in groups):
        raise ValueError("All classes required for stratified macro bootstrap")
    rng = np.random.default_rng(seed)
    draws = [np.mean([rng.choice(g, len(g), replace=True).mean() for g in groups]) * 100 for _ in range(2000)]
    return list(map(float, np.percentile(draws, [2.5, 97.5])))


def source_provenance():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted([root / "run_experiment.py", *list((root / "hkd").glob("*.py"))]):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {"source_sha256": h.hexdigest(), "git_commit": commit.stdout.strip() if commit.returncode == 0 else None}


def run(args):
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty. Use a new directory to preserve existing runs.")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires an NVIDIA CUDA GPU. Use 'smoke' for the offline CPU check.")
    if not 2 <= args.vram_gb <= 4 or not 3 <= args.hours <= 5:
        raise ValueError("This pilot is designed for --vram-gb 2..4 and --hours 3..5.")
    free, total = torch.cuda.mem_get_info(device)
    # CUDA context/cuDNN allocations are outside PyTorch's allocator cap.
    allocation_limit = min(args.vram_gb * 2**30, free) - 0.6 * 2**30
    if allocation_limit < 0.8 * 2**30:
        raise RuntimeError("Insufficient free GPU memory; leave other GPU jobs untouched.")
    torch.cuda.set_per_process_memory_fraction(allocation_limit / total, device)
    rows = read_manifest(args.data)
    teacher_path = Path(args.teacher_checkpoint) if args.teacher_checkpoint else fetch_teacher()
    budget = Budget(args.hours)  # Downloads and dataset preparation are explicitly excluded.
    metadata = {"paper": "https://arxiv.org/abs/2005.00727", "seed": args.seed,
                "teacher": "deit_base_patch16_224", "teacher_checkpoint": str(teacher_path.resolve()),
                "teacher_sha256": sha256_file(teacher_path), "manifest_sha256": sha256_file(Path(args.data) / "manifest.json"),
                "classes": CLASSES, "dataset": "coco_single", "torch": torch.__version__, "timm": timm.__version__,
                "gpu": torch.cuda.get_device_name(device), "allocator_limit_gib": allocation_limit / 2**30,
                "requested_vram_gib": args.vram_gb, "hours": args.hours,
                "benchmark_commit": "e81cc3063dc5f98a2d43366f2e86b08905e80875",
                "hkd_reference_commit": "5766e6398c574598c0b166bfadfaf573a7190a61", **source_provenance()}
    save_json(output / "metadata.json", metadata)
    try:
        teacher = load_teacher(teacher_path, device)
        plan = preflight(args, rows, teacher, device, budget)
        save_json(output / "plan.json", plan)
        print(json.dumps(plan, indent=2), flush=True)
        if args.plan_only:
            save_json(output / "status.json", {"status": "profiled_only", "results": "No accuracy experiment performed"})
            return
        seed_all(args.seed)
        auxiliary = SmallCNN(width=2).to(device)
        opt, scaler = optimizer_for(auxiliary, device)
        teacher.cpu()
        torch.cuda.empty_cache()
        start = time.monotonic()
        for epoch in range(plan["warm_epochs"]):
            values = train_epoch(args, rows, auxiliary, opt, scaler, device, "warm", epoch,
                                 plan["batch_size"], 1, budget)
            print(json.dumps(dict(phase="shared_aux_warmup", epoch=epoch + 1, **values)), flush=True)
        sync(device)
        warm_seconds = time.monotonic() - start
        torch.save(cpu_state(auxiliary), output / "aux_shared_initial.pt")
        auxiliaries = {m: copy.deepcopy(auxiliary) for m in METHODS}
        del auxiliary, opt, scaler
        teacher.to(device)
        aux_stats = paired_stage(args, rows, auxiliaries, {m: teacher for m in METHODS}, "aux",
                                 plan["aux_epochs"], plan, device, budget, output)
        del teacher
        torch.cuda.empty_cache()
        for m in METHODS:
            auxiliaries[m].load_state_dict(torch.load(output / f"aux_{m}_best.pt", weights_only=True, map_location="cpu"))
            auxiliaries[m].eval().requires_grad_(False)
        seed_all(args.seed + 10000)
        initial_student = SmallCNN(width=1).to(device)
        torch.save(cpu_state(initial_student), output / "student_shared_initial.pt")
        students = {m: copy.deepcopy(initial_student) for m in METHODS}
        del initial_student
        student_stats = paired_stage(args, rows, students, auxiliaries, "student", plan["student_epochs"],
                                     plan, device, budget, output)
        records, predictions = [], {}
        for method in METHODS:
            test_loader = loader(args.data, rows, "test", plan["batch_size"], workers=args.workers)
            last = evaluate(students[method], test_loader, device, budget)
            students[method].load_state_dict(torch.load(output / f"student_{method}_best.pt", weights_only=True, map_location="cpu"))
            selected = evaluate(students[method], test_loader, device, budget)
            predictions[method] = selected
            save_json(output / f"test_{method}.json", {"validation_selected": selected, "last_epoch": last})
            records.append({"method": method, "seed": args.seed, "patches": 196 if method == "hkd" else 98,
                            "val_selected_test_macro": selected["macro_accuracy"], "last_test_macro": last["macro_accuracy"],
                            "val_selected_test_accuracy": selected["accuracy"], "selected_epoch": student_stats["best_epoch"][method],
                            "aux_selected_epoch": aux_stats["best_epoch"][method],
                            "aux_seconds": aux_stats["seconds"][method], "student_seconds": student_stats["seconds"][method],
                            "branch_seconds": aux_stats["seconds"][method] + student_stats["seconds"][method],
                            "shared_warmup_seconds": warm_seconds,
                            "peak_reserved_gib": max(aux_stats["peaks"][method]["peak_reserved_gib"], student_stats["peaks"][method]["peak_reserved_gib"]),
                            "peak_allocated_gib": max(aux_stats["peaks"][method]["peak_allocated_gib"], student_stats["peaks"][method]["peak_allocated_gib"]),
                            "student_initial_sha256": student_stats["initial_sha256"][method],
                            "aux_initial_sha256": aux_stats["initial_sha256"][method],
                            "manifest_sha256": metadata["manifest_sha256"], "teacher_sha256": metadata["teacher_sha256"],
                            "source_sha256": metadata["source_sha256"]})
        with (output / "runs.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        full, masked = records
        summary = {"status": "complete", "macro_delta_pp_masked_minus_full": masked["val_selected_test_macro"] - full["val_selected_test_macro"],
                   "paired_test_bootstrap_95pct_pp": paired_interval(predictions["hkd"], predictions["hkd_masked"], args.seed),
                   "branch_speedup_full_over_masked": full["branch_seconds"] / masked["branch_seconds"],
                   "aux_stage_speedup_full_over_masked": full["aux_seconds"] / masked["aux_seconds"],
                   "wall_seconds_including_profile": time.monotonic() - budget.started,
                   "interpretation": "Single-seed pilot. Bootstrap covers test-sample uncertainty, not training-seed variance. "
                                     "Compare accuracy and time separately; no general winner is established."}
        save_json(output / "summary.json", summary)
        save_json(output / "status.json", {"status": "complete"})
        print(json.dumps(summary, indent=2), flush=True)
    except BudgetExceeded as exc:
        save_json(output / "status.json", {"status": "incomplete_budget", "reason": str(exc)})
        raise SystemExit(str(exc)) from exc
    except Exception as exc:
        save_json(output / "status.json", {"status": "failed", "reason": str(exc)})
        raise


def smoke():
    """No downloads, real dataset, accuracy claims, or GPU resource estimates."""
    from timm.models.vision_transformer import VisionTransformer
    torch.set_num_threads(2)
    seed_all(0)
    device = torch.device("cpu")
    teacher = MaskableTeacher(VisionTransformer(img_size=224, patch_size=16, embed_dim=48, depth=2,
                                                num_heads=3, num_classes=10))
    batch = (torch.randn(8, 3, 224, 224), torch.arange(8) % 10, torch.arange(8))
    aux = SmallCNN(2)
    opt, scaler = optimizer_for(aux, device)
    results = {}
    for keep in (196, 98):
        results[f"aux_keep_{keep}"] = train_step(aux, opt, scaler, batch, device, "aux", teacher, keep, 2)
    student = SmallCNN(1)
    opt, scaler = optimizer_for(student, device)
    aux.eval().requires_grad_(False)
    results["student"] = train_step(student, opt, scaler, batch, device, "student", aux)
    print(json.dumps({"status": "synthetic_cpu_smoke_passed", "losses": results,
                      "student_parameters": sum(p.numel() for p in student.parameters()),
                      "auxiliary_parameters": sum(p.numel() for p in aux.parameters()),
                      "not_measured": ["real accuracy", "CUDA VRAM", "target-GPU runtime"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch-teacher")
    commands.add_parser("smoke")
    train = commands.add_parser("run")
    train.add_argument("--data", required=True, help="COCO single root containing manifest.json")
    train.add_argument("--output", default="outputs/pilot")
    train.add_argument("--teacher-checkpoint", help="Official ImageNet DeiT-B .pth; otherwise download/cache")
    train.add_argument("--hours", type=float, default=4)
    train.add_argument("--vram-gb", type=float, default=3, help="Total target GiB; reserve 0.6 GiB outside the allocator")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.command == "fetch-teacher":
        print(fetch_teacher())
    elif args.command == "smoke":
        smoke()
    else:
        run(args)


if __name__ == "__main__":
    main()
