"""Independent implementation of HKD equations (5)-(10), arXiv:2005.00727.

The paper excludes self-pairs; its released code includes them. We follow the
paper on this point, and average over B*B like the released implementation.
"""
import torch
from torch import nn
from torch.nn import functional as F


def relation_probabilities(features, kernel, eps=1e-6):
    x = features.float().flatten(1)
    if x.shape[0] < 3:
        raise ValueError("HKD needs at least 3 samples; use >=8 for this pilot. Accumulation is not a substitute.")
    if kernel == "cosine":
        normalized = F.normalize(x, dim=1, eps=eps)
        similarities = ((normalized @ normalized.T + 1) / 2).clamp_min(eps)
    elif kernel == "student_t":
        # Eq. (7): d=1, Euclidean distance, NOT squared distance.
        squared = x.square().sum(1, keepdim=True)
        distance = (squared + squared.T - 2 * (x @ x.T)).clamp_min(0)
        similarities = 1 / (1 + (distance + eps).sqrt())
    else:
        raise ValueError(kernel)
    off_diagonal = ~torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
    similarities = similarities * off_diagonal
    return similarities / similarities.sum(1, keepdim=True).clamp_min(eps)


def hybrid_loss(student, teacher):
    if student.shape[0] != teacher.shape[0]:
        raise ValueError("Teacher and student must describe the same sample batch.")
    loss = student.float().sum() * 0
    for kernel in ("cosine", "student_t"):
        p = relation_probabilities(teacher.detach(), kernel)
        q = relation_probabilities(student, kernel)
        loss = loss + ((p - q) * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).mean()
    return loss


def information_flow_loss(student_features, auxiliary_features, epoch):
    if len(student_features) != 4 or len(auxiliary_features) != 4:
        raise ValueError("Expected three convolutional stages and one penultimate layer.")
    weights = [100 * 0.7 ** epoch] * 3 + [1.0]
    return sum(w * hybrid_loss(s, t) for w, s, t in zip(weights, student_features, auxiliary_features))


class SmallCNN(nn.Module):
    """CNN-1 (width=1) / CNN-1-A (width=2), adapted to 224px with 2x2 pooling.

    Same 3 conv + 2 linear layers as the paper; the auxiliary has twice the
    channels and hidden units. Adaptive pooling keeps the original FC sizes.
    """
    def __init__(self, width=1, classes=10):
        super().__init__()
        channels = [3, 8 * width, 16 * width, 32 * width]
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Conv2d(a, b, 3), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
            for a, b in zip(channels[:-1], channels[1:])
        ])
        self.embedding = nn.Linear(channels[-1] * 4, 64 * width)
        self.head = nn.Linear(64 * width, classes)

    def forward(self, images):
        features = []
        x = images
        for stage in self.stages:
            x = stage(x)
            features.append(x.flatten(1))
        # A CNN saliency proxy, not the original MaskedKD CLS attention.
        saliency = F.adaptive_avg_pool2d(x.detach().float().square().mean(1, keepdim=True), (14, 14)).flatten(1)
        embedding = F.relu(self.embedding(F.adaptive_avg_pool2d(x, (2, 2)).flatten(1)))
        features.append(embedding)
        return self.head(embedding), features, saliency


def select_patches(saliency, keep):
    if not 1 <= keep <= saliency.shape[1]:
        raise ValueError("Invalid patch budget")
    # Resolve equal saliency deterministically; retain original spatial order.
    return saliency.detach().argsort(dim=1, descending=True, stable=True)[:, :keep].sort(dim=1).values


class MaskableTeacher(nn.Module):
    """Frozen, non-distilled timm DeiT. Remove tokens before the first block."""
    def __init__(self, model):
        super().__init__()
        if model.num_prefix_tokens != 1 or model.no_embed_class:
            raise ValueError("Use non-distilled DeiT with a single CLS token.")
        self.model = model.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images, indices=None):
        m = self.model
        tokens = m._pos_embed(m.patch_embed(images))
        if indices is not None:
            patches = tokens[:, 1:].gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))
            tokens = torch.cat([tokens[:, :1], patches], dim=1)
        tokens = m.norm_pre(m.patch_drop(tokens))
        tokens = m.norm(m.blocks(tokens))
        return m.forward_head(tokens, pre_logits=True)

    @torch.no_grad()
    def chunks(self, images, indices, size):
        return torch.cat([
            self(images[i:i + size], None if indices is None else indices[i:i + size]).float()
            for i in range(0, len(images), size)
        ])
