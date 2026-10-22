"""sft-smollm — supervised fine-tuning data path: prompt-masked examples, padding,
the shifted token-level loss, and gradient-accumulation normalization.

Checks use a tiny position-wise LM, so nothing is downloaded and no `transformers`
is needed; the SmolLM fine-tune itself is exercised by the notebook.
"""
import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

ENTRIES = ["build_sft_example", "sft_collate", "sft_loss", "sft_train_step"]
DEVICE = "cpu"
EXTRAS = []

HINTS = [
    "labels are a copy of input_ids (prompt + response + end-of-turn) with every prompt position "
    "set to -100; keep them aligned with input_ids and shift inside the loss.",
    "Pad input_ids with pad_id but labels with -100, so a pad id that equals a real token id "
    "never erases a real label.",
    "sft_loss: cross_entropy(logits[:, :-1], labels[:, 1:], ignore_index=-100, reduction='sum') "
    "divided by the trained-token count of the whole optimizer step, counted across all "
    "micro-batches before any backward.",
]

IGNORE = -100


def _lists(ex):
    return [int(v) for v in ex["input_ids"]], [int(v) for v in ex["labels"]]


# ------------------------------------------------------------------- examples
def check_example_masks_prompt_and_keeps_eos(ns):
    ids, labels = _lists(ns.build_sft_example([5, 6, 7], [8, 9], 2))
    assert ids == [5, 6, 7, 8, 9, 2], f"input_ids should be prompt + response + eos, got {ids}"
    assert labels == [IGNORE, IGNORE, IGNORE, 8, 9, 2], (
        f"labels should be -100 on the prompt and the response + eos elsewhere, got {labels}")


def check_example_truncates_and_accepts_tensors(ns):
    ids, labels = _lists(ns.build_sft_example(torch.tensor([5, 6, 7]), torch.tensor([8, 9]), 2, max_len=4))
    assert ids == [5, 6, 7, 8] and labels == [IGNORE, IGNORE, IGNORE, 8], (
        f"max_len=4 should keep the first 4 tokens of both, got input_ids={ids}, labels={labels}")
    ex = ns.build_sft_example([1], [3], 2)
    assert torch.as_tensor(ex["input_ids"]).dtype == torch.long, "return torch.long tensors"


# ---------------------------------------------------------------------- collate
def check_collate_pads_right_with_ignored_labels(ns):
    eos = pad = 2
    exs = [ns.build_sft_example([5, 6], [7], eos), ns.build_sft_example([5, 6, 7], [8, 9, 10], eos),
           ns.build_sft_example([5], [], eos)]
    b = ns.sft_collate(exs, pad_id=pad)
    ids, am, lab = (torch.as_tensor(b[k]) for k in ("input_ids", "attention_mask", "labels"))
    assert tuple(ids.shape) == (3, 7) and tuple(lab.shape) == (3, 7) and tuple(am.shape) == (3, 7), \
        f"expected (3, 7) tensors, got input_ids {tuple(ids.shape)}"
    assert am.sum(1).tolist() == [4, 7, 2], f"attention_mask should mark the real tokens, got {am.tolist()}"
    assert ids[0].tolist() == [5, 6, 7, 2, 2, 2, 2], f"right-pad with pad_id, got {ids[0].tolist()}"
    assert lab[0].tolist() == [IGNORE, IGNORE, 7, 2, IGNORE, IGNORE, IGNORE], (
        "pad the labels with -100 and keep the real end-of-turn label even when it equals pad_id; "
        f"got {lab[0].tolist()}")


# ------------------------------------------------------------------------ loss
def _loss_batch():
    torch.manual_seed(0)
    B, T, V = 3, 6, 13
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    labels[0, :2] = IGNORE
    labels[1, :4] = IGNORE
    labels[2, 3:] = IGNORE
    return logits, labels


def check_loss_is_shifted_token_mean(ns):
    logits, labels = _loss_batch()
    got = torch.as_tensor(ns.sft_loss(logits, labels))
    assert got.ndim == 0, f"the loss must be a scalar, got shape {tuple(got.shape)}"
    exp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                          ignore_index=IGNORE)
    unshifted = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=IGNORE)
    assert torch.allclose(got.float(), exp, atol=1e-5), (
        f"expected the token mean of shifted cross-entropy ({float(exp):.4f}), got {float(got):.4f}"
        + (" — that is the UNshifted loss: logits[:, :-1] predict labels[:, 1:]"
           if torch.allclose(got.float(), unshifted, atol=1e-5) else ""))


def check_loss_ignores_unused_logits(ns):
    logits, labels = _loss_batch()
    base = torch.as_tensor(ns.sft_loss(logits, labels))
    unused = torch.zeros(labels.shape, dtype=torch.bool)
    unused[:, :-1] = labels[:, 1:] == IGNORE
    unused[:, -1] = True
    noisy = logits.clone()
    noisy[unused] = 100 * torch.randn(int(unused.sum()), logits.size(-1))
    assert torch.allclose(torch.as_tensor(ns.sft_loss(noisy, labels)), base, atol=1e-5), \
        "logits whose next label is -100 (and the last logit) must not affect the loss"
    all_ignored = torch.full(labels.shape, IGNORE)
    assert torch.isfinite(torch.as_tensor(ns.sft_loss(logits, all_ignored))), \
        "a batch with nothing to train must not give nan"


def check_loss_num_items_makes_microbatches_add_up(ns):
    logits, labels = _loss_batch()
    full = torch.as_tensor(ns.sft_loss(logits, labels))
    n = int((labels[:, 1:] != IGNORE).sum())
    parts = (torch.as_tensor(ns.sft_loss(logits[:2], labels[:2], n))
             + torch.as_tensor(ns.sft_loss(logits[2:], labels[2:], n)))
    assert torch.allclose(parts, full, atol=1e-5), (
        "with num_items_in_batch set to the step's token count, micro-batch losses must sum to the "
        f"full-batch loss ({float(parts):.4f} vs {float(full):.4f})")


# ------------------------------------------------------------------ train step
class _TinyLM(nn.Module):
    """Position-wise LM with the Hugging Face output contract."""

    def __init__(self, vocab=20, dim=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(torch.tanh(self.emb(input_ids))))


def _uneven_examples(ns):
    eos = 19
    return [ns.build_sft_example([1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11], eos),
            ns.build_sft_example([1, 2], [12], eos),
            ns.build_sft_example([3, 4, 5, 6], [7, 8], eos),
            ns.build_sft_example([2], [13, 14, 15], eos)]


def check_microbatching_matches_one_big_batch(ns):
    torch.manual_seed(0)
    exs = _uneven_examples(ns)
    full = ns.sft_collate(exs, 0)
    split = [ns.sft_collate(exs[:2], 0), ns.sft_collate(exs[2:], 0)]

    a = _TinyLM()
    b, ref = copy.deepcopy(a), copy.deepcopy(a)
    labels = torch.as_tensor(full["labels"])
    with torch.no_grad():
        lg = ref(input_ids=torch.as_tensor(full["input_ids"])).logits
    expected = float(F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), labels[:, 1:].reshape(-1),
                                     ignore_index=IGNORE))

    loss_split = float(ns.sft_train_step(a, split, torch.optim.SGD(a.parameters(), lr=0.5)))
    loss_whole = float(ns.sft_train_step(b, [full], torch.optim.SGD(b.parameters(), lr=0.5)))
    assert abs(loss_whole - expected) < 1e-4, \
        f"the step should return the token-mean loss ({expected:.4f}), got {loss_whole:.4f}"
    assert abs(loss_split - loss_whole) < 1e-4, (
        f"micro-batched step loss {loss_split:.4f} != one-batch {loss_whole:.4f}; count trained tokens "
        "over all micro-batches before the backward passes")
    for (name, pa), pb in zip(a.named_parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6), (
            f"parameter '{name}' differs between micro-batched and one-batch steps; averaging per-"
            "micro-batch means weights tokens unequally")


def check_train_step_learns(ns):
    torch.manual_seed(1)
    exs = _uneven_examples(ns)
    model = _TinyLM()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    batches = [ns.sft_collate(exs[:2], 0), ns.sft_collate(exs[2:], 0)]
    first = float(ns.sft_train_step(model, batches, opt))
    for _ in range(50):
        last = float(ns.sft_train_step(model, batches, opt, max_grad_norm=1.0))
    assert last < 0.5 * first, f"loss should fall on a fixed batch ({first:.3f} -> {last:.3f})"


CHECKS = [check_example_masks_prompt_and_keeps_eos, check_example_truncates_and_accepts_tensors,
          check_collate_pads_right_with_ignored_labels, check_loss_is_shifted_token_mean,
          check_loss_ignores_unused_logits, check_loss_num_items_makes_microbatches_add_up,
          check_microbatching_matches_one_big_batch, check_train_step_learns]
