# Contributing to TorchLeet

## The shape of a problem

One directory is one problem. It holds the question notebook, usually a solution
notebook, and a `problem.toml` manifest:

```
v3/llm-inference/kv-cache/
├── kv-cache.ipynb          question (TODOs for the solver)
├── kv-cache_SOLN.ipynb     solution
└── problem.toml            metadata + grading contract
```

`problem.toml` is the source of truth. `PROBLEMS.md`, `SOURCES.md`, the README
counts and the grader registry are all generated from the manifests, so **do not
edit those by hand** - CI regenerates and diffs them.

```bash
python3 scripts/generate.py check      # validate manifests (CI gate)
python3 scripts/generate.py coverage   # how much is auto-graded
python3 scripts/generate.py badges     # refresh Colab badges
python3 scripts/generate.py readme     # rebuild PROBLEMS.md + README counts
```

## Adding a problem

1. Create the directory and the question notebook. Leave `TODO` markers where the
   solver has to write code.
2. Add the solution notebook.
3. Write `problem.toml` (copy a neighbour's). `grading.entries` lists the
   functions or classes the solver must supply.
4. Run `python3 scripts/generate.py check`.

## Adding grading to a problem

Graded problems have a spec at `python/torchleet/problems/<id>.py` (leading
underscore if the id starts with a digit). A spec declares `ENTRIES`, optional
`DEVICE`/`EXTRAS`/`HINTS`, and a list of `CHECKS`.

**Write checks that verify properties, not stored outputs.** We do not ship
reference implementations, and we do not diff a solver's numbers against ours. A
correct solution that initializes weights differently, consumes randomness in a
different order, or uses a different-but-equivalent op must still pass. In
practice that means:

- **Self-consistency** - drive the solver's own code two ways that must agree.
  Cached decoding vs full recomputation; a merged LoRA layer vs an unmerged one;
  a sliding window as wide as the sequence vs full attention. Strongest option:
  it needs no oracle at all.
- **A torch built-in as the oracle** - `torch.softmax`, `torch.sigmoid`,
  `F.binary_cross_entropy`. It is already installed, so nothing ships.
- **Invariants** - shapes, dtypes, ranges, "rows sum to 1", gradients reach every
  parameter, no NaN on large inputs, loss decreases when training.

Also add at least one check that a *wrong* implementation fails. A check that
everything passes tests nothing.

Failure messages are read by someone who is stuck: say what was expected, what
happened, and where to look.

```bash
python3 scripts/validate_specs.py <id>   # grade the repo's own solution
python3 python/test_grader.py            # regression tests
```

`validate_specs.py` must stay green. If the canonical solution cannot pass your
checks, the checks are wrong.

## Devices and extras

Set `grading.extras` to what the **check** needs, not what the notebook imports.
Loading CIFAR-10 needs torchvision; verifying a CNN's shapes does not. Anything
that genuinely needs a GPU raises `Skip` so it is reported as unverified rather
than quietly passed.

## Reporting an interview question

If you were asked one of these in a real interview, please
[tell us](../../issues/new?template=interview-report.yml). That is how a company
tag moves from *inferred* to *reported*. Never share anything an NDA covers - a
description in your own words is what we need.
