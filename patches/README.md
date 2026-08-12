# Submodule patches

The `Protenix/` and `esm/` submodules have local modifications (mainly GPU
compatibility for Blackwell / sm_120) that the superproject does **not** capture
— a superproject only records each submodule's commit SHA, not working-tree edits.

These patches let you reproduce that working state on a fresh machine.

## Contents

| File | What |
|------|------|
| `submodule-commits.txt` | Base commit each patch was generated against |
| `protenix-gpu-compat.patch` | Protenix: layer_norm loader, fused_ops, pairformer, requirements |
| `esm-version.patch` | esm: version.py |
| `apply_submodule_patches.sh` | Idempotent apply script (skips if already applied) |

## Re-setup on a new node

```bash
git clone git@github.com:skkutyson/msaflow_test.git MSA_FLOW
cd MSA_FLOW
git checkout wip/dev-snapshot-20260812

git submodule update --init --recursive     # fetch Protenix + esm at recorded SHAs
bash patches/apply_submodule_patches.sh      # re-apply local GPU-compat edits
```

If a patch reports it does not apply cleanly, the submodule is likely at a
different commit than `submodule-commits.txt` — check that out first, or resolve
the conflict manually.
