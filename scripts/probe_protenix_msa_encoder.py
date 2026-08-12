"""
PROBE: faithful AF3/Protenix MSA embedding extraction on ONE example.

Goal (paper §6.8.1): run the real Protenix featurization + model trunk with default
structure-prediction parameters, halt after the MSAModule step, dump the pair
representation z, and mean-pool over the second dim -> m_seq (L, 128).

This REPLACES the shortcut in preprocessing.extract_msa_embedding_protenix, which fed
z=0, s_inputs=0, has_deletion=0, deletion_value=0 (MSA-only, out-of-distribution).

WHY A PROBE FIRST
-----------------
This path needs the FULL Protenix model (not just msa_module), real atom/relpos/deletion
featurization (biotite/rdkit), a GPU, and the Protenix checkpoint. None of that is
available on the login node. Run this on a compute node with the Protenix env to (a)
confirm the featurizer + full model + MSAModule-hook approach yields a sane (L, 128),
and (b) surface the exact Protenix APIs so we can then wire it into build_lmdb.

All APIs below were verified against the vendored Protenix source (MSA_FLOW/Protenix)
by mirroring infer_dataloader.process_one and runner/inference.py.

USAGE (on a GPU node, inside the Protenix env):
    python scripts/probe_protenix_msa_encoder.py \
        --input_a3m /path/to/one.a3m \
        --protenix_ckpt /path/to/protenix_base_default_v1.0.0.pt \
        --device cuda
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "Protenix"))


def parse_a3m_query_and_string(a3m_path: str):
    """Return (query_seq_ungapped, full_a3m_string). Query = first record, gaps removed."""
    text = Path(a3m_path).read_text()
    # first sequence record
    lines = text.splitlines()
    seqs, cur = [], []
    for ln in lines:
        if ln.startswith(">"):
            if cur:
                seqs.append("".join(cur)); cur = []
        else:
            cur.append(ln.strip())
    if cur:
        seqs.append("".join(cur))
    query = seqs[0].replace("-", "").upper()
    # keep only uppercase/gap for the query length reference (match columns)
    return query, text


def build_feature_dict(query_seq: str, a3m_string: str, device: torch.device):
    """Reuse Protenix's inference featurizer to build the real input_feature_dict.

    Mirrors protenix.data.inference.infer_dataloader.InferenceDataset.process_one.
    """
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
    from protenix.data.utils import data_type_transform, make_dummy_feature
    from protenix.utils.torch_utils import dict_to_tensor

    # Protenix takes the MSA inline via `unpairedMsa` (a3m string) — no temp file needed.
    sample_dict = {
        "name": "probe",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": query_seq,
                    "count": 1,
                    "unpairedMsa": a3m_string,
                    "pairedMsa": "",             # no pairing for a monomer
                }
            }
        ],
    }

    # 1) structure/token features (restype, residue_index, token_bonds, atom feats...)
    s2f = SampleDictToFeatures(sample_dict)
    feat, atom_array, token_array = s2f.get_feature_dict()
    feat["distogram_rep_atom_mask"] = torch.Tensor(
        atom_array.distogram_rep_atom_mask
    ).long()

    # 2) MSA features (msa, has_deletion, deletion_value, profile, deletion_mean)
    msa_feat = InferenceMSAFeaturizer.make_msa_feature(
        bioassembly=sample_dict["sequences"],
        atom_array=atom_array,
        msa_pair_as_unpair=True,   # monomer: treat as unpaired
        use_rna_msa=False,
    )
    dummy_feats = ["template"]     # no templates in this probe -> dummy template feats
    if len(msa_feat) == 0:
        dummy_feats.append("msa")
    else:
        feat.update(dict_to_tensor(msa_feat))
    feat = make_dummy_feature(features_dict=feat, dummy_feats=dummy_feats)
    feat = data_type_transform(feat_or_label_dict=feat)

    # move to device; the model accepts unbatched [..., N_token] features
    feat = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feat.items()}
    return feat


def load_full_protenix(protenix_ckpt: str, device: torch.device,
                       model_name: str = "protenix_base_default_v1.0.0"):
    """Build the FULL Protenix model and load the whole checkpoint (not just msa_module).

    Mirrors runner/inference.py::run(): base + data + inference configs, deep-updated
    with the model-specific overrides, then parsed into an attribute-style ConfigDict.
    """
    from copy import deepcopy
    from typing import Mapping

    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    from protenix.model.protenix import Protenix

    merged = deepcopy({**configs_base, "data": data_configs, **inference_configs})

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    deep_update(merged, model_configs[model_name])
    configs = parse_configs(configs=merged, fill_required_with_null=True)

    model = Protenix(configs).to(device).eval()
    state = torch.load(protenix_ckpt, map_location=device, weights_only=False)
    sd = state.get("model", state)
    # strip a possible "module." DDP prefix
    sd = { (k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items() }
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  e.g. missing:", missing[:5])
    return model


@torch.no_grad()
def extract_embedding(model, feat: dict, n_cycle: int = 4):
    """Run the trunk, capture msa_module output via a hook, mean-pool -> (L, 128)."""
    from protenix.model.protenix import update_input_feature_dict

    # Protenix.forward() preprocesses the feature dict before the trunk — mirror it:
    # relp for RelativePositionEncoding; d_lm/v_lm/pad_info for AtomAttentionEncoder.
    feat = model.relative_position_encoding.generate_relp(feat)
    feat = update_input_feature_dict(feat)

    captured = []                          # one entry per recycling cycle

    def hook(_module, _inp, out):
        z = out.detach()                   # out: [..., L, L, c_z]
        if z.dim() == 4:                   # [B, L, L, c_z]
            z = z[0]
        captured.append(z.mean(dim=1).float().cpu())   # (L, 128) — paper Eq.4

    h = model.msa_module.register_forward_hook(hook)
    try:
        # get_pairformer_output(input_feature_dict, N_cycle) -> (s_inputs, s, z)
        _ = model.get_pairformer_output(feat, N_cycle=n_cycle)
    finally:
        h.remove()

    assert len(captured) == n_cycle, f"hook fired {len(captured)}x, expected {n_cycle}"
    return captured                        # list of (L, 128), one per cycle


def plot_cycle_embeddings(m_list, out_png: str):
    """Heatmap per cycle + convergence curves across cycles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(m_list)
    stacked = torch.stack(m_list)                       # (n, L, 128)
    vmax = stacked.abs().quantile(0.99).item()          # robust shared color scale

    fig, axes = plt.subplots(n + 1, 1, figsize=(12, 2.2 * n + 3.5),
                             gridspec_kw={"height_ratios": [1] * n + [1.4]})
    for c, (ax, m) in enumerate(zip(axes[:-1], m_list), start=1):
        im = ax.imshow(m.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_ylabel(f"cycle {c}\ndim (128)")
        ax.set_xticks([])
    axes[n - 1].set_xlabel("residue index (L)")
    fig.colorbar(im, ax=axes[:-1].tolist(), fraction=0.02, pad=0.01)

    # Convergence: relative change + cosine similarity between consecutive cycles
    ax = axes[-1]
    cycles = list(range(2, n + 1))
    rel_diff = [((m_list[i] - m_list[i - 1]).norm() / m_list[i].norm()).item()
                for i in range(1, n)]
    cos_sim = [torch.nn.functional.cosine_similarity(
                   m_list[i].flatten(), m_list[i - 1].flatten(), dim=0).item()
               for i in range(1, n)]
    ax.plot(cycles, rel_diff, "o-", label="||m_c − m_{c−1}|| / ||m_c||")
    ax.set_xlabel("cycle"); ax.set_ylabel("relative change")
    ax.set_xticks(cycles); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(cycles, cos_sim, "s--", color="tab:red", label="cosine sim vs prev")
    ax2.set_ylabel("cosine similarity", color="tab:red")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax.set_title("m_seq convergence across recycling cycles")

    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_a3m", required=True)
    ap.add_argument("--protenix_ckpt", required=True)
    ap.add_argument("--n_cycle", type=int, default=10)  # base-model inference default
    ap.add_argument("--out_dir", default="logs")        # npz + png output dir
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    query, a3m = parse_a3m_query_and_string(args.input_a3m)
    print(f"[probe] query length L={len(query)}  a3m bytes={len(a3m)}")

    feat = build_feature_dict(query, a3m, device)
    print(f"[probe] feature dict keys ({len(feat)}):", sorted(feat.keys()))
    for k in ("msa", "has_deletion", "deletion_value", "restype", "profile"):
        if k in feat and torch.is_tensor(feat[k]):
            print(f"   {k}: {tuple(feat[k].shape)} {feat[k].dtype}")

    model = load_full_protenix(args.protenix_ckpt, device)
    m_list = extract_embedding(model, feat, n_cycle=args.n_cycle)

    for c, m in enumerate(m_list, start=1):
        prev = m_list[c - 2] if c > 1 else None
        rel = ((m - prev).norm() / m.norm()).item() if prev is not None else float("nan")
        print(f"[probe] cycle {c}: shape={tuple(m.shape)} mean={m.mean():.4f} "
              f"std={m.std():.4f} finite={torch.isfinite(m).all().item()} "
              f"rel_change_vs_prev={rel:.4f}")

    m_seq = m_list[-1]
    assert m_seq.shape == (len(query), 128), "shape mismatch — see feature dict / hook"

    jid = os.environ.get("SLURM_JOB_ID", "local")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"probe_cycles_{jid}.npz"
    png_path = out_dir / f"probe_cycles_{jid}.png"
    np.savez_compressed(npz_path, **{f"cycle_{c+1}": m.numpy() for c, m in enumerate(m_list)})
    plot_cycle_embeddings(m_list, str(png_path))
    print(f"[probe] saved embeddings: {npz_path}")
    print(f"[probe] saved plot     : {png_path}")
    print("[probe] OK — faithful (L,128) embedding produced.")


if __name__ == "__main__":
    main()
