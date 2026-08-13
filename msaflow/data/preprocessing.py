"""
MSAFlow data preprocessing pipeline.

Two preprocessing stages:

1. extract_msa_embedding_protenix()
   Runs the FULL Protenix trunk (real featurization + recycling via
   get_pairformer_output), captures the MSAModule pair output z with a forward
   hook, and mean-pools the last cycle to the compressed representation (L×128).
   This is the faithful AF3-encoder step described in Section 3.1 and 6.8.1,
   verified end-to-end by scripts/probe_protenix_msa_encoder.py.
   Requires CCD files in $PROTENIX_ROOT_DIR/common (default: $HOME/common).

2. extract_esm_embeddings()
   Computes ESM2-650M representations (L×1280) for query sequences and stores
   them alongside the MSA embeddings.

Both functions write to the same LMDB, keyed by protein/MSA ID.

LMDB schema per entry (pickled dict):
  {
    "msa_emb":     torch.Tensor  [L, 128]   – compressed MSA embedding
    "esm_emb":     torch.Tensor  [L, 1280]  – ESM2 query embedding
    "msa_tokens":  torch.Tensor  [N, L]     – integer token ids (0..21)
    "weights":     torch.Tensor  [N]        – Neff reweighting
    "query_seq":   str                      – query sequence (no gaps)
    "seq_len":     int                      – L
  }
"""

import os
import sys
import pickle
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import lmdb
from tqdm import tqdm

# ── Alphabet shared between MSA tokenisation and SFM decoder ─────────────────
# 20 canonical AAs (ACDEFGHIKLMNPQRSTVWY), gap '-', unknown 'X'
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "X"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
VOCAB_SIZE = len(AA_LIST)  # 22

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing statistics tracker
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessStats:
    """Accumulates per-entry statistics for end-of-run summary."""
    n_total: int = 0
    n_written: int = 0
    skip_reasons: dict = field(default_factory=lambda: defaultdict(int))
    seq_lens: list = field(default_factory=list)
    msa_depths: list = field(default_factory=list)
    neff_values: list = field(default_factory=list)
    t_start: float = field(default_factory=time.time)

    def skip(self, reason: str) -> None:
        self.skip_reasons[reason] += 1

    def record(self, seq_len: int, msa_depth: int, weights: np.ndarray) -> None:
        self.n_written += 1
        self.seq_lens.append(seq_len)
        self.msa_depths.append(msa_depth)
        self.neff_values.append(float(weights.sum()))

    def eta_str(self, file_idx: int) -> str:
        elapsed = time.time() - self.t_start
        if file_idx == 0:
            return "N/A"
        rate = file_idx / elapsed          # files/sec
        remaining = (self.n_total - file_idx) / max(rate, 1e-9)
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def log_progress(self, file_idx: int) -> None:
        elapsed = time.time() - self.t_start
        rate = file_idx / max(elapsed, 1e-9)
        gpu_mem = ""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_mem = f"  GPU {alloc:.1f}/{reserved:.1f}GB"
        logger.info(
            "Progress  [%d/%d]  written=%d  skip=%d  "
            "%.1f files/s  ETA=%s%s",
            file_idx, self.n_total,
            self.n_written,
            sum(self.skip_reasons.values()),
            rate,
            self.eta_str(file_idx),
            gpu_mem,
        )

    def log_summary(self) -> None:
        elapsed = time.time() - self.t_start
        n_skip = sum(self.skip_reasons.values())
        logger.info("=" * 60)
        logger.info("Preprocessing complete")
        logger.info("  Total files     : %d", self.n_total)
        logger.info("  Written         : %d  (%.1f%%)",
                    self.n_written, 100 * self.n_written / max(self.n_total, 1))
        logger.info("  Skipped         : %d", n_skip)
        for reason, count in sorted(self.skip_reasons.items(),
                                     key=lambda x: -x[1]):
            logger.info("    %-30s : %d", reason, count)
        logger.info("  Elapsed         : %.1f s", elapsed)
        if self.seq_lens:
            sl = np.array(self.seq_lens)
            md = np.array(self.msa_depths)
            neff = np.array(self.neff_values)
            logger.info("  seq_len   mean=%.0f  med=%.0f  min=%d  max=%d",
                        sl.mean(), np.median(sl), sl.min(), sl.max())
            logger.info("  msa_depth mean=%.0f  med=%.0f  min=%d  max=%d",
                        md.mean(), np.median(md), md.min(), md.max())
            logger.info("  Neff      mean=%.1f  med=%.1f  min=%.1f  max=%.1f",
                        neff.mean(), np.median(neff), neff.min(), neff.max())
        logger.info("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# MSA I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_a3m(path: str) -> tuple[list[str], list[str]]:
    """Parse A3M file; remove insertion columns (lowercase letters)."""
    names, seqs = [], []
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append("".join(buf))
                name = line[1:]
                names.append(name)
                buf = []
            else:
                buf.append("".join(c for c in line if c.isupper() or c == "-"))
        if name is not None:
            seqs.append("".join(buf))
    return names, seqs


def filter_msa(seqs: list[str], max_gap_frac: float = 0.1, min_seqs: int = 10) -> list[str]:
    """Keep sequences where at most max_gap_frac of columns are gaps."""
    if not seqs:
        return seqs
    L = len(seqs[0])
    filtered = [s for s in seqs if s.count("-") / max(L, 1) <= max_gap_frac]
    return filtered if len(filtered) >= min_seqs else seqs[:min_seqs]


def tokenise_msa(seqs: list[str]) -> np.ndarray:
    """Convert list of aligned sequences to integer array (N, L)."""
    N = len(seqs)
    L = len(seqs[0])
    arr = np.zeros((N, L), dtype=np.int32)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            arr[i, j] = AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"])
    return arr


def compute_sequence_weights(tokens: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Neff reweighting: w_i = 1 / |{j : hamming(i,j) < threshold}|

    This is the scheme used in the paper (Section 6.8.2):
        w_i = (1 + Σ_{j≠i} 1{d_hamming(x_i, x_j) < 0.2})^{-1}

    Vectorised implementation: O(N²·L) via broadcasting — ~100× faster than
    the naive Python double loop.
    """
    N, _ = tokens.shape
    if N == 1:
        return np.ones(1, dtype=np.float32)
    # Vectorised pairwise Hamming: (N, N) fraction of differing positions
    diff = (tokens[:, None, :] != tokens[None, :, :]).mean(axis=-1)  # (N, N)
    n_sim = (diff < threshold).sum(axis=1).astype(np.float32)        # includes self
    return 1.0 / n_sim


# ──────────────────────────────────────────────────────────────────────────────
# MSA embedding extraction via Protenix (faithful path)
#
# Verified end-to-end by scripts/probe_protenix_msa_encoder.py: real Protenix
# featurization + full model trunk with recycling, halted after the MSAModule.
# Replaces the old shortcut that fed z=0 / s_inputs=0 into a bare MSAModule
# (MSA-only, out-of-distribution).
# ──────────────────────────────────────────────────────────────────────────────

def _seqs_to_a3m(seqs: list[str]) -> str:
    """Serialise processed aligned sequences back to an A3M string.

    Insertion columns were already stripped by parse_a3m(), so this A3M holds
    match columns only — consistent with the msa_tokens stored in the entry.
    The first record must be the (gap-free) query.
    """
    return "\n".join(f">seq{i}\n{s}" for i, s in enumerate(seqs))


def _build_protenix_feature_dict(
    query_seq: str,
    a3m_string: str,
    device: torch.device,
) -> dict:
    """Build the real Protenix input_feature_dict for one monomer.

    Mirrors protenix.data.inference.infer_dataloader.InferenceDataset.process_one:
    structure/token/atom features via SampleDictToFeatures, MSA features via
    InferenceMSAFeaturizer, dummy template features.
    Requires CCD files in $PROTENIX_ROOT_DIR/common (default: $HOME/common).
    """
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.msa.msa_featurizer import InferenceMSAFeaturizer
    from protenix.data.utils import data_type_transform, make_dummy_feature
    from protenix.utils.torch_utils import dict_to_tensor

    # Protenix takes the MSA inline via `unpairedMsa` (a3m string) — no temp file.
    sample_dict = {
        "name": "msaflow",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": query_seq,
                    "count": 1,
                    "unpairedMsa": a3m_string,
                    "pairedMsa": "",           # monomer: no pairing
                }
            }
        ],
    }

    # 1) structure/token features (restype, residue_index, token_bonds, atom feats...)
    s2f = SampleDictToFeatures(sample_dict)
    feat, atom_array, _token_array = s2f.get_feature_dict()
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
    dummy_feats = ["template"]     # no templates -> dummy template feats
    if len(msa_feat) == 0:
        dummy_feats.append("msa")
    else:
        feat.update(dict_to_tensor(msa_feat))
    feat = make_dummy_feature(features_dict=feat, dummy_feats=dummy_feats)
    feat = data_type_transform(feat_or_label_dict=feat)

    # the model accepts unbatched [..., N_token] features
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feat.items()}


def load_protenix_model(
    checkpoint: str,
    device: torch.device,
    model_name: str = "protenix_base_default_v1.0.0",
):
    """Build the FULL Protenix model and load the whole checkpoint.

    Mirrors runner/inference.py::run(): base + data + inference configs,
    deep-updated with the model-specific overrides, then parsed into an
    attribute-style ConfigDict (configs_base alone is a plain dict and fails).
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
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    sd = state.get("model", state)
    # strip a possible "module." DDP prefix
    sd = {(k[len("module."):] if k.startswith("module.") else k): v
          for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info("Loaded full Protenix (%s)  missing=%d unexpected=%d",
                model_name, len(missing), len(unexpected))
    return model


@torch.no_grad()
def extract_msa_embedding_protenix(
    seqs: list[str],
    protenix_model,
    device: torch.device,
    n_cycle: int = 3,
) -> torch.Tensor:
    """
    Extract the compressed MSA embedding via the full Protenix trunk
    (mirrors Section 3.1 & 6.8.1 of the paper):

      1. Real Protenix featurization of query + MSA.
      2. Run get_pairformer_output with N_cycle recycling iterations; a forward
         hook captures the MSAModule pair output z ∈ R^(L×L×128) each cycle.
      3. Mean-pool the LAST cycle along the second spatial dim → m_seq (L, 128).

    Geometry runs in fp32 (no autocast), matching the verified probe numerics.

    Args:
        seqs:           Aligned sequences (match columns only, same length L).
                        seqs[0] must be the gap-free query.
        protenix_model: Full Protenix model from load_protenix_model().
        device:         Compute device.
        n_cycle:        Recycling cycles (paper uses few; 3 = MSAFlow default).

    Returns:
        m_seq: (L, 128) compressed MSA embedding (last cycle).
    """
    from protenix.model.protenix import update_input_feature_dict

    query_seq = seqs[0].replace("-", "")
    logger.info("Extracting MSA embedding  ----- %d seqs, L=%d, n_cycle=%d",
                len(seqs), len(query_seq), n_cycle)

    feat = _build_protenix_feature_dict(query_seq, _seqs_to_a3m(seqs), device)

    # Protenix.forward() preprocesses the feature dict before the trunk — mirror it:
    # relp for RelativePositionEncoding; d_lm/v_lm/pad_info for AtomAttentionEncoder.
    feat = protenix_model.relative_position_encoding.generate_relp(feat)
    feat = update_input_feature_dict(feat)

    captured = []                          # one entry per recycling cycle

    def hook(_module, _inp, out):
        z = out.detach()                   # out: [..., L, L, c_z]
        if z.dim() == 4:                   # [B, L, L, c_z]
            z = z[0]
        captured.append(z.mean(dim=1).float().cpu())   # (L, 128) — paper Eq. 4

    h = protenix_model.msa_module.register_forward_hook(hook)
    try:
        # get_pairformer_output(input_feature_dict, N_cycle) -> (s_inputs, s, z)
        protenix_model.get_pairformer_output(feat, N_cycle=n_cycle)
    finally:
        h.remove()

    if len(captured) != n_cycle:
        raise RuntimeError(
            f"msa_module hook fired {len(captured)}x, expected {n_cycle}"
        )
    return captured[-1]


# ──────────────────────────────────────────────────────────────────────────────
# ESM2 embedding extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_esm_embedding(
    query_seq: str,
    esm_model,
    alphabet,
    device: torch.device,
    layer: int = 33,
) -> torch.Tensor:
    """
    Extract ESM2-650M representation for a single query sequence.

    Args:
        query_seq: Amino acid sequence (no gaps), length L.
        esm_model: Loaded ESM2 model.
        alphabet:  ESM2 Alphabet object.
        device:    Compute device.
        layer:     Representation layer (33 for ESM2-650M).

    Returns:
        emb: (L, 1280) ESM2 representation (BOS/EOS tokens removed).
    """
    logger.info("Extracting ESM2 embedding  ----- query length=%d", len(query_seq))
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("query", query_seq)])
    tokens = tokens.to(device)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16):
        out = esm_model(tokens, repr_layers=[layer], return_contacts=False)

    emb = out["representations"][layer][0, 1:-1, :]  # remove BOS/EOS → (L, 1280)
    return emb.cpu()


# ──────────────────────────────────────────────────────────────────────────────
# Main LMDB building routine
# ──────────────────────────────────────────────────────────────────────────────

# A3M filenames that identify the source database rather than the protein.
# OpenProteinSet/OpenFold layouts nest these under a per-protein directory
# ({uniprot_id}/a3m/uniclust30.a3m or {pdb}_{chain}/uniclust30_hits.a3m), so
# the LMDB key must come from the directory name, not the file stem.
_DB_A3M_STEMS = {
    "uniclust30", "uniclust30_hits", "bfd_uniclust_hits",
    "mgnify_hits", "uniref90_hits", "uniprot_hits",
}


def _a3m_key(path: Path) -> str:
    """Derive the LMDB key for an a3m file (flat or OpenProteinSet layout)."""
    if path.stem not in _DB_A3M_STEMS:
        return path.stem  # flat layout: {pdb_id}__{chain_id}.a3m
    parent = path.parent
    if parent.name == "a3m":
        parent = parent.parent
    return parent.name


def build_lmdb(
    a3m_dir: str,
    output_path: str,
    protenix_checkpoint: Optional[str] = None,
    max_msa_seqs: int = 512,
    max_seq_len: int = 1024,
    device: str = "cuda",
    map_size_gb: int = 500,
    protein_list: Optional[str] = None,
    n_cycle: int = 3,
    a3m_name: Optional[str] = None,
    resume: bool = False,
    a3m_list: Optional[str] = None,
):
    """
    Process all A3M files in a3m_dir and write LMDB database.

    Args:
        a3m_dir:              Directory containing *.a3m files.
        output_path:          Path for the output LMDB.
        protenix_checkpoint:  Path to Protenix checkpoint. If None, skip MSA emb.
        max_msa_seqs:         Maximum MSA depth to use.
        max_seq_len:          Maximum sequence length.
        device:               Compute device string.
        map_size_gb:          LMDB map size in GB.
        protein_list:         Optional path to a text file with one protein ID per line.
                              When provided, only those proteins are processed.
        n_cycle:              Protenix recycling cycles for MSA embedding extraction.
        a3m_name:             Exact a3m filename to process (e.g. "uniclust30.a3m").
                              Required for OpenProteinSet-style trees where each
                              protein dir holds several db a3ms; None = every .a3m.
        resume:               Skip proteins whose key already exists in the LMDB.
        a3m_list:             Text file with one a3m path per line. Skips the
                              directory scan; defaults to the auto-written cache
                              at <output>.a3m_list.txt when that exists.
    """
    logger.info("build_lmdb started")
    logger.info("  a3m_dir    : %s", a3m_dir)
    logger.info("  output     : %s", output_path)
    logger.info("  max_seq_len: %d  max_msa_seqs: %d", max_seq_len, max_msa_seqs)
    logger.info("  device     : %s  map_size: %dGB", device, map_size_gb)
    logger.info("  protenix   : %s  n_cycle: %d", protenix_checkpoint or "disabled", n_cycle)
    logger.info("  a3m_name   : %s  resume: %s", a3m_name or "*.a3m", resume)

    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load the FULL Protenix model if checkpoint provided (faithful path needs
    # the whole trunk, not just the MSAModule)
    protenix_model = None
    if protenix_checkpoint is not None:
        sys.path.insert(0, str(Path(__file__).parents[2] / "Protenix"))
        protenix_model = load_protenix_model(protenix_checkpoint, dev)

    # Load ESM2-650M
    sys.path.insert(0, str(Path(__file__).parents[2] / "esm"))
    import esm as esm_lib
    esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(dev)
    if torch.cuda.is_available():
        logger.info("ESM2 loaded  GPU mem: %.1f GB allocated",
                    torch.cuda.memory_allocated() / 1024**3)
    else:
        logger.info("ESM2 loaded  (CPU mode)")

    # Scanning a large tree (OpenProteinSet on GPFS: 100k+ dirs) can take tens
    # of minutes, so the file list is cached next to the LMDB and reused on
    # restarts. Pass --a3m_list to use a precomputed list (e.g. from `find`).
    list_cache = Path(a3m_list) if a3m_list else Path(f"{output_path}.a3m_list.txt")
    if list_cache.is_file():
        with open(list_cache) as f:
            a3m_files = [Path(line.strip()) for line in f if line.strip()]
        logger.info("Loaded %d a3m paths from %s (delete it to rescan)",
                    len(a3m_files), list_cache)
    else:
        a3m_files = []
        if a3m_name:
            # One readdir of the top level, then construct candidate paths
            # directly — per-directory stat calls are what make glob/find take
            # hours on GPFS-scale trees. Missing candidates are skipped (with a
            # warning) at processing time instead of being stat-checked here.
            logger.info("Listing top level of %s (single readdir)", a3m_dir)
            with os.scandir(a3m_dir) as it:
                names = sorted(e.name for e in it)
            logger.info("%d top-level entries", len(names))
            for rel in (("a3m", a3m_name), (a3m_name,)):
                candidates = [Path(a3m_dir, n, *rel) for n in names]
                if any(p.is_file() for p in candidates[:50]):
                    a3m_files = candidates
                    break
            if not a3m_files:
                logger.info("Layout probe failed — falling back to recursive scan "
                            "(may take a long time on network storage)")
                a3m_files = sorted(Path(a3m_dir).glob(f"**/{a3m_name}"))
        else:
            logger.info("Scanning %s for **/*.a3m (recursive — may take a long "
                        "time on network storage)", a3m_dir)
            a3m_files = sorted(Path(a3m_dir).glob("**/*.a3m"))
        logger.info("Found %d A3M files total", len(a3m_files))
        if a3m_files:
            with open(list_cache, "w") as f:
                f.writelines(f"{p}\n" for p in a3m_files)
            logger.info("Cached file list to %s", list_cache)

    if protein_list is not None:
        if not Path(protein_list).is_file():
            raise FileNotFoundError(
                f"--protein_list points to a missing file: {protein_list!r} "
                f"(cwd={Path.cwd()}). Pass an existing file with one protein ID "
                f"per line, or omit --protein_list to process every a3m in --a3m_dir."
            )
        with open(protein_list) as f:
            allowed_ids = {line.strip() for line in f if line.strip()}
        a3m_files = [p for p in a3m_files if _a3m_key(p) in allowed_ids]
        logger.info("Filtered to %d A3M files from protein_list (%s)",
                    len(a3m_files), protein_list)
        if not a3m_files:
            logger.warning("protein_list %s matched 0 of the a3m files in %s "
                           "(check IDs match the derived LMDB keys)",
                           protein_list, a3m_dir)

    n_keys = len({_a3m_key(p) for p in a3m_files})
    if n_keys < len(a3m_files):
        logger.warning(
            "%d a3m files map to only %d distinct keys — later files overwrite "
            "earlier ones. Pass --a3m_name (e.g. uniclust30.a3m) to pick one db "
            "per protein.", len(a3m_files), n_keys,
        )

    env = lmdb.open(
        output_path,
        map_size=map_size_gb * (1024 ** 3),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    if resume:
        with env.begin() as txn:
            existing = {k.decode() for k, _ in txn.cursor()}
        n_before = len(a3m_files)
        a3m_files = [p for p in a3m_files if _a3m_key(p) not in existing]
        logger.info("Resume: %d keys already in LMDB, %d a3m files remaining",
                    n_before - len(a3m_files), len(a3m_files))

    stats = PreprocessStats(n_total=len(a3m_files))

    for file_idx, a3m_path in enumerate(tqdm(a3m_files, desc="Building LMDB")):
        key = _a3m_key(a3m_path)
        try:
            # ── Parse ────────────────────────────────────────────────────────
            _names, seqs = parse_a3m(str(a3m_path))
            if not seqs:
                stats.skip("empty_file")
                continue

            n_seqs_raw = len(seqs)
            seqs = filter_msa(seqs)
            seqs = [s[:max_seq_len] for s in seqs]
            L = len(seqs[0])

            if L == 0:
                stats.skip("zero_length")
                continue
            if L > max_seq_len:
                stats.skip("seq_too_long")
                continue

            query_seq = seqs[0].replace("-", "")
            L_query = len(query_seq)
            if L_query == 0:
                stats.skip("query_all_gaps")
                continue
            # Faithful featurization requires the first A3M record (query) to be
            # gap-free so it matches the proteinChain sequence exactly.
            if protenix_model is not None and "-" in seqs[0]:
                stats.skip("query_has_gaps")
                continue

            seqs = [s[:L_query] for s in seqs]
            tokens = tokenise_msa(seqs[:max_msa_seqs])   # (N, L_query)
            weights = compute_sequence_weights(tokens)    # (N,)
            N = tokens.shape[0]

            logger.debug(
                "[%s] raw_seqs=%d  after_filter=%d  used=%d  L_aligned=%d  L_query=%d  Neff=%.1f",
                key, n_seqs_raw, len(seqs), N, L, L_query, weights.sum(),
            )

            entry = {
                "msa_tokens": tokens,
                "weights": weights,
                "query_seq": query_seq,
                "seq_len": L_query,
                "msa_emb": None,
                "esm_emb": None,
            }

            # ── ESM2 embedding ───────────────────────────────────────────────
            entry["esm_emb"] = extract_esm_embedding(
                query_seq, esm_model, alphabet, dev
            ).half().numpy()
            esm_shape = entry["esm_emb"].shape
            if esm_shape[0] != L_query:
                stats.skip("esm_length_mismatch")
                logger.warning("[%s] ESM2 shape %s != L_query %d", key, esm_shape, L_query)
                continue

            # Protenix MSA embedding (faithful full-trunk path)
            if protenix_model is not None:
                entry["msa_emb"] = extract_msa_embedding_protenix(
                    seqs[:max_msa_seqs], protenix_model, dev, n_cycle=n_cycle
                ).half().numpy()
                msa_shape = entry["msa_emb"].shape
                if msa_shape[0] != L_query:
                    stats.skip("msa_emb_length_mismatch")
                    logger.warning("[%s] msa_emb shape %s != L_query %d", key, msa_shape, L_query)
                    continue

            # ── Write ────────────────────────────────────────────────────────
            with env.begin(write=True) as txn:
                txn.put(key.encode(), pickle.dumps(entry))
            stats.record(L_query, N, weights)

        except Exception as exc:
            logger.warning("Skipped %s: %s", key, exc)
        finally:
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    env.close()
    stats.log_summary()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    parser = argparse.ArgumentParser(description="Build MSAFlow LMDB dataset")
    parser.add_argument("--a3m_dir",              required=True)
    parser.add_argument("--output",               required=True)
    parser.add_argument("--protenix_checkpoint",  default=None)
    parser.add_argument("--max_msa_seqs",         type=int, default=512)
    parser.add_argument("--max_seq_len",          type=int, default=1024)
    parser.add_argument("--device",               default="cuda")
    parser.add_argument("--map_size_gb",          type=int, default=500)
    parser.add_argument("--protein_list",          default=None,
                        help="Text file with one protein ID per line to process (optional)")
    parser.add_argument("--n_cycle",              type=int, default=3,
                        help="Protenix recycling cycles (3 = MSAFlow default)")
    parser.add_argument("--a3m_name",             default=None,
                        help="Exact a3m filename to process (e.g. uniclust30.a3m); "
                             "default processes every .a3m under --a3m_dir")
    parser.add_argument("--resume",               action="store_true",
                        help="Skip proteins whose key already exists in the LMDB")
    parser.add_argument("--a3m_list",             default=None,
                        help="Text file with one a3m path per line (skips the "
                             "directory scan; auto-cached at <output>.a3m_list.txt)")
    args = parser.parse_args()

    build_lmdb(
        a3m_dir=args.a3m_dir,
        output_path=args.output,
        protenix_checkpoint=args.protenix_checkpoint,
        max_msa_seqs=args.max_msa_seqs,
        max_seq_len=args.max_seq_len,
        device=args.device,
        map_size_gb=args.map_size_gb,
        protein_list=args.protein_list,
        n_cycle=args.n_cycle,
        a3m_name=args.a3m_name,
        resume=args.resume,
        a3m_list=args.a3m_list,
    )
