#!/usr/bin/env python3
"""
OpenStructure(OST) 기반 folding 결과 채점기.

폴딩된 Protenix 예측 CIF들을 ground-truth CIF와 OST `compare-structures`로 비교해
lddt / tm_score / gdt_ts / rmsd 를 계산하고, monomer_protein_ost.csv 형식(+ mode 컬럼)
으로 CSV를 쓴다.  run_fold_benchmark_gpu.sh 의 folding 이후 단계로 호출된다.

예측 파일 레이아웃 (Protenix):
    <pred_root>/<pdb_id>_<tag>/protenix_out/**/<name>_sample_<rank>.cif
    같은 폴더의 <name>_summary_confidence_sample_<rank>.json  → ranking_score
    상위 폴더명 seed_<seed>                                    → seed

OST 실행 경로는 아래 순서로 자동 탐지 (OST_CMD 로 강제 지정 가능):
    1) $OST_CMD (예: "apptainer exec /path/ost.sif ost compare-structures")
    2) PATH 의 `compare-structures`
    3) PATH 의 `ost`  → `ost compare-structures`
    4) $OST_SIF 지정 시 → `apptainer exec $OST_SIF ost compare-structures`
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

CSV_FIELDS = [
    "pdb_id", "chain_id", "seed", "sample", "ranking_score", "prediction_path",
    "dockq_score", "irmsd", "lrmsd", "len_dockq",
    "lddt", "tm_score", "gdt_ts", "rmsd", "mode",
]


# ── OST 실행 커맨드 탐지 ──────────────────────────────────────────────────────
def resolve_ost_cmd():
    env_cmd = os.environ.get("OST_CMD", "").strip()
    if env_cmd:
        return env_cmd.split()
    if shutil.which("compare-structures"):
        return ["compare-structures"]
    if shutil.which("ost"):
        return ["ost", "compare-structures"]
    sif = os.environ.get("OST_SIF", "").strip()
    if sif:
        runner = shutil.which("apptainer") or shutil.which("singularity")
        if runner:
            return [runner, "exec", sif, "ost", "compare-structures"]
    sys.exit(
        "ERROR: OST(compare-structures)를 찾을 수 없습니다.\n"
        "  다음 중 하나로 제공하세요:\n"
        "   - OST_SIF=/path/to/openstructure.sif  (apptainer 이미지)\n"
        "   - OST_CMD='apptainer exec /path/ost.sif ost compare-structures'\n"
        "   - 또는 `ost`/`compare-structures` 를 PATH 에 두기\n"
        "  OST 이미지 예: apptainer pull ost.sif "
        "docker://registry.scicore.unibas.ch/schwede/openstructure:latest"
    )


# ── 중첩 dict 에서 첫 매칭 키 값 찾기 ────────────────────────────────────────
def deep_get(obj, key):
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], (int, float)):
            return obj[key]
        for v in obj.values():
            r = deep_get(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = deep_get(v, key)
            if r is not None:
                return r
    return None


def extract_ref_chain(gt_path, chain_id, out_path):
    """GT(어셈블리, .cif/.cif.gz)에서 지정 chain 만 떼어 단일-체인 mmCIF 로 저장.

    예측이 chain A 단량체이므로, 복합체 전체와 비교하면 사슬간(A-B) contact 를
    예측이 재현 못 한 것으로 lDDT 가 깎인다. chain A 만 남긴 레퍼런스로 비교해야
    순수 monomer 채점이 된다. chain 을 못 찾으면 None 반환(호출부에서 폴백/스킵).
    """
    import gemmi  # venv 에 존재 (0.6.x)
    st = gemmi.read_structure(gt_path)  # .gz 자동 처리
    if len(st) == 0:
        return None
    while len(st) > 1:          # 첫 model 만 유지
        del st[1]
    model = st[0]
    names = [ch.name for ch in model]
    if chain_id not in names:
        return None
    for name in list(names):    # 대상 chain 외 전부 제거
        if name != chain_id:
            model.remove_chain(name)
    st.setup_entities()
    st.make_mmcif_document().write_file(out_path)
    return out_path


def run_ost(ost_cmd, model_cif, ref_cif, tmpdir):
    """OST compare-structures 실행 → (lddt, tm_score, gdt_ts, rmsd)."""
    out_json = os.path.join(tmpdir, "ost_out.json")
    cmd = ost_cmd + [
        "-m", model_cif,
        "-r", ref_cif,
        "-o", out_json,
        "-mf", "mmcif", "-rf", "mmcif",
        "--lddt", "--rigid-scores", "--tm-score",
        "--min-pep-length", "4",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(out_json):
        sys.stderr.write(
            f"  [OST 실패] {os.path.basename(model_cif)}\n"
            f"    cmd: {' '.join(cmd)}\n"
            f"    stderr: {proc.stderr[-800:]}\n"
        )
        return (None, None, None, None)
    with open(out_json) as fh:
        data = json.load(fh)
    return (
        deep_get(data, "lddt"),
        deep_get(data, "tm_score"),
        deep_get(data, "gdt_ts"),
        deep_get(data, "rmsd"),
    )


# ── 예측 파일 열거 ────────────────────────────────────────────────────────────
SEED_RE = re.compile(r"seed[_-](\d+)")
SAMPLE_RE = re.compile(r"sample[_-](\d+)")


def enumerate_predictions(pred_root, pdb_id):
    """(cif_path, seed, sample, ranking_score) 목록 반환."""
    cifs = sorted(glob.glob(
        os.path.join(pred_root, f"{pdb_id}_*", "protenix_out", "**", "*_sample_*.cif"),
        recursive=True,
    ))
    out = []
    for cif in cifs:
        m_seed = SEED_RE.search(cif)
        seed = int(m_seed.group(1)) if m_seed else 101
        m_samp = SAMPLE_RE.search(os.path.basename(cif))
        sample = int(m_samp.group(1)) if m_samp else 0
        # ranking_score: 같은 폴더의 summary_confidence_sample_<rank>.json
        rank_score = ""
        summ = cif.replace(f"_sample_{sample}.cif",
                           f"_summary_confidence_sample_{sample}.json")
        if os.path.exists(summ):
            try:
                with open(summ) as fh:
                    rank_score = json.load(fh).get("ranking_score", "")
            except Exception:
                pass
        out.append((cif, seed, sample, rank_score))
    return out


def load_chain_map(chain_csv):
    chains = {}
    if chain_csv and os.path.exists(chain_csv):
        with open(chain_csv) as fh:
            for row in csv.DictReader(fh):
                chains[row["pdb_id"]] = row.get("chain_id", "A")
    return chains


def find_gt(gt_dir, pdb_id):
    for cand in (f"{pdb_id}.cif", f"{pdb_id}.cif.gz"):
        p = os.path.join(gt_dir, cand)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="OST-based folding scorer")
    ap.add_argument("--pred_root", required=True, help="fold 출력 root (OUTPUT_DIR/folds)")
    ap.add_argument("--gt_dir", required=True, help="ground-truth CIF 디렉토리")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--mode", default="fewshot", help="zeroshot|fewshot|nomsa|shallow")
    ap.add_argument("--targets", required=True, help="공백 구분 pdb_id 목록")
    ap.add_argument("--chain_csv", default=None, help="pdb_id,chain_id CSV (기본 chain A)")
    ap.add_argument("--score_full_complex", action="store_true",
                    help="GT 어셈블리 전체와 비교(기본은 대상 chain 만 떼어 단일-체인 비교)")
    args = ap.parse_args()

    ost_cmd = resolve_ost_cmd()
    sys.stderr.write(f"OST 커맨드: {' '.join(ost_cmd)}\n")

    chain_map = load_chain_map(args.chain_csv)
    targets = args.targets.split()
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for pdb_id in targets:
            gt = find_gt(args.gt_dir, pdb_id)
            if gt is None:
                sys.stderr.write(f"[skip] GT 없음: {args.gt_dir}/{pdb_id}.cif\n")
                continue

            chain = chain_map.get(pdb_id, "A")
            # 기본: GT 어셈블리에서 대상 chain 만 떼어 단일-체인 레퍼런스로 채점
            if not args.score_full_complex:
                ref_single = os.path.join(tmpdir, f"{pdb_id}_chain{chain}.cif")
                extracted = extract_ref_chain(gt, chain, ref_single)
                if extracted is None:
                    sys.stderr.write(
                        f"[warn] {pdb_id}: GT 에 chain '{chain}' 없음 → 어셈블리 전체로 비교\n")
                else:
                    gt = extracted

            preds = enumerate_predictions(args.pred_root, pdb_id)
            if not preds:
                sys.stderr.write(f"[skip] 예측 CIF 없음: {pdb_id}\n")
                continue

            for cif, seed, sample, rank_score in preds:
                lddt, tm, gdt, rmsd = run_ost(ost_cmd, cif, gt, tmpdir)
                rows.append({
                    "pdb_id": pdb_id,
                    "chain_id": chain_map.get(pdb_id, "A"),
                    "seed": seed,
                    "sample": sample,
                    "ranking_score": rank_score,
                    "prediction_path": cif,
                    "dockq_score": "", "irmsd": "", "lrmsd": "", "len_dockq": 0,
                    "lddt": lddt if lddt is not None else "",
                    "tm_score": tm if tm is not None else "",
                    "gdt_ts": gdt if gdt is not None else "",
                    "rmsd": rmsd if rmsd is not None else "",
                    "mode": args.mode,
                })
                sys.stderr.write(
                    f"  {pdb_id} seed={seed} sample={sample}  "
                    f"lddt={lddt} tm={tm} gdt_ts={gdt} rmsd={rmsd}\n"
                )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[ost_score] {len(rows)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
