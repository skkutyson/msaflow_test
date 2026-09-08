#!/usr/bin/env python3
"""
OpenStructure(OST) 기반 folding 결과 채점기 — **chain-mapping 버전**.

`ost_score.py` 와 동일하되, GT 어셈블리에서 gemmi 로 대상 chain 만 떼어내는 대신
원본 어셈블리 GT 를 그대로 레퍼런스로 두고 `compare-structures` 의
`-c/--chain-mapping <model>:<ref>` 옵션으로 대상 chain 만 매핑해 채점한다.

    gemmi 버전 :  예측(A 단량체)  vs  A만 남긴 GT
    이 버전    :  예측(A 단량체)  vs  원본 어셈블리 GT   (-c A:A)

두 버전의 lddt/tm 를 비교해서 `-c A:A` 만으로 gemmi 전처리를 대체할 수 있는지
검증하는 용도. (매핑만 하고 unmapped 체인 contact 가 여전히 페널티로 잡히면
lddt 가 gemmi 버전보다 낮게 나온다.)

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
        # ~ 는 셸이 아니라 여기서 확장(따옴표 안에 있으면 미확장 상태로 넘어옴)
        return [os.path.expanduser(tok) for tok in env_cmd.split()]
    if shutil.which("compare-structures"):
        return ["compare-structures"]
    if shutil.which("ost"):
        return ["ost", "compare-structures"]
    sif = os.path.expanduser(os.environ.get("OST_SIF", "").strip())
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


def run_ost(ost_cmd, model_cif, ref_cif, tmpdir, chain_mapping=None):
    """OST compare-structures 실행 → (lddt, tm_score, gdt_ts, rmsd).

    chain_mapping 이 주어지면 '-c <model>:<ref>' 로 대상 chain 만 매핑한다.
    (gemmi 로 ref 를 미리 자르지 않고 원본 어셈블리 GT 를 그대로 넘길 때 사용)
    """
    out_json = os.path.join(tmpdir, "ost_out.json")
    cmd = ost_cmd + [
        "-m", model_cif,
        "-r", ref_cif,
        "-o", out_json,
        "-mf", "mmcif", "-rf", "mmcif",
        "--lddt", "--rigid-scores", "--tm-score",
        "--min-pep-length", "4",
    ]
    if chain_mapping:
        cmd += ["-c", chain_mapping]
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


def split_target(name):
    """타겟명 '8xjifdj_assembly1__A' → (gt_id='8xjifdj_assembly1', chain='A').

    예측/a3m 은 chain 접미사 '__A' 를 붙이지만 GT 파일은 그게 없는
    '{gt_id}.cif' 이다. 접미사가 없으면 chain 은 None(호출부에서 폴백).
    """
    if "__" in name:
        base, chain = name.rsplit("__", 1)
        return base, chain
    return name, None


def find_gt(gt_dir, gt_id):
    for cand in (f"{gt_id}.cif", f"{gt_id}.cif.gz"):
        p = os.path.join(gt_dir, cand)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description="OST-based folding scorer (chain-mapping 버전, gemmi 미사용)")
    ap.add_argument("--pred_root", required=True, help="fold 출력 root (OUTPUT_DIR/folds)")
    ap.add_argument("--gt_dir", required=True, help="ground-truth CIF 디렉토리")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--mode", default="fewshot", help="zeroshot|fewshot|nomsa|shallow")
    ap.add_argument("--targets", required=True, help="공백 구분 pdb_id 목록")
    ap.add_argument("--chain_csv", default=None, help="pdb_id,chain_id CSV (기본 chain A)")
    ap.add_argument("--model_chain", default=None,
                    help="예측 CIF 의 체인명(-c <model>:<ref>). 기본은 ref chain 과 동일(A:A)")
    ap.add_argument("--score_full_complex", action="store_true",
                    help="-c 없이 GT 어셈블리 전체와 비교(chain 매핑 자동 추정)")
    args = ap.parse_args()

    ost_cmd = resolve_ost_cmd()
    sys.stderr.write(f"OST 커맨드: {' '.join(ost_cmd)}\n")

    chain_map = load_chain_map(args.chain_csv)
    targets = args.targets.split()
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for target in targets:
            # 타겟명의 '__A' 접미사에서 GT id 와 chain 을 분리
            gt_id, chain_from_name = split_target(target)
            chain = chain_from_name or chain_map.get(target) or chain_map.get(gt_id) or "A"

            gt = find_gt(args.gt_dir, gt_id)
            if gt is None:
                sys.stderr.write(f"[skip] GT 없음: {args.gt_dir}/{gt_id}.cif\n")
                continue

            # gemmi 로 ref 를 자르지 않고, 원본 어셈블리 GT 에 대해 '-c model:ref' 매핑
            if args.score_full_complex:
                chain_mapping = None
            else:
                model_chain = args.model_chain or chain
                chain_mapping = f"{model_chain}:{chain}"

            preds = enumerate_predictions(args.pred_root, target)
            if not preds:
                sys.stderr.write(f"[skip] 예측 CIF 없음: {target}\n")
                continue

            for cif, seed, sample, rank_score in preds:
                lddt, tm, gdt, rmsd = run_ost(
                    ost_cmd, cif, gt, tmpdir, chain_mapping=chain_mapping)
                rows.append({
                    "pdb_id": gt_id,
                    "chain_id": chain,
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
                    f"  {gt_id} chain={chain} map={chain_mapping} "
                    f"seed={seed} sample={sample}  "
                    f"lddt={lddt} tm={tm} gdt_ts={gdt} rmsd={rmsd}\n"
                )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[ost_score_chainmap] {len(rows)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
