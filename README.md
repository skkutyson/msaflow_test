# MSAFlow

**MSAFlow: A Unified Approach for MSA Representation, Augmentation, and Family-based Protein Design**

> Reproduction of the MSAFlow paper (ICLR 2026 under review).  
> Statistical Flow Matching on the Fisher-Rao manifold for generative MSA modeling.

---

## Overview

Multiple Sequence Alignments (MSAs) encode evolutionary information critical for protein structure prediction. MSAFlow addresses two core challenges:

1. **Orphan proteins** — insufficient homologs found by HHBlits/MMSeqs2
2. **Storage efficiency** — compressing thousands of sequences into a compact latent

The framework combines:
- A **generative autoencoder**: AF3 MSAModule encodes an MSA → compressed embedding; SFM decoder reconstructs sequences
- A **latent flow matching model**: maps a single ESM2 embedding → synthetic MSA embedding (zero-shot)

### Three generation modes

`msaflow/inference/generate.py`가 세 가지 생성 경로를 제공합니다 (논문 Section 3.3):

| Mode | 함수 | Input | Use case |
|------|------|-------|----------|
| **Reconstruct** | `reconstruct(msa_seqs)` | Deep MSA | MSA를 Protenix로 인코딩 후 SFM 디코더로 재구성 — 압축·복원 (6.5% storage, pLDDT 89.0) |
| **Few-shot augment** | `augment_shallow(shallow_seqs)` | Shallow MSA | 얕은 MSA 인코딩(Rec 트랙) + ESM2 기반 합성 임베딩(Syn 트랙)을 **결합** (Syn+Rec, Section 6.5) |
| **Zero-shot** | `generate_zeroshot(query_seq)` | Single sequence | ESM2 쿼리 표현만으로 MSA 임베딩을 생성 → 디코딩. MSA 전혀 불필요 → orphan / de novo 단백질 |

모든 모드는 gap 없는 아미노산 서열 리스트를 반환하므로 그대로 FASTA/A3M로 저장해
폴딩기 입력으로 쓸 수 있습니다. `generate_zeroshot_seeds(...)`는 seed별 블록을 yield 해
다양성 분해(within/between-seed) 분석에 사용합니다.

---

## Architecture

```
Single sequence ──► ESM2-650M ──► Latent FM (130M) ──┐
                                                       ▼
Shallow/Deep MSA ──► AF3 MSAModule ──► Compress ──► SFM Decoder (130M) ──► Generated MSA
```

### SFM Decoder
- 12 × DiT blocks, hidden dim 768, 12 attention heads (~130M params)
- **Position-wise AdaLN**: each residue position gets its own (shift, scale, gate) modulated by the per-residue MSA embedding — enabling fine-grained evolutionary conditioning
- Operates on the **Fisher-Rao statistical manifold** via the mapping π: µ → √µ

### Latent FM Encoder
- 12 × DiT blocks, hidden dim 768, 12 attention heads (~130M params)
- Conditional rectified flow: ESM2 embedding (L×1280) → MSA embedding (L×128)
- SDE inference with temperature control for diversity/fidelity tradeoff

---

## Repository Structure

```
msaflow/
├── configs/
│   ├── decoder.yaml             # SFM decoder training config (+ *_alpha05 / *_overfit 변형)
│   ├── latent_fm.yaml           # Latent FM training config (+ *_foldbench)
│   ├── fold_benchmark.yaml      # end-to-end 벤치(증강→폴딩→OST) · MSA 생성도 이 설정을 읽음
│   ├── fold_benchmark_af3.yaml  # AF3(colabfold af3-preview) 백엔드용 변형
│   ├── colabfold_batch.yaml     # colabfold_batch 공통 플래그
│   └── accelerate_{1,2,4}gpu.yaml  # 멀티-GPU accelerate 런치 설정
├── data/
│   ├── dataset.py               # MSADecoderDataset, LatentFMDataset
│   ├── preprocessing.py         # ShardStore 빌더 (Protenix MSA + ESM2 임베딩)
│   └── shard_store.py           # 샤드 단위 임베딩 저장소 (LMDB 대체)
├── models/
│   ├── sfm_decoder.py           # SFM decoder (DiT + position-wise AdaLN)
│   └── latent_fm.py             # Latent FM encoder + SDE sampler
├── training/
│   ├── train_decoder.py         # Decoder training loop (Accelerate)
│   └── train_latent_fm.py       # Latent FM training loop (Accelerate)
├── inference/
│   ├── generate.py              # 생성 3모드 (reconstruct / augment / zeroshot) + CLI
│   ├── fold_benchmark.py        # 증강 → Protenix(AF3) 폴딩 → OST GT 채점 end-to-end
│   └── quality_eval.py          # 생성 서열 품질/다양성 평가
└── utils/
    ├── alpha_geometry.py        # α-geometry (Fisher-Rao) exp/log/geodesic 연산
    └── spherical.py             # spherical exp/log maps, geodesic interpolation

configs/multimodel.yaml          # (리포 루트) 2-phase 멀티모델 추론 설정
scripts/                         # 런처: run_preprocess / run_train_* / run_msa_gen /
                                 #       run_fold_benchmark_* / run_multimodel.py 등
                                 # (preprocess.sh, train_*.sh = ada 클러스터용 SLURM 래퍼)

esm/                             # Meta ESM2 (submodule)
Protenix/                        # ByteDance Protenix / AF3 (submodule)
LFM/                             # Latent Flow Matching reference (submodule)
```

---

## Installation

```bash
# 서브모듈 포함 clone
git clone --recurse-submodules https://github.com/DeepFoldProtein/MSA_FLOW.git
cd MSA_FLOW

# uv로 가상환경 생성 및 의존성 설치
uv venv
uv sync

# pre-commit 훅 설정
pre-commit install

# 서브모듈 설치
cd Protenix && pip install -e . && cd ..
cd esm && pip install -e . && cd ..
```

> 이미 clone한 경우 서브모듈 초기화:
> ```bash
> git submodule update --init --recursive
> ```

---

## Training Pipeline

전처리부터 학습까지 3단계로 구성됩니다.

```
A3M 파일 (uniclust30)
    ↓  Step 1. preprocess.sh
ShardStore 디렉토리 (msaflow_openfold/)
    ↓  Step 2. train_decoder.sh
decoder_ema_final.pt
    ↓  Step 3. train_latent_fm.sh
latent_fm_ema_final.pt
```

> **저장 포맷:** 전처리 산출물은 LMDB가 아니라 **ShardStore**입니다 — append-only 바이너리
> 샤드들의 디렉토리(`msaflow.data.shard_store`)로, NFS/GPFS에서 안전하지 않던 기존 LMDB를
> 대체했습니다. `--output`/`OUTPUT_STORE`는 `.lmdb` 파일이 아니라 디렉토리 경로입니다.

### Step 1. 전처리 — ShardStore 빌드

OpenFold A3M 파일을 읽어 Protenix MSA embedding (L×128)과 ESM2 query embedding (L×1280)을
샤드 디렉토리로 저장합니다.

```bash
# 경로를 환경변수로 지정
A3M_DIR=/PATH/TO/openfold/uniclust30 \
OUTPUT_STORE=/PATH/TO/shardstore \
SHARD_MAX_GB=4 \
sbatch scripts/preprocess.sh
```

> Protenix 체크포인트 경로는 스크립트 내부에서 컨테이너 경로로 고정됩니다(`checkpoints/...pt`).
> 체크포인트 없이 돌리면 ESM2 embedding만 저장되는데, decoder 학습에는 Protenix embedding이
> 필요합니다. `RESUME=1`(기본)이면 이미 처리된 키를 건너뛰고 이어서 씁니다.

### Step 2. SFM Decoder 학습

```bash
STORE_PATH=/PATH/TO/shardstore \
VAL_STORE_PATH=/PATH/TO/val_shardstore \
OUTPUT_DIR=/PATH/TO/runs/decoder \
sbatch scripts/train_decoder.sh
```

| 설정 | 값 |
|------|-----|
| epochs | 7 |
| LR | 1e-5 |
| warmup steps | 5000 |
| weight decay | 0.1 |
| precision | BF16 |
| EMA decay | 0.9999 |

### Step 3. Latent FM 학습

```bash
STORE_PATH=/PATH/TO/shardstore \
OUTPUT_DIR=/PATH/TO/runs/latent_fm \
sbatch scripts/train_latent_fm.sh
```

| 설정 | 값 |
|------|-----|
| epochs | 15 |
| LR | 2.6e-4 |
| warmup steps | 3000 |
| weight decay | 0.1 |
| precision | BF16 |
| EMA decay | 0.9999 |

### SLURM 의존성 체인 (자동 순서 실행)

```bash
JID1=$(sbatch --parsable scripts/preprocess.sh)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 scripts/train_decoder.sh)
sbatch --dependency=afterok:$JID2 scripts/train_latent_fm.sh
```

---

## Quick Start (SLURM 없이 직접 실행)

### 1. Build the ShardStore dataset

```bash
python -m msaflow.data.preprocessing \
  --a3m_dir      /PATH/TO/openfold/uniclust30 \
  --output       /PATH/TO/shardstore \
  --protenix_checkpoint /models/protenix.ckpt \
  --max_msa_seqs 512 \
  --max_seq_len  1024 \
  --shard_max_gb 4 \
  --device       cuda
```

> `--output`은 ShardStore 디렉토리입니다(`.lmdb` 파일 아님). `--resume`로 이어쓰기,
> `--shard_max_gb`로 샤드 크기 상한(기본 4GB)을 조절합니다.

### 2. Train the SFM decoder

```bash
accelerate launch \
  --config_file msaflow/configs/accelerate_2gpu.yaml \
  msaflow/training/train_decoder.py \
  --config      msaflow/configs/decoder.yaml \
  --store_path  /PATH/TO/shardstore \
  --output_dir  /runs/decoder
```

### 3. Train the latent FM encoder

```bash
accelerate launch \
  --config_file msaflow/configs/accelerate_2gpu.yaml \
  msaflow/training/train_latent_fm.py \
  --config      msaflow/configs/latent_fm.yaml \
  --store_path  /PATH/TO/shardstore \
  --output_dir  /runs/latent_fm
```

### 4. Inference

**Zero-shot** (single sequence → synthetic MSA):
```bash
python msaflow/inference/generate.py \
  --mode          zeroshot \
  --query_seq     MKTAYIAKQRQISFVKSHFSRQ... \
  --decoder_ckpt  /runs/decoder/decoder_ema_final.pt \
  --latent_fm_ckpt /runs/latent_fm/latent_fm_ema_final.pt \
  --n_seqs        32 \
  --output        generated.fasta
```

**Reconstruction** (deep MSA → compressed → reconstructed):
```bash
python msaflow/inference/generate.py \
  --mode           reconstruct \
  --input_a3m      query.a3m \
  --decoder_ckpt   /runs/decoder/decoder_ema_final.pt \
  --protenix_ckpt  /models/protenix.pt \
  --n_seqs         32 \
  --output         reconstructed.a3m
```

**Few-shot augmentation** (shallow MSA → augmented):
```bash
python msaflow/inference/generate.py \
  --mode            augment \
  --input_a3m       shallow.a3m \
  --decoder_ckpt    /runs/decoder/decoder_ema_final.pt \
  --latent_fm_ckpt  /runs/latent_fm/latent_fm_ema_final.pt \
  --protenix_ckpt   /models/protenix.pt \
  --output          augmented.a3m
```

---

## Benchmarking

생성 결과물을 구조 예측 정확도로 평가하는 길은 두 갈래입니다. **둘 다 YAML config가 진입점**이며,
서로 섞이지 않도록 출력이 모드/모델/regime별로 자동 분리됩니다.

### (A) 단일 백엔드 end-to-end — `inference/fold_benchmark.py`

`msaflow/configs/fold_benchmark.yaml` 하나로 **증강 → Protenix(또는 AF3) 폴딩 → OST GT 채점**까지
한 번에 돕니다. `fold_benchmark_af3.yaml`은 같은 흐름을 colabfold af3-preview 백엔드로 돌리는 변형입니다.

```bash
# GPU 컴퓨트 노드에서
bash scripts/run_fold_benchmark_gpu.sh msaflow/configs/fold_benchmark.yaml
# 또는 4-GPU 샤딩
GPUS="0 1 2 3" bash scripts/run_fold_benchmark_4gpu.sh
```

config의 `run.mode`로 네 가지 평가 모드를 고릅니다:

| mode | 의미 |
|------|------|
| `zeroshot` | seed별 MSA를 각각 폴딩 → best pLDDT 선택 |
| `fewshot`  | Syn+Rec로 얕은 MSA를 증강한 뒤 한 번 폴딩 |
| `shallow`  | 원본 얕은 MSA 그대로 폴딩 (증강 전 baseline) |
| `nomsa`    | MSA 없이 서열만으로 폴딩 (baseline) |

출력은 `{output_dir}/{mode}/`로 자동 분리됩니다.
⚠️ OST 채점이 헤더에서 chain을 뽑으므로 타겟명은 반드시 `pdb_id__chain` 형식이어야 합니다.

### (B) 2-phase 멀티모델 파이프라인 — `scripts/run_multimodel.py`

여러 구조 예측기(AF2/AF3/Protenix/Boltz2/ESMFold…)를 **MSA-regime별로** 교차 비교합니다.
MSA 생성(phase1)과 추론(phase2)을 분리하는 것이 핵심입니다.

```
phase1  MSA 생성          regime별 입력 디렉토리 생성 (no_msa / zero_shot / few_shot / full)
          ▼
phase2  model × regime    uv run python scripts/run_multimodel.py run
          ▼                  → output_root/<model>/<regime>/
collect 결과 정리          uv run python scripts/run_multimodel.py collect
          ▼                  → summary.csv (plddt/ptm/iptm/…)
OST     GT 채점            scripts/ost_log_to_csv.py 등으로 lddt/tm/gdt/rmsd 복구
```

**phase1 (MSA 생성)** — zero-shot a3m 대량 생성 런처. `fold_benchmark.yaml`에서
체크포인트·`shallow_msa_dir`·`targets.csv`를 읽습니다:

```bash
GPUS="0 1 2 3" bash scripts/run_msa_gen.sh
#   기본 N_SEEDS=32 × N_SEQS=32 = 1024 seqs/target → {OUTPUT_DIR}/{target}.a3m
```

**phase2 (멀티모델 추론)** — 리포 루트 `configs/multimodel.yaml`:

```bash
uv run python scripts/run_multimodel.py run     -c configs/multimodel.yaml [--dry-run]
uv run python scripts/run_multimodel.py collect -c configs/multimodel.yaml [-o summary.csv]
```

`configs/multimodel.yaml` 핵심 필드:

| 필드 | 뜻 |
|------|-----|
| `runner` | 추론 실행기 (`uv run colabfold_batch`) |
| `models` | `--model-type` 목록 (alphafold2_ptm, alphafold3, protenix, boltz2, esmfold2 …) |
| `regimes` | regime명 → phase1에서 만든 MSA 입력 경로. 경로 없으면 자동 SKIP |
| `output_root` | 결과 루트 (`<model>/<regime>/`로 정리) |
| `common_flags` | 모든 실행 공통 colabfold 플래그 (예: `--num-seeds 1`) |
| `model_flags` | 모델별 추가 플래그 (예: `alphafold3: --accept-alphafold3-terms`) |

> `regime`은 colabfold 플래그가 아니라 **"어떤 MSA 입력을 넣느냐 + 폴더 라벨"**입니다.
> `no_msa`(서열만) / `zero_shot`(MSAFlow 생성) / `few_shot`(얕은+증강) / `full`(완전 MSA)로
> 같은 모델을 교차 비교해, MSAFlow가 MSA-regime별로 얼마나 기여하는지 분리해서 봅니다.

---

## Configuration Reference (`msaflow/configs/`)

모든 파이프라인은 YAML이 single source of truth이고, CLI 플래그로 개별 값을 override 할 수 있습니다.

- **`fold_benchmark.yaml`** — end-to-end 벤치 + phase1 MSA 생성의 공통 설정
  - `run`: mode / output_dir / device / resume·regenerate·refold / 샤딩
  - `targets`: `list` 또는 `csv`(pdb_id,chain_id) / `fasta` / `shallow_msa_dir`
  - `model`: `protenix_model` / `protenix_ckpt`(Rec 트랙) / `decoder_ckpt` / `latent_fm_ckpt`
  - `augment`: `n_seqs` / `n_seeds` / `n_steps`(ODE) / `temperature`
  - `fold`: `n_fold_seeds` / `fold_seed_base` / `n_sample` / 참조 PDB TM-score
  - `ost`: `enable` / `gt_dir` / `chain_csv` / `out_csv`
- **`fold_benchmark_af3.yaml`** — 위를 colabfold af3-preview 백엔드로 돌리는 변형
- **`decoder.yaml` / `latent_fm.yaml`** — 학습 설정 (§ Training Pipeline 참고)
- **`configs/multimodel.yaml`** (리포 루트) — 2-phase phase2 설정 (위 표 참고)

---

## Key Results (from paper)

### MSA Autoencoding (CAMEO benchmark, 50 proteins)

| Method | pLDDT | TM-score | Storage |
|--------|-------|----------|---------|
| No MSA | ~47   | 0.33     | —       |
| Ground Truth (7k seqs) | **91.6** | **0.89** | 100% |
| **MSAFlow Recon (32 seqs)** | **89.0** | **0.86** | **6.5%** |
| MSAFlow Zero-shot | 75–80 | 0.62–0.70 | — |

### MSA Augmentation (zero-shot, 200 proteins from CAMEO/CASP/PDB)

| Method | pLDDT | TM-score |
|--------|-------|----------|
| No/Shallow MSA | 73.1 | 0.55 |
| EvoDiff (650M) | 67.7 | 0.49 |
| MSAGPT (3B) | 71.6 | 0.53 |
| **MSAFlow (130M)** | **75.2** | **0.62** |

### Enzyme Design (EC classes, <20 training sequences)

| Method | Acc×Uniqueness (Fixed) | Acc×Uniqueness (Variable) |
|--------|------------------------|---------------------------|
| ProfileBFN | 42–100% | N/A |
| MSAGPT | — | 15–38% |
| **MSAFlow** | **83–100%** | **51–92%** |

---

## Mathematical Background

### Statistical Flow Matching (SFM)

MSAFlow operates on the **Fisher-Rao statistical manifold** via the isometric mapping:

```
π : µ → x = √µ    (probability simplex → unit sphere)
π⁻¹: x → µ = x²  (unit sphere → probability simplex)
```

**Training** interpolates along spherical geodesics:
```
x_t = exp_{x₀}(t · log_{x₀}(x₁))     t ∈ [0,1]
```

**Target velocity** (points from xₜ toward x₁):
```
u_t(xₜ | x₁) = log_{xₜ}(x₁) / (1 − t)
```

**Loss**:
```
L_SFM(θ) = E_{t, sᵢ, µ₀}[ ‖v_θ(xₜ | m, t) − uₜ‖² ]
```

### Latent Rectified Flow

Straight-line interpolation in R^(L×128):
```
z_t = (1−t)·z₁ + t·z₀,   z₀ ~ N(0, I)
```

```
L_RFM(θ) = E_{t, z₀, z₁}[ ‖v_θ(zₜ, e, t) − (z₀ − z₁)‖² ]
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.1 | Core deep learning |
| Accelerate | ≥0.26 | Distributed training |
| OmegaConf | ≥2.3 | Configuration |
| fair-esm | — | ESM2-650M embeddings |
| Protenix | — | AF3 MSAModule encoder |

---

## Citation

```bibtex
@article{msaflow2026,
  title   = {MSAFlow: A Unified Approach for MSA Representation, Augmentation, and Family-based Protein Design},
  author  = {Anonymous},
  journal = {ICLR 2026 (under review)},
  year    = {2026}
}
```

---

## Acknowledgements

- [Protenix](https://github.com/bytedance/protenix) — AF3 MSAModule implementation
- [ESM](https://github.com/facebookresearch/esm) — ESM2 protein language model
- [LFM](https://github.com/FieteLab/LFM) — Latent Flow Matching reference implementation
- [Statistical Flow Matching (SFM)](https://arxiv.org/abs/2405.16441) — Cheng et al., 2024
