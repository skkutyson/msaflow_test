# MSAFlow

**Flow Matching for MSA Generation** — 쿼리 서열(또는 얕은 MSA)로부터 합성 MSA를
생성해, 구조 예측기(Protenix/AF3 등)의 입력으로 쓰는 생성 파이프라인.

핵심 아이디어: deep MSA의 임베딩 분포를 latent flow-matching으로 학습해 두고,
추론 시 ESM2 쿼리 표현에서 MSA 임베딩을 **샘플링** → SFM 디코더로 아미노산 서열로
**디코딩**한다. 이렇게 만든 합성 MSA를 폴딩에 넣어 orphan/shallow-MSA 단백질의
구조 정확도를 끌어올리는 것이 목표.

---

## 1. 패키지 구조

```
msaflow/
├── configs/            # 모든 실험 설정 (YAML) — 파이프라인의 진입점
│   ├── fold_benchmark.yaml      # end-to-end 벤치(증강→폴딩→OST) · phase1 MSA 생성도 여기 설정을 읽음
│   ├── fold_benchmark_af3.yaml  # AF3(colabfold af3-preview) 백엔드용 변형
│   ├── colabfold_batch.yaml     # colabfold_batch 공통 플래그
│   ├── decoder*.yaml            # SFM 디코더 학습 설정
│   ├── latent_fm*.yaml          # latent flow-matching 학습 설정
│   └── accelerate_*gpu.yaml     # 멀티-GPU accelerate 런치 설정
├── models/             # SFM 디코더, latent-FM 인코더 등 모델 정의
├── data/               # a3m 파싱·필터, 임베딩 추출(Protenix/ESM2), 데이터셋/샤드
├── inference/
│   ├── generate.py         # 생성 3모드 (reconstruct / augment_shallow / generate_zeroshot)
│   ├── fold_benchmark.py   # 증강→Protenix 폴딩→OST 채점 end-to-end
│   └── quality_eval.py     # 생성 서열 품질/다양성 평가
├── training/           # train_decoder.py, train_latent_fm.py
└── utils/              # α-geometry 등 보조 연산
```

> 관련 런처 스크립트는 리포 루트 `scripts/`, 멀티모델 설정은 리포 루트 `configs/`에 있다.

---

## 2. 생성 모드 (generation modes)

`inference/generate.py`가 제공하는 세 가지 경로 (논문 Section 3.3):

| 모드 | 함수 | 입력 | 설명 |
|------|------|------|------|
| **reconstruct** | `reconstruct(msa_seqs)` | deep MSA | MSA를 Protenix로 인코딩 후 SFM 디코더로 32개 서열 재구성 (압축·복원) |
| **few-shot** | `augment_shallow(shallow_seqs)` | 얕은 MSA | 얕은 MSA 인코딩(Rec 트랙) + ESM2 기반 합성 임베딩(Syn 트랙)을 **결합** (Syn+Rec, Section 6.5) |
| **zero-shot** | `generate_zeroshot(query_seq)` | 쿼리 서열만 | ESM2 쿼리 표현만으로 MSA 임베딩을 생성 후 디코딩 (MSA 전혀 불필요) |

모든 모드는 gap 없는 아미노산 서열 리스트를 반환 → FASTA/A3M로 저장한다.
`generate_zeroshot_seeds(...)`는 seed별 블록을 yield 해 다양성 분해(within/between-seed)에
쓸 수 있다.

---

## 3. 두 가지 벤치마크 경로

MSAFlow 결과물을 평가하는 길은 두 갈래다. **둘 다 config(YAML)가 진입점**이다.

### (A) 단일 백엔드 end-to-end — `fold_benchmark.py`

`msaflow/configs/fold_benchmark.yaml` 하나로 증강→Protenix(AF3) 폴딩→OST GT 채점까지.

```bash
# GPU 컴퓨트 노드에서
bash scripts/run_fold_benchmark_gpu.sh msaflow/configs/fold_benchmark.yaml
# 또는 4-GPU 샤딩
GPUS="0 1 2 3" bash scripts/run_fold_benchmark_4gpu.sh
```

config의 `run.mode`로 네 가지 모드를 고른다:

| mode | 의미 |
|------|------|
| `zeroshot` | seed별 MSA를 각각 폴딩 → best pLDDT |
| `fewshot`  | Syn+Rec로 얕은 MSA 증강 후 한 번 폴딩 |
| `shallow`  | 원본 얕은 MSA 그대로 폴딩 (증강 전 baseline) |
| `nomsa`    | MSA 없이 서열만으로 폴딩 (baseline) |

출력은 `{output_dir}/{mode}/` 로 자동 분리되어 모드끼리 섞이지 않는다.

### (B) 2-phase 멀티모델 파이프라인 — `run_multimodel.py`

여러 구조 예측기(AF2/AF3/Protenix/Boltz2/ESMFold…)를 **MSA-regime별로** 교차 비교.
MSA 생성(phase1)과 추론(phase2)을 분리한다.

```
phase1  MSA 생성          regime별 입력 디렉토리 생성
          │                (no_msa / zero_shot / few_shot / full)
          ▼
phase2  model × regime    uv run python scripts/run_multimodel.py run
          │                  → output_root/<model>/<regime>/
          ▼
collect 결과 정리          uv run python scripts/run_multimodel.py collect
          │                  → summary.csv (plddt/ptm/iptm/…)
          ▼
OST     GT 채점            scripts/ost_log_to_csv.py 등으로 lddt/tm/gdt/rmsd 복구
```

**phase1 (MSA 생성)** — zero-shot a3m 대량 생성 런처:

```bash
# msaflow/configs/fold_benchmark.yaml 에서 체크포인트·shallow_msa_dir·targets.csv 를 읽음
GPUS="0 1 2 3" bash scripts/run_msa_gen.sh
#   기본 N_SEEDS=32 × N_SEQS=32 = 1024 seqs/target → {OUTPUT_DIR}/{target}.a3m
```

**phase2 (멀티모델 추론)** — `configs/multimodel.yaml`(리포 루트):

```bash
uv run python scripts/run_multimodel.py run     -c configs/multimodel.yaml [--dry-run]
uv run python scripts/run_multimodel.py collect -c configs/multimodel.yaml [-o summary.csv]
```

`configs/multimodel.yaml` 핵심 필드:

| 필드 | 뜻 |
|------|-----|
| `runner` | 추론 실행기 (`uv run colabfold_batch`) |
| `models` | `--model-type` 목록 (예: alphafold2_ptm, alphafold3, protenix, boltz2, esmfold2) |
| `regimes` | regime명 → phase1에서 만든 MSA 입력 경로. 경로 없으면 자동 SKIP |
| `output_root` | 결과 루트 (`<model>/<regime>/`로 정리) |
| `common_flags` | 모든 실행 공통 colabfold 플래그 (예: `--num-seeds 1`) |
| `model_flags` | 모델별 추가 플래그 (예: `alphafold3: --accept-alphafold3-terms`) |

> `regime`은 colabfold 플래그가 아니라 **"어떤 MSA 입력을 넣느냐 + 폴더 라벨"**이다.
> `no_msa`(서열만) / `zero_shot`(MSAFlow 생성) / `few_shot`(얕은+증강) / `full`(완전 MSA)로
> 같은 모델을 교차 비교한다.

---

## 4. 설정 파일 레퍼런스 (`configs/`)

모든 파이프라인은 YAML이 single source of truth. CLI 플래그로 개별 값 override 가능.

- **`fold_benchmark.yaml`** — end-to-end 벤치 + phase1 MSA 생성의 공통 설정
  - `run`: mode / output_dir / device / resume·regenerate·refold / 샤딩
  - `targets`: `list` 또는 `csv`(pdb_id,chain_id) / `fasta` / `shallow_msa_dir`
    - ⚠️ OST 채점이 헤더에서 chain을 뽑으므로 타겟명은 반드시 `pdb_id__chain` 형식
  - `model`: `protenix_model` / `protenix_ckpt`(Rec 트랙) / `decoder_ckpt` / `latent_fm_ckpt`
  - `augment`: `n_seqs` / `n_seeds` / `n_steps`(ODE) / `temperature`
  - `fold`: `n_fold_seeds` / `fold_seed_base` / `n_sample` / 참조 PDB TM-score
  - `ost`: `enable` / `gt_dir` / `chain_csv` / `out_csv`
- **`fold_benchmark_af3.yaml`** — 위를 colabfold af3-preview 백엔드로 돌리는 변형
- **`multimodel.yaml`** (리포 루트) — 2-phase phase2 설정 (§3-B 표 참고)

---

## 5. 학습 (training)

```bash
# SFM 디코더
bash scripts/run_train_decoder.sh      # configs/decoder.yaml
# latent flow-matching 인코더
bash scripts/run_train_latent_fm.sh    # configs/latent_fm.yaml
```

멀티-GPU는 `configs/accelerate_{1,2,4}gpu.yaml`로 accelerate 런치.

---

## 6. 실행 환경 메모

- **GPU 작업은 반드시 컴퓨트 노드에서** (로그인 노드 금지). 스크립트들은 `nvidia-smi`
  존재를 확인하고, 없으면 `srun --gres=gpu:...`를 안내하며 종료한다.
- `run_multimodel.py`는 `uv run`으로 호출 → venv 수동 활성화 불필요 (리포 루트에서 실행).
- 용량 이슈로 MSA·폴드·CSV 산출물은 `/store/yjlee4/...`에 저장 (GPU 노드에서만 마운트).
- AF3 가중치는 첫 사용 시 자동 다운로드되지만 **오프라인 노드에서는 실패** →
  인터넷 되는 노드에서 미리 받아 `--model-dir`로 지정 (`multimodel.yaml` 주석 참고).
  `alphafold3`만 `--accept-alphafold3-terms` 동의 필요(비상업용).
