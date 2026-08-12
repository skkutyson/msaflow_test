# MSAFlow 개발 환경 가이드

전처리부터 훈련까지 전 과정을 처음부터 끝까지 설명한다.

---

## 재접속 후 빠른 시작 (이미 세팅 완료된 경우)

> 서버에 다시 로그인했을 때. 처음 세팅은 섹션 2~6 참고.

```bash
source ~/.bashrc                      # uv 등 PATH 적용
cd /mnt/MSA_FLOW

sbatch scripts/preprocess.sh         # 전처리 (LMDB 아직 없을 때)
sbatch scripts/train_decoder.sh      # Decoder 훈련
sbatch scripts/train_latent_fm.sh    # Latent FM 훈련

squeue -u $USER                       # 잡 상태 확인
tail -f logs/preprocess_<JOBID>.out  # 로그 실시간 확인
```

**재접속마다 `uv sync`나 sandbox 재빌드는 필요 없다.**
`.venv`와 sandbox는 디스크에 그대로 남아 있다.
`source ~/.bashrc`는 `uv` 명령어가 PATH에 없을 때만 필요하다
(이미 `uv`가 보이면 생략 가능).

---

## 목차

1. [환경 개요](#1-환경-개요)
2. [레포지토리 클론](#2-레포지토리-클론)
3. [의존성 설치 (uv)](#3-의존성-설치-uv)
4. [Apptainer Sandbox 빌드](#4-apptainer-sandbox-빌드)
5. [Sandbox 초기화 (setup_sandbox.sh)](#5-sandbox-초기화-setup_sandboxsh)
6. [스모크 테스트](#6-스모크-테스트)
7. [전처리: LMDB 생성](#7-전처리-lmdb-생성)
8. [훈련: Decoder](#8-훈련-decoder)
9. [훈련: Latent FM](#9-훈련-latent-fm)
10. [경로 구조 한눈에 보기](#10-경로-구조-한눈에-보기)
11. [자주 쓰는 명령어 모음](#11-자주-쓰는-명령어-모음)
12. [트러블슈팅](#12-트러블슈팅)

---

## 1. 환경 개요

```
클러스터 실행 스택:

  SLURM                ← 잡 스케줄러 (sbatch)
    └── Apptainer      ← 컨테이너 런타임 (CUDA 환경 격리)
          └── uv run   ← Python 가상환경 실행 (의존성 격리)
                └── accelerate launch  ← 멀티 GPU 분산 학습
```

| 도구 | 역할 |
|------|------|
| **uv** | Python 의존성 관리 및 `.venv` 생성 |
| **Apptainer sandbox** | CUDA 드라이버 및 시스템 라이브러리 격리 |
| **SLURM** | GPU 노드 할당 및 잡 큐 관리 |
| **Accelerate** | 멀티 GPU 분산 학습 래퍼 |

**주요 경로**

| 항목 | 경로 |
|------|------|
| 레포 | `/home/yjlee4/MSA_FLOW` |
| Apptainer sandbox | `/home/yjlee4/msaflow_sandbox` |
| uniclust30 데이터 | `/gpfs/database/openfold/uniclust30` |
| 출력 (LMDB, 체크포인트) | `/gpfs/deepfold/users/yjlee4/` |

---

## 2. 레포지토리 클론

```bash
git clone --recurse-submodules https://github.com/kimsukimsu/MSA_GENERATION /mnt/MSA_FLOW
cd /mnt/MSA_FLOW
```

이미 클론되어 있는데 서브모듈이 비어 있다면:

```bash
git submodule update --init --recursive
```

서브모듈은 두 개다:

| 서브모듈 | 경로 | 용도 |
|----------|------|------|
| `esm` | `esm/` | Meta fair-esm (단백질 언어 모델) |
| `Protenix` | `Protenix/` | ByteDance AlphaFold-3 구현체 |

---

## 3. 의존성 설치 (uv)

### uv 설치 (없을 경우)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 가상환경 생성 및 의존성 설치

```bash
cd /mnt/MSA_FLOW

# Python 3.12로 venv 생성 (torch가 3.14 미지원)
uv venv --python 3.12

# 의존성 설치 (pyproject.toml 기준)
uv sync
```

> **왜 Python 3.12인가?**
> torch 2.7.x는 `cp311~cp313` 휠만 제공한다. Python 3.14 이상에서는 휠이 없어 설치 실패한다.
> `pyproject.toml`에 `requires-python = ">=3.11,<3.14"` 로 상한이 고정되어 있다.

### 의존성 구조 (왜 복잡한가)

```
pyproject.toml
  ├── protenix  ─── [tool.uv.sources] → Protenix/ (로컬 서브모듈)
  │                  └── fair-esm==2.0.0  ←─ Protenix가 요구
  └── fair-esm  ─── [tool.uv.sources] → esm/ (로컬 서브모듈)
                     [tool.uv] override-dependencies  ←─ 전역 강제 적용
```

`[tool.uv.sources]`는 루트 프로젝트의 직접 의존성에만 적용된다.
Protenix처럼 path dependency의 하위 의존성까지 덮으려면
`override-dependencies`로 전역 강제가 필요하다.

### pre-commit 훅 설치 (선택)

```bash
uv run pre-commit install
```

---

## 4. Apptainer Sandbox 빌드

Apptainer sandbox는 **CUDA 런타임이 포함된 컨테이너 디렉터리**다.
`.venv`와는 별개로, GPU 드라이버 호환성을 보장하기 위해 사용한다.

### 4-1. 기본 이미지로 sandbox 생성 (최초 1회)

```bash
# CUDA 12.4 + Ubuntu 22.04 기반 이미지 사용 (클러스터 드라이버 버전에 맞게 조정)
apptainer build --sandbox \
    /home/yjlee4/msaflow_sandbox \
    docker://nvcr.io/nvidia/pytorch:24.03-py3
```

> - `--sandbox`: 수정 가능한 디렉터리 형태로 빌드 (`.sif` 단일 파일이 아님)
> - `nvcr.io/nvidia/pytorch:24.03-py3`: PyTorch + CUDA가 포함된 NVIDIA 공식 이미지
> - 클러스터 CUDA 드라이버 버전은 `nvidia-smi` 로 확인 후 맞는 태그 선택

### 4-2. 이미지 태그 선택 기준

```bash
nvidia-smi  # Driver Version 확인
# CUDA 12.4 이상  → pytorch:24.03-py3
# CUDA 12.1 이상  → pytorch:23.11-py3
# CUDA 11.8 이상  → pytorch:23.03-py3
```

### 4-3. 기존 sandbox 수정

sandbox 내부를 변경해야 할 경우 (`--writable --fakeroot` 필요):

```bash
# 패키지 추가 설치
apptainer exec --writable --fakeroot \
    /home/yjlee4/msaflow_sandbox \
    pip install some-package

# 셸로 직접 진입해서 확인
apptainer shell --writable --fakeroot \
    /home/yjlee4/msaflow_sandbox
```

> `--fakeroot`: root 권한 없이 컨테이너 내부를 root처럼 조작 가능

---

## 5. Sandbox 초기화 (setup_sandbox.sh)

sandbox를 새로 만들거나 마운트 포인트가 없을 때 **최초 1회** 실행한다.

```bash
bash scripts/setup_sandbox.sh
```

이 스크립트가 하는 일:

1. **uv 설치** — sandbox 내부에 `pip install uv`
2. **마운트 포인트 생성** — 컨테이너 내부에 `/mnt/MSA_FLOW`, `/mnt/lmdb`, `/mnt/output` 디렉터리 생성
3. **uv sync** — 레포를 bind-mount한 상태에서 `.venv` 구성

> **왜 마운트 포인트를 미리 만들어야 하는가?**
> `--bind /host/path:/mnt/output` 옵션은 컨테이너 내부에 `/mnt/output`이
> 이미 존재해야 동작한다. 없으면 `FATAL: destination doesn't exist` 오류 발생.

### 환경 변수로 경로 재정의

기본값을 바꾸려면 환경 변수를 앞에 붙인다:

```bash
REPO_DIR=/other/path SANDBOX=/other/sandbox bash scripts/setup_sandbox.sh
```

---

## 6. 스모크 테스트

설치가 제대로 됐는지 확인한다. GPU와 외부 파일 불필요.

### 호스트에서 실행

```bash
cd /mnt/MSA_FLOW
uv run python tests/test_smoke.py
```

### Sandbox 안에서 실행

```bash
apptainer exec --nv \
    --bind /mnt/MSA_FLOW:/mnt/MSA_FLOW \
    /home/yjlee4/msaflow_sandbox \
    bash -c "cd /mnt/MSA_FLOW && uv run python tests/test_smoke.py"
```

정상 출력:

```
=== MSAFlow smoke tests ===
  PASS  msaflow imports
  PASS  parse_a3m  (2 seqs, L=5)
  PASS  tokenise_msa  shape=(2, 5)
  PASS  compute_sequence_weights  weights=[...]
  PASS  esm import
  PASS  protenix import

All 6 tests passed.
```

---

## 7. 전처리: LMDB 생성

uniclust30 `.a3m` 파일들을 읽어 Protenix로 MSA 임베딩을 추출하고
LMDB 데이터베이스로 저장한다.

### SLURM 잡으로 실행 (권장)

```bash
sbatch scripts/preprocess.sh
```

### 주요 파라미터 재정의

```bash
A3M_DIR=/gpfs/database/openfold/uniclust30 \
OUTPUT_LMDB=/gpfs/deepfold/users/yjlee4/msaflow.lmdb \
MAX_MSA_SEQS=512 \
MAX_SEQ_LEN=1024 \
sbatch scripts/preprocess.sh
```

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `A3M_DIR` | `/gpfs/database/openfold/uniclust30` | uniclust30 a3m 파일 디렉터리 |
| `OUTPUT_LMDB` | `/gpfs/deepfold/users/yjlee4/msaflow.lmdb` | 출력 LMDB 경로 |
| `MAX_MSA_SEQS` | `512` | MSA당 최대 시퀀스 수 |
| `MAX_SEQ_LEN` | `1024` | 최대 시퀀스 길이 |
| `MAP_SIZE_GB` | `500` | LMDB 최대 크기 (GB) |

### Apptainer bind-mount 구성

```
호스트                          컨테이너 내부
/mnt/MSA_FLOW              →   /mnt/MSA_FLOW   (코드 + 체크포인트)
/gpfs/database/.../uniclust30 → /mnt/data      (a3m 입력)
/gpfs/deepfold/users/yjlee4   → /mnt/output    (LMDB 출력)
```

### 로그 확인

```bash
tail -f /mnt/MSA_FLOW/logs/preprocess_<JOBID>.out
```

---

## 8. 훈련: Decoder

SFM Decoder (MSA 임베딩 → 아미노산 시퀀스 복원)를 학습한다.
129M 파라미터, 7 epoch, bf16.

### SLURM 잡으로 실행

```bash
sbatch scripts/train_decoder.sh
```

### 파라미터 재정의

```bash
LMDB_PATH=/gpfs/deepfold/users/yjlee4/msaflow.lmdb \
OUTPUT_DIR=/gpfs/deepfold/users/yjlee4/runs/decoder \
sbatch scripts/train_decoder.sh
```

### 설정 파일 수정

`msaflow/configs/decoder.yaml`:

```yaml
training:
  epochs: 7
  lr: 1.0e-5
  mixed_precision: "bf16"
  use_wandb: false        # wandb 쓰려면 true + wandb login
  output_dir: /runs/decoder  # 컨테이너 내부 경로 (bind-mount된 /mnt/output이 아님!)
```

> `output_dir`는 컨테이너 내부 경로다. 실제 체크포인트는
> `--bind "$OUTPUT_DIR":/mnt/output` 으로 호스트 `OUTPUT_DIR`에 저장된다.

### GPU 수 변경

`train_decoder.sh`의 SBATCH 헤더와 accelerate config를 함께 수정한다:

```bash
# train_decoder.sh
#SBATCH --gres=gpu:4          # GPU 수 변경

ACCEL_CONFIG=/mnt/MSA_FLOW/msaflow/configs/accelerate_4gpu.yaml  # config도 맞게
```

---

## 9. 훈련: Latent FM

Latent Flow Matching (MSA 잠재 공간에서 생성 모델 학습).
130M 파라미터, 15 epoch, bf16.

### SLURM 잡으로 실행

```bash
sbatch scripts/train_latent_fm.sh
```

### 파라미터 재정의

```bash
LMDB_PATH=/gpfs/deepfold/users/yjlee4/msaflow.lmdb \
OUTPUT_DIR=/gpfs/deepfold/users/yjlee4/runs/latent_fm \
sbatch scripts/train_latent_fm.sh
```

### 설정 파일

`msaflow/configs/latent_fm.yaml`:

```yaml
training:
  epochs: 15
  lr: 2.6e-4
  mixed_precision: "bf16"
  output_dir: /runs/latent_fm
```

---

## 10. 경로 구조 한눈에 보기

```
호스트 파일시스템
├── /home/yjlee4/MSA_FLOW/                ← 이 레포 (git)
│   ├── pyproject.toml                    ← uv 의존성 정의
│   ├── .venv/                            ← uv가 생성한 Python 3.12 venv
│   ├── esm/                              ← git submodule (fair-esm)
│   ├── Protenix/                         ← git submodule (Protenix)
│   ├── msaflow/
│   │   ├── configs/                      ← decoder.yaml, latent_fm.yaml, accelerate_*.yaml
│   │   ├── data/preprocessing.py         ← 전처리 (a3m → LMDB)
│   │   ├── training/train_decoder.py     ← Decoder 학습 진입점
│   │   └── training/train_latent_fm.py   ← Latent FM 학습 진입점
│   ├── train.sh                         ← 간단 실행용 (SLURM 없이 apptainer 직접 호출)
│   ├── scripts/
│   │   ├── setup_sandbox.sh             ← sandbox 초기화 (최초 1회)
│   │   ├── preprocess.sh                ← 전처리 SLURM 잡
│   │   ├── train_decoder.sh             ← Decoder 훈련 SLURM 잡
│   │   └── train_latent_fm.sh           ← Latent FM 훈련 SLURM 잡
│   └── logs/                            ← SLURM stdout/stderr
│
├── /home/yjlee4/
│   └── msaflow_sandbox/                 ← Apptainer sandbox (CUDA 환경)
│
├── /gpfs/deepfold/users/yjlee4/
│   ├── msaflow.lmdb                     ← 전처리 결과 LMDB
│   └── runs/
│       ├── decoder/                     ← Decoder 체크포인트
│       └── latent_fm/                   ← Latent FM 체크포인트
│
└── /gpfs/database/openfold/uniclust30/  ← uniclust30 a3m 파일 (읽기 전용)
```

### 컨테이너 내부 경로 매핑

| 호스트 | 컨테이너 내부 | 용도 |
|--------|--------------|------|
| `/home/yjlee4/MSA_FLOW` | `/mnt/MSA_FLOW` | 코드 + venv |
| `/gpfs/database/openfold/uniclust30` | `/mnt/data` | a3m 입력 데이터 |
| `/gpfs/deepfold/users/yjlee4` | `/mnt/output` | 결과 출력 |

---

## 11. 자주 쓰는 명령어 모음

```bash
# 잡 제출
sbatch scripts/preprocess.sh
sbatch scripts/train_decoder.sh
sbatch scripts/train_latent_fm.sh

# 잡 상태 확인
squeue -u $USER

# 로그 실시간 확인
tail -f /mnt/MSA_FLOW/logs/decoder_<JOBID>.out

# 잡 취소
scancel <JOBID>

# sandbox 셸 진입 (디버깅)
apptainer shell --nv \
    --bind /mnt/MSA_FLOW:/mnt/MSA_FLOW \
    /home/yjlee4/msaflow_sandbox

# sandbox 안에서 직접 실행 (SLURM 없이)
apptainer exec --nv \
    --bind /mnt/MSA_FLOW:/mnt/MSA_FLOW \
    --bind /gpfs/deepfold/users/yjlee4:/mnt/output \
    /home/yjlee4/msaflow_sandbox \
    bash -c "cd /mnt/MSA_FLOW && uv run python tests/test_smoke.py"

# 의존성 재설치 (pyproject.toml 변경 후)
uv sync

# 새 패키지 추가
uv add some-package
uv sync
```

---

## 12. 트러블슈팅

### `FATAL: destination /mnt/output doesn't exist in container`

sandbox 내부에 마운트 포인트가 없다는 뜻이다.

```bash
apptainer exec --writable --fakeroot \
    /home/yjlee4/msaflow_sandbox \
    mkdir -p /mnt/MSA_FLOW /mnt/lmdb /mnt/output
```

또는 `bash scripts/setup_sandbox.sh` 재실행.

---

### `No solution found: fair-esm==2.0.0`

Protenix의 `requirements.txt`에 박힌 `fair-esm` 경로가 이전 환경을 가리키거나,
`protenix.egg-info`가 구버전 경로를 캐시하고 있을 때 발생한다.

```bash
# 1. Protenix/requirements.txt의 fair-esm 경로가 /mnt/MSA_FLOW/esm인지 확인
grep fair-esm Protenix/requirements.txt

# 2. 캐시 삭제
rm -rf Protenix/protenix.egg-info

# 3. 재시도
uv sync
```

---

### `torch` 설치 실패 (Python 버전 불일치)

torch 2.7.x는 Python 3.14 이상을 지원하지 않는다.

```bash
# venv 재생성
rm -rf .venv
uv venv --python 3.12
uv sync
```

---

### GPU가 보이지 않는 (`--nv` 관련)

```bash
# 호스트에서 GPU 확인
nvidia-smi

# sandbox 안에서 GPU 확인
apptainer exec --nv /home/yjlee4/msaflow_sandbox nvidia-smi
```

`--nv` 플래그가 없으면 컨테이너에서 GPU가 보이지 않는다.
SLURM 잡에서는 `#SBATCH --gres=gpu:N` 으로 먼저 GPU를 할당받아야 `--nv`가 동작한다.

---

### 서브모듈이 비어 있음

```bash
git submodule update --init --recursive
```

---

### uv가 sandbox 안에 없음

```bash
apptainer exec --writable --fakeroot \
    /home/yjlee4/msaflow_sandbox \
    pip install uv -q
```
