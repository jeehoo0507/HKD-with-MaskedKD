# HKD + teacher masking: 작은 비교 실험

**DeiT-B → 보조 CNN → 학생 CNN** 파이프라인에서 교사 토큰 제거의 영향을 비교합니다.
기준은 [Passalis et al., CVPR 2020](https://arxiv.org/abs/2005.00727)과
[공식 HKD 구현](https://github.com/passalis/pkth/tree/5766e6398c574598c0b166bfadfaf573a7190a61)입니다.
데이터와 결과 기록 방식은 [KSHS AIM Lab benchmark](https://github.com/jeehoo0507/kshs-aimlab-benchmarks/tree/e81cc3063dc5f98a2d43366f2e86b08905e80875)를 따릅니다.

목표는 **3~5시간의 최대 실행 예산, GPU VRAM 2~4 GiB**에서 작은 paired pilot을 실행하는 것입니다.
현재 구현은 CPU에서 검증했으며, 목표 GPU의 실제 메모리와 실행 시간 또는 정확도는 측정하지 않았습니다.
실제 GPU가 너무 느리면 사전 측정 후 실행을 중단합니다. 시간 안에 성공하는 것을 무조건 보장하지 않습니다.
데이터·가중치 다운로드와 설치 시간은 실행 예산에서 제외합니다. 빠른 GPU에서는 3시간 전에 끝날 수 있습니다.

비교할 두 조건은 다음과 같습니다.

| 조건 | 1단계: DeiT-B → 보조 교사 | 2단계: 보조 교사 → 학생 |
|---|---|---|
| `hkd` | 196개 이미지 토큰 + CLS | 전체 이미지, 같은 HKD 손실 |
| `hkd_masked` | 중요도 상위 98개 토큰 + CLS | 전체 이미지, 같은 HKD 손실 |

두 조건의 **최종 학생 구조·파라미터 수·초기 가중치가 같습니다**.
서로 다른 학생 아키텍처를 비교하는 실험이 아닙니다.
두 조건의 보조 교사도 동일한 지도학습 초기 체크포인트에서 출발합니다.

| 역할 | 구조 | 학습 가능한 파라미터 |
|---|---|---:|
| 원래 교사 | ImageNet pretrained, non-distilled DeiT-B/16 | 0 — 전체 고정 |
| 보조 교사 | CNN-1-A 변형: Conv 16/32/64, FC 128, 10클래스 head | 57,994 |
| 최종 학생 | CNN-1 변형: Conv 8/16/32, FC 64, 10클래스 head | 15,050 |

보조 교사와 학생은 3개 convolution stage와 penultimate FC를 일대일로 대응시킵니다.
보조 교사의 채널/은닉 뉴런 수는 학생의 2배입니다. 파라미터 총수까지 같게 하는 조건은 아닙니다.
각 조건 안에서 해당 모델은 같고, 두 조건 사이에서 바뀌는 것은 1단계의 토큰 선택뿐입니다.

논문과 동일하게 유지한 부분:

- 보조 교사에서 최종 학생으로 이어지는 2단계 구조.
- 1단계는 penultimate feature의 cosine + Student-t 관계 분포를 Jeffreys divergence로 맞춤.
- 2단계는 대응하는 4개 표현의 관계 손실을 합산. 앞의 3개 가중치는 `100 * 0.7**epoch`, 마지막은 1.
- Adam, 학습률 `0.001`. 분류 실험의 최종 학생에는 CE를 더함(Appendix A.3).

작은 실험을 위해 바뀐 부분:

- 원 논문의 ResNet-18 교사/데이터 대신 **ImageNet DeiT-B 특징과 COCO single** 사용.
  DeiT-B를 COCO에 fine-tune하지 않으며, 1000클래스 logits도 증류하지 않습니다.
  HKD는 클래스 수가 아닌 특징 사이의 관계를 전이하므로 10클래스 교사 head 학습이 필요하지 않습니다.
- CNN-1의 입력을 32에서 **224×224**로 변경. 마지막 공간 특징을 2×2로 adaptive pooling해 FC 크기 유지.
- 보조 교사의 초기 지도학습은 공유 5 epoch. 공식 코드의 긴 사전학습보다 짧습니다.
- 관계 배치는 32/16/8 중 메모리에 들어가는 값. 공식 배치 128보다 작습니다.
  관계 배치 전체를 한 번에 학습하며 gradient accumulation으로 대체하지 않습니다.
- 논문 식 (5)~(8)에 따라 self-pair를 제외합니다. 공식 공개 코드에는 diagonal이 포함되어 있어
  엄밀한 코드 재현과 차이가 있습니다. 손실 reduction은 공개 코드처럼 B×B 원소 평균입니다.
- CNN 마스크는 마지막 공간 특징의 `mean(channel, feature**2)`를 14×14로 pool해서 선택합니다.
  원본 [MaskedKD](https://arxiv.org/abs/2302.10494)의 학생 CLS attention과 다른 **CNN-guided 변형**입니다.
  교사 positional embedding을 더한 다음 토큰을 제거하며 CLS는 보존합니다.
  마스크를 고르는 보조 CNN은 항상 전체 이미지를 봅니다.

따라서 결과는 **HKD 방법을 적용한 작은 실험**이며 원 논문 수치의 재현이라고 보고하면 안 됩니다.
또한 마스킹 효과가 2단계까지 전달되는지를 측정합니다. CNN 보조 교사의 2단계 연산을
마스킹으로 줄이는 실험은 아닙니다. 무작위 마스크 대조군은 이번 최소 비교에 포함하지 않았습니다.

실행 환경은 Python 3.12와 NVIDIA CUDA GPU를 기준으로 합니다.
[PyTorch의 버전별 설치 안내](https://pytorch.org/get-started/previous-versions/)에서 GPU/드라이버에 맞는 빌드를 선택하세요.
Linux에서 CUDA 12.6 빌드를 사용할 수 있는 환경의 설치 예시는 다음과 같습니다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install 'pytest>=8,<9'
python run_experiment.py smoke
python -m pytest -q
```

이미 준비된 benchmark 데이터가 있으면 그 경로를 그대로 사용하세요.
없으면 아래 명령으로 **고정된 reference 버전의 준비 스크립트**를 가져와 실행할 수 있습니다.
이미지와 segmentation mask, checkpoint는 Git에 넣지 않습니다.

```bash
bash scripts/prepare_reference.sh
bash external/benchmarks/datasets/coco_single/setup.sh
python external/benchmarks/scripts/check_assets.py coco external/benchmarks/data/coco_single --full
python run_experiment.py fetch-teacher
```

데이터는 기존 manifest 그대로 train 3,210 / validation 530 / test 297을 사용합니다.
클래스 순서, split seed, 이미지 수가 다르면 중단합니다. 재분할·부분 집합 추출은 하지 않습니다.
입력은 224로 resize하고 train에만 horizontal flip을 적용합니다.
라벨·세그멘테이션을 마스크 생성에 사용하지 않습니다.

기본 실행은 다음 한 명령입니다. `--data`를 실제 준비된 경로로 변경하세요.

```bash
python run_experiment.py run \
  --data external/benchmarks/data/coco_single \
  --output outputs/pilot-seed0 \
  --hours 4 --vram-gb 3 --seed 0
```

2 GiB 제한이면 `--vram-gb 2`, 4 GiB 제한이면 `--vram-gb 4`를 사용합니다.
실험별 GPU 프로세스를 동시에 띄우지 않습니다. 각 epoch에서 두 조건을 순차 실행하고
조건 실행 순서는 epoch마다 뒤집어서 실행 순서의 영향을 줄입니다.
시간은 기본 4시간, `--hours 3`~`--hours 5`로 바꿀 수 있습니다.
측정만 할 때는 새 output 경로와 `--plan-only`를 사용하세요.

프로파일과 학습 예산은 다음 방식으로 결정합니다.

1. AMP, frozen teacher, 관계 배치 32 → 16 → 8, 교사 microbatch 4 → 2 → 1 순으로 검사.
   **교사만 작은 묶음으로 추론한 특징을 모아서** 관계 배치 전체의 손실을 계산합니다.
2. 실제 이미지 로딩과 forward/backward/optimizer step, validation을 측정합니다.
3. 공유 warmup 5 epoch에 더해, 조건당 `(보조 epoch, 학생 epoch)`를
   `(20,50)`, `(15,50)`, `(10,50)`, `(10,30)` 중 시간에 들어오는 첫 설정으로 고릅니다.
   40% 시간 여유와 마무리 180초를 포함합니다. 성능을 보고 설정을 고르지 않습니다.
4. 최소 설정도 들어오지 않으면 `incomplete_budget`으로 종료합니다.
   실행 중에도 시간을 검사하며, 미완료 조건의 성능으로 승자를 출력하지 않습니다.

PyTorch allocator 상한은 요청 VRAM에서 0.6 GiB를 뺀 값입니다.
CUDA context와 일부 라이브러리 메모리는 이 상한 밖에 있으므로 실제 사용량은
`nvidia-smi`에서도 확인해야 합니다. CSV의 peak는 **PyTorch allocated/reserved** 측정값입니다.
다른 사용자의 GPU 프로세스를 종료하거나 설정을 변경하지 않습니다.

결과를 볼 때는 다음 파일만 확인하면 됩니다.

| 파일 | 내용 |
|---|---|
| `summary.json` | masked−기본 macro accuracy 차이(%p), 1단계 및 전체 branch 속도비, test bootstrap 구간 |
| `runs.csv` | 조건당 1행, validation 선택/마지막 epoch test 점수, 시간, VRAM, 초기값·교사·manifest·코드 해시 |
| `plan.json`, `metadata.json` | 실제 선택된 epoch/배치, 프로파일, GPU, 데이터 및 코드 출처 |
| `aux_history.json`, `student_history.json` | epoch별 손실·validation·소요 시간·AMP update skip 수 |
| `test_*.json` | 샘플별 예측과 클래스별 정확도, validation 선택/마지막 epoch 구분 |
| `*_best.pt`, `*_last.pt` | 보조/학생 checkpoint. 최선의 epoch는 validation macro로만 선택 |
| `*_paired_progress.pt` | 마지막 완료된 paired epoch의 모델·optimizer·scaler 상태 |

주 지표는 test **macro accuracy**입니다. 같은 학습량에서 정확도와 비용을 따로 비교하세요.
탐색 기준은 "일반 HKD 대비 정확도 하락 1%p 이내이면서 실제 시간이 줄어드는가"로 잡을 수 있습니다.
1 seed와 test 297장만으로 일반적인 우열을 단정하지 않습니다.
bootstrap은 같은 test 샘플에 대한 paired/클래스별 재표집이며 학습 seed 변동을 반영하지 않습니다.
관계 손실이 0에 가깝거나 AMP skip이 계속 발생하면 학습 상태부터 점검하세요.

전체 branch 시간은 보조 교사 학습과 최종 학생 학습의 합이며 validation 시간을 포함합니다.
공통 준비 5 epoch는 별도 기록합니다. 데이터 다운로드·사전 측정·checkpoint 저장·마지막 test는
branch 속도비에서 제외하고, 사전 측정을 포함한 전체 wall time도 별도로 제공합니다.
두 조건의 최종 학생은 같은 구조이므로 **최종 추론 속도 개선을 주장하지 않습니다**.

미완료 시 `status.json`을 먼저 확인하세요. 부분 결과는 정상 완료된 비교로 취급하지 않습니다.
checkpoint는 보존하지만 이 최소 runner에는 자동 resume CLI가 없습니다.
기존 output 폴더를 덮어쓰지 않으며, 재실행에는 새 폴더가 필요합니다.

구현 위치: `hkd/core.py`(모델·손실·토큰 제거), `hkd/data.py`(manifest·paired augmentation),
`run_experiment.py`(프로파일·학습·결과), `tests/`(수식·단계별 학습·비교 조건 검증).
