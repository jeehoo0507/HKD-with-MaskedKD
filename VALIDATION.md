# 검증 기록 — 2026-09-29

환경: macOS arm64, uv 0.11.24, Python 3.12.12, PyTorch 2.8.0, timm 1.0.19. CPU 실행.
Python은 `.local/python/`, 패키지는 `.venv/`에 `./uv.sh sync --locked`로 설치했습니다.

실행한 검증:

- `./uv.sh run --locked python -m pytest -q`: **18 passed**, 10.98초.
- `./uv.sh run --locked python run_experiment.py smoke`: 두 종류의 보조 교사 전이와 최종 학생 학습 step 성공, loss/gradient finite.
- `./uv.sh run --locked python scripts/check_isolation.py`: **isolated**. 실제 Python 런타임·가상환경·캐시·임시 경로가 모두 저장소 내부임을 확인.
- 상속된 외부 캐시/가상환경 경로를 wrapper가 교체하고, 부모 프로세스 환경은 그대로임을 확인.
- annotation 다운로드의 HTTP 200 재시작/206 이어받기와 teacher 저장 위치를 mock으로 검증. 별도 curl/pip 실행 없음.
- 실제 DeiT-B 구조(random weights)에서 원래 forward와 마스킹 없는 wrapper의 penultimate feature 일치.
- 토큰 선택의 위치 정보/CLS 보존, 197 → 99 토큰 감소, teacher microbatch 결과 일치.
- NumPy로 별도 계산한 논문 관계 손실과 PyTorch 값 일치. 교사 gradient 없음.
- 동일 초기값/데이터 순서, 2단계 paired 학습, validation 선택 및 checkpoint 저장 검증.
- 시간 부족 시 미완료 결과를 정상 비교로 출력하지 않는 경로 검증.
- CLI help, Python compile, 준비 스크립트 shell syntax 검사 통과.

측정하지 않은 항목: 실제 COCO 정확도, CUDA AMP 동작, GPU VRAM, 목표 GPU 학습 시간,
실제 pretrained teacher checkpoint 다운로드/로드 및 전체 데이터 학습.
테스트의 random teacher와 synthetic 데이터 결과는 과학적 성능 결과가 아닙니다.
목표 GPU에서 실행할 때 자동 preflight가 배치/시간을 검사하지만 실제 완료를 보장하지는 않습니다.
