# 테스트 결과 정리 (최종 갱신: 2026-06-18)

## 2026-06-17 review 오판 반영 재학습

### 대상과 방법

- review 오판 원본 13개 전부 반영: glass 4개, normal 5개, scream 4개
- 신규 생성 데이터: glass 460개(standalone 80 + normal 혼합 380), normal 60개, scream 572개(standalone 100 + normal 혼합 472)
- 전체 학습 입력: embedding 108,709개
- source-group 분리: train 86,808개 / validation 21,901개, source group 920 / 231개
- 모델: `model/glass_classifier.h5` (2026-06-18 재학습), `model/glass_classifier.tflite` 재변환
- baseline: `model/baselines/glass_classifier_20260615.h5`
- 평가셋: `evaluation_data/review_20260617`의 원본 13개만 사용(증강본 제외)
- 정책: glass threshold 0.97 / 1 frame, scream threshold 0.92 / 3 consecutive frames

> 이 평가셋은 이번 학습에 반영된 오판 원본으로 구성된 hard-case 교정 확인용 셋이다. 따라서 아래 수치는 독립 holdout 일반화 성능이 아니라, 수집된 오판이 얼마나 교정됐는지를 나타낸다.

### baseline 대비 결과

| 지표 | 2026-06-15 baseline | 2026-06-18 재학습 | 변화 |
|---|---:|---:|---:|
| 전체 정답 | 0/13 (0.0%) | 6/13 (46.2%) | +6개, +46.2%p |
| macro F1 | 0.000 | 0.329 | +0.329 |
| normal false alarm | 5/5 (100%) | 0/5 (0%) | 100% 감소 |
| glass → normal | 4/4 | 3/4 | 25% 감소 |
| normal → scream | 5/5 | 0/5 | 100% 감소 |
| scream → normal | 4/4 | 4/4 | 변화 없음 |
| scream/normal 양방향 혼동 합계 | 9/9 | 4/9 | 55.6% 감소 |
| glass/normal 양방향 혼동 합계 | 4/4 | 3/4 | 25.0% 감소 |

재학습 모델 confusion matrix (행=actual, 열=predicted):

| actual \\ predicted | glass | normal | scream |
|---|---:|---:|---:|
| glass | 1 | 3 | 0 |
| normal | 0 | 5 | 0 |
| scream | 0 | 4 | 0 |

### 임계값 탐색

- 탐색 범위: glass/scream 각각 0.70~0.99, 0.01 간격(900개 조합)
- 목적 함수: `macro F1 - normal false alarm rate`
- 재학습 모델 최고 점수: 0.329 (macro F1 0.329, normal false alarm 0.000)
- 동률 최고 조합에 현행 scream 0.92가 포함된다. 예: glass 0.70, scream 0.92
- 현행 glass 0.97에서도 실제 confusion matrix와 점수는 동일하므로, 13개 교정셋만 근거로 운영 임계값을 낮추지 않고 **0.97/0.92를 유지**한다.
- scream 4개는 threshold 조정만으로 복구되지 않았다. 현재 3-frame 연속 조건과 mean/margin 조건을 함께 만족하지 못하므로, 독립 scream/normal holdout을 추가한 뒤 연속 프레임 조건까지 별도로 탐색해야 한다.

### 재현 명령과 산출물

```powershell
python src/evaluate_model.py --dataset-dir evaluation_data/review_20260617 --model-path model/baselines/glass_classifier_20260615.h5 --search-thresholds --json-output test_results/evaluations/review_20260617_baseline.json
python src/evaluate_model.py --dataset-dir evaluation_data/review_20260617 --model-path model/glass_classifier.h5 --search-thresholds --json-output test_results/evaluations/review_20260617_retrained.json
```

- baseline 상세 결과: `test_results/evaluations/review_20260617_baseline.json`
- 재학습 상세 결과: `test_results/evaluations/review_20260617_retrained.json`

---

## 2026-06-16 라즈베리파이 현장 테스트(이전 결과)

## 테스트 환경
- 라즈베리 파이 4 + 라발리에 USB 마이크 (device 1)
- 모델: glass_classifier.h5 (8.0MB, 윤아씨 Jun 15 모델)
- 임계값: glass=97.0%, scream=92.0%
- 연속 프레임: glass=1, scream=2
- 환경: 조용한 방, 선풍기 꺼짐

## glass 감지 테스트

| 테스트 상황 | 시도 횟수 | 감지 성공 | 오판 | 비고 |
|------------|---------|---------|------|------|
| 조용한 환경 (아무 소리 없음) | 10분 모니터링 | 0회 | 3회 ❌ | 원인 불명, 혼자 울림 |
| 유리 효과음 (유튜브, 10초 이상) | 3회 | 3회 ✅ | 0회 | 신뢰도 98~100% |
| 유리 효과음 (짧게 1번) | 5회 | 0회 ❌ | 0회 | 2초 윈도우 내 포착 실패 |

## scream 감지 테스트

| 테스트 상황 | 시도 횟수 | 감지 성공 | 오판 | 비고 |
|------------|---------|---------|------|------|
| 직접 소리지름 (크게) | 2회 | 1회 ✅ | 0회 | 신뢰도 98~99% |
| 노트북 타자 소리 | - | 0회 | 1회 ❌ | scream 98.2% 오판 |
| 휴대폰 놓는 소리 | - | 0회 | 1회 ❌ | glass 오판 |

## normal 감지 테스트

| 테스트 상황 | 모니터링 시간 | normal 판정 | 비고 |
|------------|------------|-----------|------|
| 조용한 환경 | 10분 | 대부분 ✅ | glass 3회 오판 제외 |

## 종합 분석

### 잘 되는 것
- 유리 효과음 (10초 이상 긴 것): 3/3 정상 감지
- 직접 소리지름: 1~2회 중 감지

### 문제점
1. **조용한 환경에서 glass 오판 (3회)**: 배경 노이즈 문제로 추정
2. **짧은 유리 효과음 미감지**: 2초 윈도우 내 포착 어려움
3. **일상 충격음 오판**: 노트북 타자 → scream, 휴대폰 놓는 소리 → glass

### 추가 의견
- 실제 위협 상황의 비명(공포, 고통)과 일상적 소음(카페, 교실 말소리, 충격음) 구분 필요
- 현재 모델이 일상적 충격음을 위협음으로 오판하는 경우 있음
- 카페/교실 등 배경 소음 환경에서의 추가 테스트 필요

## scream 오판 대책 (2026-06-16)

### 관찰
- 직접 소리지름은 크게 낼 때 감지되지만, 노트북 타자 소리가 scream 98.2%로 오판됨
- 즉, 현재 모델은 순간적인 충격/고주파 배경음을 scream embedding과 혼동할 수 있음
- max pooling만 강하게 믿으면 짧은 배경음 피크가 위협 이벤트로 승격됨

### 즉시 적용한 정책
- scream 연속 프레임 기준을 2 → 3으로 강화
- scream 평균 확률이 35% 미만이면 순간 피크로 보고 normal 처리
- scream 최대 확률이 다른 클래스 최대 확률보다 15%p 이상 높을 때만 확정
- glass 정책은 짧은 충격음을 잡아야 하므로 기존 기준 유지

### 추가 데이터 대책
- normal에 노트북 타자, 책상 두드림, 마우스 클릭, 휴대폰 내려놓기, 의자 끄는 소리 추가
- scream은 효과음보다 실제 사람 목소리 비율을 늘리고, 작게/멀리/배경음 섞인 샘플을 별도로 수집
- 카페/교실/선풍기/컴퓨터 팬 소리를 normal hard negative로 넣고 재학습

### 재테스트 항목
- 직접 소리지름 10회 이상: 감지율 확인
- 노트북 타자 5분, 조용한 방 10분: scream 오판 0회 목표
- 카페/교실 배경음: scream 오판 발생 구간은 normal hard negative로 재수집

### 라즈베리파이 수동 리뷰 수집
- 실행: `python src/record_review_samples.py --count 10 --device 1`
- 각 녹음 후 판정 결과를 보고 실제 라벨을 입력
- `Enter`: 판정이 맞음, 학습 데이터에 추가하지 않음
- `g`, `n`, `s`: 실제 라벨이 각각 glass, normal, scream이라는 뜻이며, 판정과 다르면 `data/<실제라벨>`에 저장
- 이후 `python src/preprocess.py` 또는 `retrain.bat` 실행 시 오판 녹음이 학습 데이터에 포함됨
