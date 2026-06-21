# 테스트 결과 정리

## 최종 Fusion 평가 결과 (2026-06-21)

최종 실행 정책은 YAMNet의 embedding과 AudioSet score를 한 번의 forward pass에서
함께 사용한다. 높은 custom-classifier 확신은 단독으로 허용하고, 중간 확신 경로는
같은 프레임에서 YAMNet과 동의해야 한다. 정책값은 source-separated validation
156개에서만 선택했다. 대규모 grid search 대신 recall-oriented, balanced,
precision-oriented 세 후보만 이번 재검증용으로 고정하고, 두 이벤트 recall이 각각 70% 이상인
후보 중 Macro F1이 가장 높은 balanced 정책을 선택했다. 선택 프로그램은 test를
로드하지 않으며, test 평가는 별도 프로그램으로 실행한다.

| 평가 대상 | Test Macro F1 | Normal 오탐률 | Glass Recall | Scream Recall |
|---|---:|---:|---:|---:|
| 최종 frame-aligned Fusion | **0.837** | **9.2%** | 75.9% | **78.9%** |
| 이전 classifier-only 기준 | 0.801 | 13.3% | **82.8%** | 71.1% |

- 최종 Fusion confusion matrix 기준으로 test 165개 중 141개를 맞혔다. Normal 오탐은
  9/98, glass 감지는 22/29, scream 감지는 30/38이다.
- YAMNet score와 embedding은 동일한 forward pass에서 나오므로 Fusion을 위해 YAMNet을
  다시 호출하지 않는다.
- test split은 이전 프로젝트 반복에서 이미 분석된 적이 있다. 이번 정책 탐색 코드는
  validation만 사용했지만, test 결과는 완전히 새로운 미공개 holdout이 아니라는 한계가
  있다. 따라서 새 외부 데이터에서 재검증하는 것이 후속 과제다.
- Ablation 결과는 다음과 같다.
  - 최종 strong 경로만 사용: Macro F1 0.791, normal 오탐률 7.1%, glass recall 62.1%,
    scream recall 71.1%.
  - 같은 custom 경로에서 YAMNet 조건 제거: Macro F1 0.616, normal 오탐률 58.2%.
  - 최종 frame-aligned Fusion: Macro F1 0.837, normal 오탐률 9.2%.
  따라서 최종 개선은 단순 custom threshold 변경만으로 설명되지 않으며, YAMNet
  조건이 완화 경로의 대량 오탐을 억제한다.
- 모든 정제 음원, 파일 예측, 오프라인 평가, 실시간 기본 window를 3초로 통일했다.
  실시간 hop은 1초다.
- 현재 Windows PC에서 3초 음원 30개를 대상으로 한 YAMNet + H5 head + Fusion 추론은
  평균 57.4ms, p95 60.6ms였다. 이는 마이크 buffering과 경보 장치 시간을 제외하며
  Raspberry Pi 성능을 의미하지 않는다.
- 정책 선택: `test_results/training/fusion_policy_selection_20260621.json`
- 분리 평가: `test_results/evaluations/final_fusion_ablation_20260621.json`
- PC benchmark: `test_results/benchmarks/end_to_end_pc_20260621.json`

## TODO: Raspberry Pi 전체 지연 측정 (오늘 저녁)

현재 발표 수치에는 Raspberry Pi 전체 지연이 포함되어 있지 않다. 아래 측정을 완료하기
전까지 PC의 57.4ms 평균값을 Raspberry Pi 성능이나 실제 경보 지연으로 설명하지 않는다.

- [ ] 실제 데모에 사용할 Raspberry Pi에서 최종 Fusion 코드를 실행한다.
- [ ] 기기 모델, RAM, OS, Python/TensorFlow 버전, 전원 모드를 기록한다.
- [ ] 3초 window와 1초 hop을 유지하고, 최소 5회 warm-up 후 30회 이상 측정한다.
- [ ] 순수 추론 지연을 측정한다: YAMNet 1회 + H5 head + Fusion decision.
- [ ] 전체 지연을 별도로 측정한다: 이벤트가 마이크에 입력된 시점부터 실제 경보
  출력(GPIO/화면/네트워크 중 데모에서 사용하는 출력)이 발생한 시점까지.
- [ ] 두 측정 모두 mean, median, p95, max를 기록하고, 정상/유리 파손/비명 사례를
  모두 포함한다.
- [ ] 결과를 `test_results/benchmarks/end_to_end_raspberry_pi_20260621.json`에 저장하고
  이 문서 및 발표자료의 Raspberry Pi 미측정 문구를 실제 결과로 갱신한다.

주의: 기존 `src/benchmark_end_to_end.py`는 파일 로드와 추론만 측정하므로 마이크
buffering, 3초 window 대기, 1초 hop 정렬 대기, GPIO/카메라/네트워크 출력 시간을
포함하지 않는다. 따라서 이 스크립트의 Raspberry Pi 실행 결과만으로 "전체 지연"을
주장하면 안 된다. 실제 전체 지연은 외부 타임스탬프 또는 동기화된 로그로 별도 측정한다.

## 이전 classifier-only 평가 결과 (2026-06-20)

아래 수치는 원본 음원 기준으로 train/validation/test를 분리하고, 동일 YouTube ID와
정확 중복 파일이 여러 split에 포함되지 않도록 수정한 뒤 얻은 결과다. 범용
Smash/Breaking/Crushing 라벨로 수집된 기존 AudioSet glass 158개는 제외하고,
엄격한 Glass/Shatter 라벨로 새로 수집한 43개만 포함했다. Test set은 165개 원본
그룹이며, 후보와 임계값 선택에는 validation만 사용했다.

| 최종 평가 대상 | Test Macro F1 | Normal 오탐률 | Glass Recall | Scream Recall |
|------|--------------:|---------------:|-------------:|--------------:|
| 기존 모델 + validation 선택 정책 | **0.801** | 13.3% | **82.8%** | **71.1%** |

- Candidate C는 validation에서 normal 오탐률 3.2%와 glass recall 84.0%를 달성했지만
  scream recall이 60.5%라 승격 기준 70%를 충족하지 못했다. 운영 모델은 교체하지 않았다.
- 별도 탐색적 ablation에서 기존 모델 + 당시 YAMNet fusion은 test Macro F1 0.804,
  normal 오탐률 7.1%, glass recall 82.8%, scream recall 57.9%였다. 이는 구성요소
  분석 결과이며 모델 선택 수치로 사용하지 않는다.
- 데이터 정제 전후 test 구성이 달라졌으므로 이전 0.738과 현재 0.801은 완전히 동일한
  표본의 직접 비교가 아니다. 현재 수치는 라벨 신뢰도가 더 높은 평가셋의 기준선이다.
- 13개 review set에서 측정한 84.6%는 알려진 실패 사례에 대한 회귀 테스트이며,
  독립적인 일반화 성능이 아니다.
- 현재 통합 데모는 TensorFlow Hub YAMNet과 H5 분류기를 사용한다. TFLite 파일은
  분류기 head 변환 결과이며 전체 파이프라인이 TFLite로 실행되는 것은 아니다.
- PC에서 분류기 head의 H5 추론은 10개 embedding frame 기준 평균 약 69ms였다.
  현재 PC 환경은 TensorFlow 2.21과 tensorflow-intel 2.15가 동시에 설치되어 TFLite
  native wrapper ABI가 충돌하므로 TFLite 수치는 측정하지 못했다. Raspberry Pi의
  clean LiteRT 환경에서 별도로 측정해야 한다.
- 상세 결과: `test_results/training/run_20260620_strict_labels.json`,
  `test_results/evaluations/strict_label_ablation_20260620.json`,
  `test_results/data_audit/audioset_label_audit_20260620.json`,
  `test_results/benchmarks/classifier_pc_20260620.json`

## 폐기된 이전 운영 정책 (기록용)

아래 0.97/0.92 정책은 2026-06-21 최종 frame-aligned Fusion으로 대체되었다.
현재 `main.py`, `realtime_detect.py`, `predict.py`는 이 이전 정책을 사용하지 않는다.

당시 메인 실행 경로에는 6월 20일에 재평가한 **기존 H5 분류기 + YAMNet
corroboration** 정책을 적용했다. 모델 자체는 교체하지 않았다.

| 정책값 | 적용값 |
|---|---:|
| custom threshold (glass / scream) | 0.97 / 0.92 |
| 최소 연속 프레임 (glass / scream) | 1 / 3 |
| scream mean / margin | 0.35 / 0.15 |
| YAMNet support (Glass·Shatter / Screaming) | 0.10 / 0.05 |
| fusion custom floor (glass / scream) | 0.70 / 0.92 |
| fusion 최소 연속 프레임 (glass / scream) | 1 / 2 |

채택 근거는 다음과 같다.

- 6월 19일에 수집한 남성 scream false negative 3개는 기존 정책에서 0/3,
  Candidate C에서 1/3이었지만, 기존 모델에 YAMNet fusion을 적용하면 3/3을 감지했다.
- 기존 13개 review 회귀셋에서도 fusion은 scream 4/4와 normal 5/5를 유지했고,
  전체 11/13(84.6%)이었다. 다만 표본이 작고 알려진 실패 사례 중심이므로 일반화
  성능으로 해석하지 않는다.
- 엄격 라벨 test 165개에서 같은 fusion 구성은 Macro F1 0.804, normal 오탐률 7.1%,
  glass recall 82.8%, scream recall 57.9%였다. 공식 classifier-only 선택 정책의
  scream recall 71.1%보다 낮으므로, 이번 결정은 전체 recall 최대화가 아니라 현장
  오탐 억제와 확인된 남성 scream 회귀 복구를 우선한 운영상 trade-off다.
- YAMNet 단독 감지는 허용하지 않는다. custom classifier가 floor와 연속 프레임
  조건을 먼저 충족한 경우에만 YAMNet이 보조 근거가 되어 연속 프레임 조건을
  완화한다. 이 제한으로 외부 범용 모델의 단독 오탐이 즉시 경보가 되는 것을 막는다.
- 6월 20일 classifier-only 결과 재현을 위해 해당 정책은
  `LEGACY_CLASSIFIER_ONLY_POLICY`로만 보존한다. 최종 Fusion 정책과 혼동하지 않는다.

근거 파일: `test_results/archive/evaluations/review_20260620_male_scream_baseline.json`,
`review_20260620_male_scream_candidate_c.json`,
`review_20260620_male_scream_yamnet_fusion.json`,
`review_20260620_review_yamnet_fusion.json`,
`test_results/evaluations/strict_label_ablation_20260620.json`.

## 실환경 기능 테스트 (2026-06-19, 참고용)

아래 결과는 소수의 수동 시연과 5분 모니터링 결과로, 공식 정확도나 시간당 오탐률로
해석하지 않는다.

> 이 절은 폐기된 0.97/0.92 정책으로 수행한 소수 수동 시연의 기록이다. 현재 최종
> frame-aligned Fusion과 정책값이 다르므로 현재 시스템의 성능으로 인용하지 않는다.

## 테스트 환경
- 라즈베리 파이 4 + 라발리에 USB 마이크 (device 1)
- 모델: glass_classifier.h5 (8.0MB, Jun 19 최신 모델)
- 임계값: glass=97.0%, scream=92.0%
- 연속 프레임: glass=1, scream=3
- 추가 필터: scream_mean>=35%, margin>=15%
- 환경: 조용한 방, 선풍기 꺼짐

## glass 감지 테스트

| 테스트 상황 | 시도 횟수 | 감지 성공 | 오판 | 비고 |
|------------|---------|---------|------|------|
| 유리 효과음 (유튜브, 긴 것) | 2회 | 2회 ✅ | 0회 | 신뢰도 99.9% |
| 조용한 환경 (아무 소리 없음) | 5분 모니터링 | 0회 | 1회 ❌ | 마지막에 혼자 울림 |

## scream 감지 테스트

| 테스트 상황 | 시도 횟수 | 감지 성공 | 오판 | 비고 |
|------------|---------|---------|------|------|
| 직접 소리지름 | 1회 | 1회 ✅ | 0회 | 신뢰도 99.7% |

## 해당 기능 테스트에서 확인한 개선점
- scream 추가 필터 적용으로 오판 감소
- 조용한 환경에서 오판 횟수 감소
- 직접 소리질러도 scream 감지 성공

## 남은 문제
- 조용한 환경에서 glass 수치가 90% 이상 지속적으로 높음
- 간헐적 glass 오판 여전히 발생
