# 테스트 결과 정리

## 공식 평가 결과 (2026-06-20)

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
- 별도 탐색적 ablation에서 기존 모델 + 배포 YAMNet fusion(RMS gate 제외)은 test Macro F1 0.804,
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

## 실환경 기능 테스트 (2026-06-19, 참고용)

아래 결과는 소수의 수동 시연과 5분 모니터링 결과로, 공식 정확도나 시간당 오탐률로
해석하지 않는다.

> 이 절은 당시의 레거시 데모 정책(glass=97%, scream=92%, YAMNet fusion)을 기록한
> 것이다. 현재 `main.py`, `realtime_detect.py`, `predict.py`는 공식 평가와 동일한
> classifier-only 배포 정책(glass=95%, scream=40%, 연속 프레임 1/3)을 사용한다.
> 따라서 아래 수동 시연 결과를 현재 통합 정책의 성능으로 인용하지 않는다.

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
