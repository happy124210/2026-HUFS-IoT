# 테스트 결과 정리 (2026-06-15)

## 테스트 환경
- 라즈베리 파이 4 + 라발리에 USB 마이크
- 모델: glass_classifier.h5 (8.0MB, 윤아씨 최신 모델)
- 임계값: glass=0.97, scream=0.92
- MIN_EVENT_FRAMES=3, MIN_RMS=0.02, WINDOW_SECONDS=2.0

## 테스트 결과

### glass 감지
| 테스트 | 결과 | 신뢰도 |
|--------|------|--------|
| 유리 효과음 (긴 것) | ✅ 감지 | 100% |
| 유리 효과음 (짧게 한 번) | ❌ 감지 못함 | - |
| 조용한 환경 배경음 | ❌ glass 오판 | 100% |

### scream 감지
| 테스트 | 결과 | 신뢰도 |
|--------|------|--------|
| 직접 소리지름 | ❌ 감지 못함 | - |
| 선풍기 소리 | ❌ scream 오판 | ~95% |

### normal 감지
| 테스트 | 결과 |
|--------|------|
| 조용한 환경 | ✅ 대부분 normal |
| 말소리 | ✅ normal |

## 문제점 분석
1. **짧은 유리 소리 감지 못함**: WINDOW_SECONDS=2.0으로 줄였으나 여전히 짧은 소리 포착 어려움
2. **scream 감지 민감도 낮음**: 직접 소리질러도 감지 못함 → scream 임계값 0.92가 너무 높을 수 있음
3. **선풍기/배경 소음 오판**: glass/scream 오판 여전히 있음

## 개선 제안
- glass는 순간음 특성에 맞춰 1개 프레임, scream은 2개 연속 프레임으로 판정
- 학습과 실시간 추론을 YAMNet 프레임 단위로 통일
- peak 정규화를 제거하고 실제 입력 음량을 유지
- 선풍기 등 오탐 음원을 `normal` hard negative로 추가 후 재학습
- scream 임계값은 먼저 낮추지 않고 별도 평가 데이터로 재산정

## 적용 및 검증 순서
1. 선풍기, 에어컨, 청소기 등 오탐 음원을 `data/normal`에 추가
2. 실제 사용자 비명을 거리와 음높이를 달리해 `data/scream`에 추가
3. `retrain.bat`으로 전처리, 증강, 프레임 단위 재학습 실행
4. 별도 평가 파일을 `evaluation_data/glass`, `evaluation_data/normal`, `evaluation_data/scream`에 배치
5. `python src/evaluate_model.py --search-thresholds` 실행
6. 출력된 recall과 normal false alarm rate를 확인한 뒤 `src/detection_policy.py`의 임계값 확정

## 프레임 단위 재학습 결과 (2026-06-15)
- 학습 샘플: 105,898개 YAMNet 프레임
- 분할: train 84,666 / validation 21,232
- 원본 그룹 분할: train 910 / validation 228
- 새 모델 저장: `model/glass_classifier.h5` (2026-06-15 21:02)

### 기존 마이크 녹음 재검증
| 파일 | 실제 내용 | 결과 | 최대 확률 / 판정 근거 |
|------|-----------|------|-----------------------|
| `test.wav` | 일반 말소리 | normal | scream 100% 프레임이 있었지만 연속 조건 불충족 |
| `glass_test2.wav` | 유리 효과음 | glass | glass 100%, 6개 연속 프레임 |
| `scream_test.wav` | 짧은 비명 효과음 | normal | scream 98.8%, 1개 프레임 |
| `scream_test2.wav` | 직접 소리지름 | normal | scream 99.8%, 0.92 이상 연속 1개 프레임 |

### 후속 조정
- scream은 단일 임계값을 낮추지 않고 hysteresis 판정을 적용한다.
- 시작 프레임은 높은 임계값을 요구하고, 인접 지속 프레임은 더 낮은 임계값으로 확인한다.
- 실제 라즈베리파이 평가 데이터 확보 후 임계값을 확정한다.
