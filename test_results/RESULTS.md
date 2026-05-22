# 라발리에 마이크 + 모델 테스트 결과

## 환경
- 하드웨어: Raspberry Pi + Lavalier USB Microphone (AB13X USB Audio)
- 녹음 포맷: 16kHz, mono, 16-bit PCM
- 모델: YAMNet embedding + Dense classifier (glass_classifier.h5)

## 테스트 결과

### 1. 학습 데이터 검증 (sanity check)
- 파일: `data_clean/glass/156578__splicesound__picture-frame-dropped...`
- 결과: **glass 98.3%**, normal 1.7%
- 결론: 모델 정상 동작 확인

### 2. 라발리에 마이크 - 말소리 녹음
- 파일: `test_results/test.wav`
- 내용: "테스트" 발화 5초 녹음
- 결과: glass 0.3%, **normal 99.7%**
- 결론: 일반 발화는 정확히 normal로 분류

### 3. 라발리에 마이크 - 유리 깨지는 효과음 녹음
- 파일: `test_results/glass_test2.wav`
- 내용: 유튜브 유리 깨지는 효과음 5초 녹음 (스피커 → 마이크)
- 녹음 품질: RMS -25.77dB, Max 0.99 (양호)

**평균 pooling 방식 (기존):**
- glass 35.5%, normal 64.5% → 오판 (normal)

**프레임별 max pooling 방식 (개선):**
| 프레임 | glass | normal |
|--------|-------|--------|
| 0 | 7.7% | 92.3% |
| 1 | **90.8%** | 9.2% |
| 2 | 4.8% | 95.2% |
| 3 | 2.6% | 97.4% |
| 4 | **99.1%** | 0.9% |
| 5 | 8.2% | 91.8% |

- 최대값 결과: **glass 99.1%** (프레임 4) → 정확히 glass 분류

## 인사이트
- 짧은 임팩트성 소리는 mean pooling이 무음 구간에 묻혀 오판됨
- 프레임별 max pooling으로 전환 시 짧은 유리 깨짐도 정확히 검출
- 실시간 감지 시스템에서는 max pooling 방식이 더 적합

## 다음 단계
- 다양한 유리 효과음으로 추가 테스트
- False positive 테스트 (박수, 발걸음, 책 떨어지는 소리 등)
- 실시간 감지 스크립트 구현
