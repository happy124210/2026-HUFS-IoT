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

---

## 2차 테스트 결과 (2026-06-04) — 3-class 검증

### 4. 비명 효과음 (학습 데이터)
- 파일: `data_clean/scream/scream_001_clean.wav`
- 결과: glass 0.0%, normal 0.0%, **scream 99.9%**
- 결론: ✅ scream 클래스 정확히 분류

### 5. 비명 효과음 (라발리에 마이크 녹음)
- 파일: `test_results/scream_test.wav`
- 내용: 유튜브 scream 효과음을 라발리에 마이크로 녹음
- 결과: glass 0.0%, normal 0.1%, **scream 99.8%** (max pooling 기준)
- 결론: ✅ 실제 마이크 환경에서도 scream 정확히 감지

### 6. 직접 소리지르기 (실제 사람 목소리)
- 파일: `test_results/scream_test2.wav`
- 내용: 라발리에 마이크 앞에서 직접 소리지름
- 결과: glass 0.8%, normal 28.3%, **scream 70.9%** (max pooling 기준)
- 결론: ✅ 실제 사람 목소리도 scream으로 분류

## 전체 테스트 요약

| 테스트 | 입력 방식 | 결과 | 신뢰도 |
|--------|---------|------|--------|
| 말소리 | 라발리에 마이크 | ✅ normal | 99.7% |
| 유리 효과음 (mean) | 라발리에 마이크 | ❌ normal | 35.5% |
| 유리 효과음 (max) | 라발리에 마이크 | ✅ glass | 99.1% |
| 학습 데이터 glass | wav 파일 | ✅ glass | 98.3% |
| 학습 데이터 scream | wav 파일 | ✅ scream | 99.9% |
| 비명 효과음 | 라발리에 마이크 | ✅ scream | 99.8% |
| 직접 소리지름 | 라발리에 마이크 | ✅ scream | 70.9% |

## 인사이트
- 효과음 vs 실제 목소리 신뢰도 차이 (99.8% vs 70.9%) — Domain Mismatch
- 오디오 증강으로 다양한 목소리 톤 학습 필요
- 3-class (glass/normal/scream) 모두 실제 마이크 환경에서 검증 완료
