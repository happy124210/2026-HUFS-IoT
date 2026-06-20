# Edge AI Threat-Sound Detection — 한국어 번역본

발표 시간: 발표 8분 + 질의응답 2분. 실제 발표는 영어로 진행합니다.

## 슬라이드 1 — Edge AI 위협음 감지 (0:25)

안녕하세요. 저희 프로젝트는 유리 파손음과 비명이라는 두 가지 위협음을 감지하는 엣지 AI 보안 시스템입니다. Raspberry Pi에서 실시간으로 오디오를 분류하고, 물리적 경보와 원격 알림을 실행합니다. 오늘은 베이스라인, 저희 모델, 실패했던 접근, 최종적으로 효과가 있었던 방법, 그리고 향후 개선 방향을 설명하겠습니다.

## 슬라이드 2 — 왜 위협음을 들어야 하는가? (0:40)

왜 소리를 사용해야 할까요? 카메라와 움직임 센서는 유용하지만 사각지대가 있습니다. 마이크는 음원이 카메라 시야 밖에 있어도 이벤트를 감지할 수 있습니다. 또한 저희 설계는 엣지에서 추론하므로 원본 오디오를 계속 업로드할 필요가 없습니다. 위협이 확인되면 부저와 LED를 작동하고, 사진을 촬영하고, 로그를 기록하며, Telegram 메시지를 즉시 전송합니다.

슬라이드 문구: 카메라 사각지대 보완 / 로컬 처리로 프라이버시 확보 / 부저·LED·카메라·로그·Telegram 자동 실행.

## 슬라이드 3 — 기준 베이스라인: YAMNet-only ablation (0:50)

기준 아키텍처는 YAMNet-only detector입니다. 오디오를 16kHz로 리샘플링한 뒤 Google의 AudioSet 사전학습 네트워크에 입력합니다. YAMNet은 521개 범용 사운드 클래스 확률을 출력하고, 고정 규칙으로 일부 점수를 glass, scream, normal로 매핑합니다. 엄격한 테스트셋에서 수행한 사후 탐색적 ablation 결과는 Macro F1 0.756, glass recall 20/29, scream recall 19/38이었습니다. 이 비교는 모델이나 임계값 선택에 사용하지 않았습니다.

슬라이드 구조: 16kHz 단일 채널 오디오 → YAMNet → 521개 범용 클래스 확률 → 3개 목표 클래스 규칙 매핑.

## 슬라이드 4 — 유지된 모델: YAMNet 임베딩 + custom head (1:05)

유지된 모델은 YAMNet을 고정 특징 추출기로 사용하고 과제 전용 분류기를 추가합니다. 각 프레임은 1,024차원 임베딩이 됩니다. 분류기는 Dense 512, 256, 128과 3클래스 softmax로 구성되며 batch normalization, dropout, L2 regularization을 사용합니다. 공식 정책은 validation에서 선택되었으며 classifier-only 임계값 glass 0.95, scream 0.40과 연속 프레임 1/3을 사용합니다.

슬라이드 구조: YAMNet 1,024차원 임베딩 → Dense 512/256/128 → glass/normal/scream softmax → 클래스별 임계값 및 연속 프레임 정책.

## 슬라이드 5 — 더 큰 데이터가 아니라 더 엄격한 실험 (0:50)

가장 큰 방법론적 개선은 평가 프로토콜이었습니다. smash, breaking, crushing 같은 광범위한 라벨에서 수집된 158개 클립은 실제 유리가 아닌 경우가 많아 제외했습니다. 또한 모든 정제 및 증강 파일을 원본 소스 기준으로 그룹화하고, 325개의 중복 alias를 병합했으며, 같은 YouTube ID가 서로 다른 split에 들어가지 않도록 했습니다. 후보 모델과 임계값은 validation에서만 선택했습니다. 테스트셋은 165개의 독립 원본 그룹으로 구성되며, 선택이 끝난 뒤 한 번만 평가했습니다.

슬라이드 수치: Train 732 / Validation 156 / Test 165 / 중복 alias 325개 병합 / 광범위 glass 라벨 158개 제외.

## 슬라이드 6 — 공식 테스트 결과 (0:50)

유지된 모델은 165개 원본 그룹 중 136개를 맞혀 accuracy 82.4%, Macro F1 0.801을 기록했습니다. Glass recall은 24/29, scream recall은 27/38입니다. Normal 98개 중 13개에서 오탐이 발생했습니다. 표본 수가 크지 않으므로 백분율과 분자/분모를 함께 제시해야 합니다. 승격 조건은 validation에서 기존 모델보다 높은 Macro F1, normal 오탐률 5% 이하, 두 이벤트 recall 70% 이상이었으며 어떤 후보도 모두 충족하지 못했습니다.

슬라이드 수치: Test 165 / Accuracy 136/165 / Macro F1 0.801 / Normal false alarm 13/98 / Glass recall 24/29 / Scream recall 27/38.

## 슬라이드 7 — 시행착오가 모델뿐 아니라 평가를 바꾸었다 (1:00)

세 가지 실패가 접근 방식을 바꾸었습니다. 첫째, 진단용 예시에서 mean pooling이 긴 구간 안의 짧은 유리 충격음을 숨길 수 있음을 확인했습니다. 이는 frame-level 판단의 동기가 되었지만 공식 정확도 결과는 아닙니다. 둘째, 광범위한 AudioSet 라벨의 노이즈 때문에 검증된 glass와 shatter 소스만 유지했습니다. 셋째, Candidate C는 validation 오탐률을 3.2%로 낮췄지만 scream recall이 60.5%로 떨어졌고 Macro F1도 기존 모델보다 낮아 승격하지 않았습니다.

## 슬라이드 8 — 공식 정책과 원래 데모 정책 비교 (0:45)

공식 정책과 원래 데모 정책은 서로 다른 위험을 최적화합니다. 공식 classifier-only 정책은 Macro F1 0.801, normal 오탐률 13.3%, glass recall 82.8%, scream recall 71.1%입니다. 원래 데모 정책은 더 높은 임계값, YAMNet corroboration, scream mean·margin 필터를 사용했고 실제 마이크 루프에는 RMS gate도 있었습니다. 다만 과거 오프라인 ablation에는 RMS gate가 포함되지 않았습니다. 해당 ablation의 Macro F1은 0.804, 오탐률은 7.1%, glass recall은 82.8%, scream recall은 57.9%였습니다. 이 비교는 모델 선택에 사용하지 않았습니다.

## 슬라이드 9 — 향후 개선 가능성 (0:55)

세 가지 개선 방향이 있습니다. 데이터 측면에서는 실제 배포 공간의 비명과 충격음, 말소리, 발걸음 같은 hard negative가 더 필요합니다. 모델 측면에서는 일부 YAMNet 계층 fine-tuning, calibration, 소형 temporal network를 검토할 수 있습니다. 배포 측면에서 현재 런타임은 TensorFlow Hub + H5이며 TFLite 파일은 분류기 head만 변환한 것입니다. 따라서 Raspberry Pi 전체 파이프라인 지연과 시간당 오탐을 별도로 측정해야 합니다.

## 슬라이드 10 — 결론 (0:40)

결론적으로 전이학습은 감지기를 실용적으로 만들었고, 원본 소스 단위 분할과 명시적 validation gate는 평가를 더 타당하게 만들었습니다. 공식 정책과 원래 데모 정책의 비교는 오탐 감소와 비명 미탐 증가 사이의 trade-off를 보여줍니다. 다음 목표는 위협 recall을 희생하지 않으면서 오탐을 줄이는 것입니다. 감사합니다. 질문 받겠습니다.
