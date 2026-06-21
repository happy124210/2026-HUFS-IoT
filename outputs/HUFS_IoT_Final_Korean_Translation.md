# Edge AI 위협음 감지 — 한국어 해설본

예상 발표 시간: 약 9분. 실제 발표는 영어로 진행합니다.

## 슬라이드 1 — Edge AI 위협음 감지

이 프로젝트는 유리 파손음과 비명을 감지하는 Edge AI 보안 시스템입니다. Raspberry Pi에서 실시간 오디오 분류를 실행하고 물리적 경보 및 원격 알림으로 연결합니다.

## 슬라이드 2 — 왜 위협음을 감지하는가?

카메라와 움직임 센서에는 사각지대가 있습니다. 마이크는 독립적인 감지 신호를 추가하며, Edge inference는 원본 오디오를 계속 업로드하지 않도록 합니다. 이벤트가 확인되면 로컬 및 원격 대응을 즉시 실행할 수 있습니다.

## 슬라이드 3 — 프로젝트 아이디어와 목표

핵심 개념은 소리를 추가적인 보안 센서로 사용하는 것입니다. 세 클래스의 안정적인 감지, 개인정보를 보호하는 로컬 추론, IoT 대응 연결을 목표로 했습니다. 따라서 threat recall, false alarm, Raspberry Pi 실용성의 균형이 중요합니다.

## 슬라이드 4 — 전체 시스템 설계

오디오는 rolling window로 수집됩니다. YAMNet이 embedding과 semantic score를 생성하고, custom head가 세 클래스 확률을 출력합니다. Fusion이 최종 이벤트를 결정한 뒤 IoT 계층이 경보를 실행합니다.

## 슬라이드 5 — 하드웨어 구조

모든 AI 추론은 Raspberry Pi 4에서 실행됩니다. 마이크는 오디오를 입력하고, 카메라는 증거 사진을 촬영하며, LED와 부저는 로컬 경보를 제공합니다. PC의 PuTTY는 원격 terminal일 뿐 AI를 실행하지 않습니다.

## 슬라이드 6 — 소프트웨어와 코드 구조

realtime_detect.py가 실시간 루프를 실행하고 audio_pipeline.py가 오디오 전처리를 담당합니다. H5 custom model이 과제 분류를 수행하며 detection_policy.py가 Fusion 규칙을 적용합니다. GPIO, 카메라, Telegram, 이메일은 각각 별도 모듈로 구성됩니다.

## 슬라이드 7 — YAMNet-only 베이스라인

베이스라인은 custom head를 제거하고 YAMNet의 범용 AudioSet score를 세 클래스로 직접 매핑합니다. 낮은 scream recall은 범용 score만으로는 우리 과제에 충분히 특화되지 않는다는 점을 보여줍니다.

## 슬라이드 8 — 최종 모델 구조

한 번의 YAMNet 실행에서 두 출력이 나옵니다. 1,024차원 embedding은 custom dense head에 입력되고, 521개 AudioSet class 중 관련 score는 보조 확인에 사용됩니다. 1,024는 특징 수이고 521은 범용 클래스 수입니다.

## 슬라이드 9 — Frame-aligned Fusion 정책

두 가지 판정 경로가 있습니다. 강한 classifier 증거가 여러 프레임 동안 지속되면 직접 감지합니다. 더 민감한 경로는 같은 프레임에서 classifier와 관련 YAMNet score가 함께 동의할 때만 감지합니다.

## 슬라이드 10 — 데이터 출처

AudioSet으로 다양한 음향 환경을 확보하고, Raspberry Pi 마이크 직접 녹음으로 실제 배포 영역을 반영했습니다. Speech, music, impact, footsteps 같은 현실적인 normal hard negative도 포함했습니다.

## 슬라이드 11 — 증강 및 최종 데이터 규모

Gain 변화, noise, filtering, 여러 SNR과 위치에서의 background mixing을 적용했습니다. 최종 학습 입력은 source audio 732개와 증강·혼합 오디오 4,458개를 합친 총 5,190개입니다.

## 슬라이드 12 — 학습과 평가 방법

원본 source 기준으로 데이터를 분리하고 YAMNet을 동결한 뒤 frame-level embedding을 cache했습니다. Regularization과 early stopping으로 dense head를 학습하고 validation의 최소 recall 조건을 이용해 Fusion candidate를 선택했습니다.

## 슬라이드 13 — 시행착오

Mean pooling은 짧은 glass impact를 희석했고, 광범위한 label은 노이즈를 만들었으며, YAMNet 동의 없는 완화 threshold는 normal 오탐을 증가시켰습니다. 각 실패가 최종 설계 변경으로 이어졌습니다.

## 슬라이드 14 — 베이스라인과 최종 Fusion 비교

Final Fusion은 YAMNet-only보다 scream recall과 Macro F1을 크게 개선했습니다. 대신 normal false alarm은 다소 증가했습니다. Test split을 이전 반복에서 확인한 적이 있으므로 새로운 외부 데이터 검증이 필요합니다.

## 슬라이드 15 — 결과와 기여

과제 특화 classifier, frame-level Fusion, Raspberry Pi edge deployment, source-separated evaluation이 주요 기여입니다. 오디오 모델을 실제 IoT 보안 대응과 연결했다는 점이 프로젝트의 실용적 의미입니다.

## 슬라이드 16 — Raspberry Pi 실시간 처리

실시간 시스템은 3초 rolling window와 1초 hop을 사용합니다. Raspberry Pi에서 WAV load, YAMNet, H5 classifier, Fusion decision을 포함한 모델 처리 평균은 약 287ms였습니다.

## 슬라이드 17 — 향후 개선

새로운 외부 데이터 검증, 실제 마이크 threat 및 hard negative 추가, calibration 또는 compact temporal model, 지속 부하 시험, 장시간 시간당 오탐 측정이 필요합니다.

## 슬라이드 18 — 결론

Transfer learning으로 실용적인 detector를 만들었고 custom head가 YAMNet feature를 과제에 맞게 특화했습니다. Frame-aligned Fusion은 과제 특화 증거와 범용 의미 증거를 결합했습니다. 다음 단계는 새로운 외부 데이터 검증입니다.
