# Edge AI Threat-Sound Detection — 한국어 번역본

발표 시간: 발표 8분 + 질의응답 2분. 실제 발표는 영어로 진행합니다.

## 슬라이드 1 — Edge AI 위협음 감지 (0:25)

안녕하세요. 저희 프로젝트는 유리 파손음과 비명을 감지하는 엣지 AI 보안 시스템입니다. Raspberry Pi에서 실시간으로 오디오를 분류하고 물리적 경보와 원격 알림을 실행합니다. 오늘은 베이스라인, 최종 Fusion 모델, 시행착오, 최종적으로 효과가 있었던 방법과 향후 개선 방향을 설명하겠습니다.

## 슬라이드 2 — 왜 위협음을 들어야 하는가? (0:40)

카메라와 움직임 센서는 유용하지만 사각지대가 있습니다. 마이크는 음원이 카메라 시야 밖에 있어도 이벤트를 감지할 수 있습니다. 또한 엣지에서 추론하므로 원본 오디오를 계속 업로드할 필요가 없습니다. 위협이 확인되면 부저와 LED를 작동하고, 사진을 촬영하고, 로그를 기록하며, Telegram 메시지를 전송합니다.

## 슬라이드 3 — YAMNet만으로 충분하지 않은 이유 (0:50)

이 슬라이드는 한 가지 질문에 답합니다. 왜 YAMNet만 사용하지 않았는가? 이를 확인하기 위한 대조 비교에서 custom head를 제거하고 YAMNet의 521개 범용 AudioSet score를 고정 규칙으로 glass, scream, normal에 직접 매핑했습니다. 엄격한 test set에서 Macro F1은 0.756이었고 비명은 38개 중 19개, 즉 50%만 감지했습니다. 이는 범용 의미 점수만으로는 프로젝트의 목표 클래스에 충분히 특화되지 않았다는 뜻입니다. 이 비교는 YAMNet-only 방식의 한계를 이해하기 위한 분석이며 모델이나 임계값 선택에는 사용하지 않았습니다. 이러한 한계 때문에 다음 슬라이드의 과제 전용 custom head가 필요합니다.

## 슬라이드 4 — 최종 모델: custom head + YAMNet Fusion (1:05)

최종 시스템은 한 번의 YAMNet forward pass에서 나오는 두 출력을 모두 사용합니다. 1,024차원 embedding은 Dense 512, 256, 128과 3-class softmax로 구성된 과제 전용 classifier에 입력됩니다. 동시에 선택된 AudioSet score는 의미적 보조 증거로 사용됩니다. 최종 규칙은 두 경로로 구성됩니다. Classifier 증거가 여러 프레임 동안 지속되면 단독으로 감지하고, 짧은 corroboration 경로에서는 같은 프레임에서 classifier와 YAMNet이 동의해야 합니다.

## 슬라이드 5 — 더 큰 데이터가 아니라 더 엄격한 실험 (0:55)

가장 큰 방법론적 개선은 평가 프로토콜입니다. 광범위한 라벨의 158개 클립을 제외하고, 정제·증강 파일을 원본 소스 기준으로 그룹화했으며, 325개의 중복 alias를 병합하고 동일 YouTube ID가 split을 넘지 못하게 했습니다. 이번 재평가에서는 선택 프로그램을 실행하기 전에 recall-oriented, balanced, precision-oriented 세 후보만 고정했습니다. Test를 로드하지 않는 별도 선택 프로그램에서 validation만 비교했고, 두 이벤트 recall 70% 이상을 유지하면서 Macro F1이 가장 높은 balanced 정책을 선택했습니다.

## 슬라이드 6 — 최종 Fusion 결과 (0:55)

최종 Fusion 시스템은 위협음 recall과 normal 오탐 사이에서 가장 좋은 전체 균형을 보였고 Macro F1 0.837을 기록했습니다. 클래스별 결과에서도 두 위협음과 normal 오디오 전반에 걸쳐 균형 잡힌 동작을 보였습니다. 중요한 한계는 정책 선택에는 validation만 사용했지만 이 test split이 이전 프로젝트 반복에서 이미 확인된 적이 있다는 점입니다. 따라서 이 결과는 사후 engineering 평가로 해석하며 새로운 외부 데이터 검증이 여전히 필요합니다.

## 슬라이드 7 — 시행착오가 시스템을 바꾸었다 (1:00)

세 가지 교훈이 접근 방식을 바꿨습니다. 첫째, 처음에는 YAMNet에서 일반적으로 사용하는 clip-level averaging으로 시작했습니다. 이 방법은 음원 전체에서 지배적인 소리를 요약하는 clip tagging에는 적합합니다. 하지만 저희의 목표는 짧은 이벤트 감지이므로 유리 충격음이 주변 normal 프레임에 희석되었습니다. 따라서 frame-level 증거를 유지하도록 바꿨습니다. 둘째, 광범위한 AudioSet 라벨의 노이즈 때문에 검증된 glass와 shatter 소스만 유지했습니다. 셋째, 높은 오탐률의 ablation은 classifier-only 성능이 아닙니다. Fusion을 전제로 설계한 low-threshold shortcut은 그대로 두고 필수 YAMNet gate만 제거한 의도적으로 불안전한 변형입니다. 오탐이 급증한 결과는 완화 경로가 두 신호의 동의가 있을 때만 안전하다는 것을 보여줍니다.

## 슬라이드 8 — 실시간 Fusion (0:50)

YAMNet의 각 forward pass에서는 시스템이 사용하는 두 가지 출력이 나옵니다. Embedding은 custom head에 입력되고, 521개 AudioSet score는 의미적 보조 근거로 사용됩니다. 모든 정제 음원, 오프라인 평가, 파일 예측, 실시간 기본값을 3초 window로 통일했고 실시간 hop은 1초입니다. 현재 Windows PC에서 30개 파일을 측정한 결과 YAMNet, H5 classifier, Fusion 추론은 평균 57ms, p95 61ms였습니다. Raspberry Pi에서 이벤트 입력부터 경보 출력까지의 전체 지연은 평균 [0000]ms, p95 [0000]ms였습니다. 대괄호 값은 임시 입력란이므로 발표 전에 오늘 저녁 측정값으로 반드시 교체해야 합니다.

## 슬라이드 9 — 향후 개선 가능성 (0:50)

데이터 측면에서는 실제 배포 공간의 비명과 충격음, 말소리, 발걸음 같은 hard negative가 더 필요합니다. 모델 측면에서는 일부 YAMNet 계층 fine-tuning, calibration, 소형 temporal network를 검토할 수 있습니다. 배포 측면에서는 현재 TensorFlow Hub와 H5를 사용하고 TFLite는 classifier head만 포함합니다. 새로운 외부 데이터 검증, Raspberry Pi 지속 부하 조건의 지연 검증, 장시간 시간당 오탐 측정이 필요합니다.

## 슬라이드 10 — 결론 (0:40)

전이학습은 감지기를 실용적으로 만들었고, custom head는 YAMNet 특징을 프로젝트 목적에 맞게 전문화했습니다. Frame-aligned Fusion은 과제 특화 정보와 범용 의미 정보를 결합합니다. 최종 Fusion 시스템은 엄격한 라벨 test에서 Macro F1 0.837을 기록했습니다. 다음 단계는 새로운 외부 데이터에서 결과를 검증하고 장시간 현장 시험을 완료하는 것입니다. 감사합니다. 질문 받겠습니다.
