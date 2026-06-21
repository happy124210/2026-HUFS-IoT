# Edge AI 위협음 감지 — 예상 Q&A 한국어 해설

## 1. 베이스라인 모델은 무엇인가요?

Custom classifier를 제거하고 YAMNet의 범용 AudioSet score를 고정 규칙으로 glass, scream, normal에 직접 매핑한 YAMNet-only 비교 방식입니다. 과제 전용 classifier가 필요한 이유를 확인하기 위한 기준입니다.

## 2. YAMNet만으로 부족한 이유는 무엇인가요?

YAMNet은 우리 환경의 세 클래스만 구분하도록 학습된 모델이 아니라 521개의 범용 소리를 인식하도록 학습된 모델입니다. 의미적 단서는 제공하지만 우리 마이크와 배포 환경에 충분히 특화되어 있지 않습니다.

## 3. 521은 무엇인가요?

YAMNet이 구분하는 범용 소리 클래스의 개수입니다. Speech, Screaming, Music, Shatter 등의 확률을 출력합니다. 우리 Fusion 정책은 이 중 목표 이벤트와 관련된 score만 사용합니다.

## 4. 1,024-dimensional embedding은 무엇인가요?

YAMNet이 각 프레임의 음향 특징을 숫자 1,024개로 표현한 벡터입니다. 1,024개의 클래스가 아닙니다. Custom classifier는 이 특징을 입력받아 glass, normal, scream 확률을 출력합니다.

## 5. YAMNet을 두 번 실행하나요?

아닙니다. 한 번의 YAMNet 실행에서 1,024차원 embedding과 521개 AudioSet score가 함께 나옵니다. Embedding은 custom classifier에 사용하고, 관련 AudioSet score는 의미적 보조 확인에 사용합니다.

## 6. Custom classifier의 구조는 무엇인가요?

YAMNet embedding을 입력으로 받는 과제 전용 dense neural network입니다. 1,024차원 입력 뒤에 512, 256, 128 unit의 dense layer와 3-class softmax가 있습니다. 과적합을 줄이기 위해 batch normalization, dropout, L2 regularization을 사용합니다.

## 7. YAMNet을 동결한 이유는 무엇인가요?

YAMNet은 이미 유용한 범용 음향 특징을 학습했습니다. 동결하면 학습 비용과 제한된 데이터에서의 과적합 위험을 줄일 수 있습니다. 데이터가 더 확보되면 일부 layer를 fine-tuning할 수 있습니다.

## 8. 최종 시스템의 이름은 무엇인가요?

전체 시스템은 YAMNet–custom classifier Fusion system입니다. Frame-aligned Fusion은 별도 신경망의 이름이 아니라 최종 판정 정책입니다.

## 9. Frame-aligned는 무슨 뜻인가요?

Custom classifier와 YAMNet이 같은 시간 프레임에서 같은 이벤트를 지지해야 한다는 의미입니다. 서로 다른 시점의 증거를 억지로 결합하지 않습니다.

## 10. 최종 판정 정책은 어떻게 동작하나요?

두 경로가 있습니다. 강한 classifier 결과가 여러 프레임 동안 지속되면 직접 감지합니다. 더 민감한 완화 경로는 classifier threshold를 낮추는 대신 같은 프레임의 관련 YAMNet score가 함께 동의해야 합니다.

## 11. 모든 감지에 YAMNet 동의를 요구하지 않는 이유는 무엇인가요?

범용 YAMNet score는 과제 전용 classifier가 올바르게 감지한 목표음도 놓칠 수 있습니다. 모든 경우에 동의를 요구하면 recall이 감소합니다. 따라서 강하고 지속적인 classifier 증거는 단독으로 허용하고, 약한 증거에만 YAMNet 확인을 요구합니다.

## 12. 정책이란 무엇인가요?

신경망이 확률을 출력한 뒤 최종 라벨과 경보 여부를 정하는 규칙입니다. 클래스별 threshold, 연속 프레임 수, 두 신호를 결합하는 조건이 정책에 포함됩니다.

## 13. Gating check의 목적은 무엇인가요?

완화된 classifier threshold를 YAMNet 확인 없이 사용했을 때 normal 오탐이 많이 발생하는지 확인한 실험입니다. 결과적으로 완화 경로에는 두 신호의 동의가 필요하다는 것을 확인했습니다. Custom classifier 전체가 사용할 수 없다는 의미는 아닙니다.

## 14. Mean pooling을 사용하지 않은 이유는 무엇인가요?

Mean pooling은 클립 전체의 지배적인 소리를 요약하는 clip tagging에 적합합니다. 하지만 유리 파손음은 짧기 때문에 모든 프레임을 평균하면 증거가 주변 normal 구간에 희석될 수 있습니다. 그래서 최종 판정에는 frame-level 정보를 유지했습니다.

## 15. 데이터는 어디서 얻었나요?

AudioSet의 라벨된 YouTube 오디오로 다양한 음향 환경을 확보하고, Raspberry Pi 마이크 직접 녹음으로 실제 배포 환경과 실패 사례를 반영했습니다.

## 16. 어떤 오디오 증강을 사용했나요?

Gain 변화, noise 추가, filtering, 여러 SNR과 시간 위치에서 threat event를 normal background와 혼합하는 방법을 사용했습니다. Pitch shift와 time stretching은 최종 기본 증강 실행에 포함됐다는 근거가 없어 최종 설명에서 제외했습니다.

## 17. 데이터 누수는 어떻게 방지했나요?

생성된 개별 클립이 아니라 원본 source 기준으로 분리했습니다. 중복 파일, 같은 YouTube source의 클립, 동일 원본에서 생성된 증강본을 같은 group에 유지해 train, validation, test를 넘나들지 않게 했습니다.

## 18. Precision, Recall, F1은 무엇인가요?

Precision은 위협이라고 감지한 것 중 실제 위협의 비율입니다. Recall은 실제 위협 중 감지한 비율입니다. F1은 두 지표의 조화평균이므로 오탐과 미탐이 모두 적어야 높아집니다.

## 19. Macro F1은 무엇인가요?

Glass, normal, scream의 F1을 각각 구한 뒤 동일한 비중으로 평균한 값입니다. 데이터가 많은 normal 성능이 낮은 위협음 성능을 가리지 못하게 합니다.

## 20. Macro F1 0.837은 무슨 의미인가요?

세 클래스별 F1의 평균이 0.837이라는 의미입니다. 모든 클래스의 정확도가 각각 83.7%라는 뜻은 아닙니다.

## 21. 결과의 가장 중요한 한계는 무엇인가요?

최종 정책 선택에는 validation을 사용했지만 현재 test split은 이전 프로젝트 반복에서 확인된 적이 있습니다. 따라서 완전히 새로운 외부 데이터에 대한 일반화가 증명된 것은 아니며, 사후 engineering 평가로 해석해야 합니다.

## 22. AI는 PC와 Raspberry Pi 중 어디에서 실행되나요?

AI 추론은 Raspberry Pi에서 실행됩니다. PC의 PuTTY는 Raspberry Pi 프로세스를 원격으로 실행하고 확인하는 terminal일 뿐 추론을 수행하지 않습니다.

## 23. Raspberry Pi 지연 측정에는 무엇이 포함되나요?

WAV 파일 한 번 로드, YAMNet 실행, H5 custom classifier 실행, Fusion 판정을 포함한 model-processing 시간입니다. 측정 평균은 약 287ms였습니다.

## 24. 287ms가 이벤트부터 경보까지의 전체 지연인가요?

아닙니다. 측정된 model-processing 범위의 시간입니다. 실시간 시스템은 rolling window와 hop interval을 사용하므로 이벤트가 window 내부에서 발생한 시점도 전체 반응 시간에 영향을 줍니다.

## 25. Edge inference가 중요한 이유는 무엇인가요?

원본 오디오를 계속 서버로 전송하지 않아도 되고, 네트워크 연결에 대한 의존성을 줄이며, 로컬에서 즉시 물리적 경보를 실행할 수 있습니다.

## 26. 위협을 감지하면 무엇을 하나요?

부저와 LED를 작동하고, 카메라 사진을 촬영하고, 이벤트 로그를 기록하며, Telegram과 이메일 알림을 전송할 수 있습니다.

## 27. 다음 개선 과제는 무엇인가요?

새로운 외부 데이터 검증, 실제 마이크 위협음과 hard negative 추가, 장시간 시간당 오탐 측정, 지속 부하 Raspberry Pi 시험이 우선입니다. Calibration, 일부 YAMNet fine-tuning, 소형 temporal network도 검토할 수 있습니다.

## 28. 최종 데이터셋 규모는 얼마인가요?

최종 manifest에는 독립적인 source group이 총 1,053개 있습니다. Train 732개, validation 156개, test 165개입니다. Source cap을 적용한 증강으로 학습용 오디오 4,458개가 추가되어 최종 학습 입력은 총 5,190개였습니다. 클래스별로 glass 1,808개, normal 1,732개, scream 1,650개이며 여기서 12,919개의 YAMNet embedding frame을 선택해 학습했습니다.

## 전체 시스템 한 문장 요약

YAMNet을 한 번 실행해 1,024차원 embedding과 521개 AudioSet score를 얻고, custom classifier가 embedding을 세 클래스에 맞게 전문화하며, 최종 정책이 강하고 지속적인 classifier 증거와 같은 프레임의 YAMNet 보조 확인 경로를 결합하는 Raspberry Pi 기반 위협음 감지 시스템입니다.
