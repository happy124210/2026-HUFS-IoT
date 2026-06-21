# Edge AI Threat-Sound Detection — English Presentation Script

Target timing: 8 minutes presentation + 2 minutes Q&A.

## Slide 1 — Edge AI Threat-Sound Detection (0:25)

Good afternoon. Our project is an edge AI security system that listens for two threat sounds: breaking glass and screams. It runs on a Raspberry Pi, classifies audio in real time, and triggers physical and remote alerts. I will focus on the baseline, our final Fusion model, what failed, what finally worked, and what should improve next.

## Slide 2 — Why listen for threats? (0:40)

Why use sound at all? Cameras and motion sensors are useful, but they have blind spots. A microphone can detect an event even when the source is outside the camera view. Our design also keeps inference on the edge: raw audio does not need to be continuously uploaded. When a threat is confirmed, the system can activate a buzzer and LED, take a photo, write a log, and send Telegram and email notifications.

## Slide 3 — Why YAMNet alone is not enough (0:50)

Why is YAMNet alone not enough? We remove our custom head and map YAMNet's generic AudioSet scores directly into glass, scream, or normal using fixed rules. This approach was notably weak at detecting screams. The result shows that generic semantic scores alone are not sufficiently specialized for our targets, which motivates the task-specific head on the next slide.

## Slide 4 — Final model: custom head + YAMNet Fusion (1:05)

Our final system uses both outputs from one YAMNet execution. The 1,024-dimensional embeddings enter our task-specific dense classifier, while selected scores from the 521 AudioSet classes provide supporting evidence. The final rule has two paths: sustained classifier evidence can trigger directly, while the more sensitive path requires the classifier and YAMNet to agree on the same frame.

## Slide 5 — Where the data came from—and how we evaluated it (0:45)

We used two complementary data sources. AudioSet supplied labeled YouTube audio with broad acoustic diversity, while direct Raspberry Pi microphone recordings captured the deployment domain and useful failure cases. For training, we varied gain, added noise and filtering, and mixed threat events with normal backgrounds at different signal-to-noise ratios and positions. The table separates independent sources from generated audio: 732 training sources produced 4,458 augmented or mixed files, giving 5,190 training audio files.

## Slide 6 — Baseline vs Final Fusion (0:55)

This table compares the YAMNet-only baseline and our Final Fusion on the same test groups. Final Fusion substantially improved scream recall and increased overall Macro F1. The trade-off was a modest increase in normal false alarms, which is reasonable only if missing a threat is considered more costly. Because this test split had been inspected in earlier project iterations, we still need fresh external validation.

## Slide 7 — Trial and error changed the system (1:00)

Three lessons changed our approach. First, we began with conventional clip-level averaging commonly used with YAMNet. This is appropriate when the goal is to summarize the dominant sound across a clip. Our task, however, is transient-event detection, so short glass impacts were diluted by surrounding normal frames. We therefore retained frame-level evidence. Second, broad AudioSet labels introduced label noise, while live microphone reviews exposed deployment-specific errors. We tightened the labels and fed verified failure cases back into training. Third, a relaxed classifier threshold improved sensitivity but also triggered on many normal sounds when used without YAMNet corroboration. We therefore require both signals to agree on the same frame for this path.

## Slide 8 — Real-time Fusion (0:50)

The live system analyzes a rolling audio window at regular intervals. On Raspberry Pi, model processing averaged about 287 milliseconds. This value measures WAV loading, YAMNet, the H5 classifier, and the Fusion decision together in one run.

## Slide 9 — Where we can improve next (0:50)

There are three practical directions for improvement. For data, we need more real microphone screams and hard negatives such as impacts, speech, and footsteps from the deployment room. For the model, selective YAMNet fine-tuning, calibration, or a compact temporal network could improve context. For deployment, the current runtime is TensorFlow Hub plus H5, while the TFLite file contains only the classifier head. We therefore need fresh external validation, sustained-load Raspberry Pi latency validation, and long-duration false alarms per hour.

## Slide 10 — Conclusion (0:40)

To conclude, transfer learning made the detector practical, our custom head specialized YAMNet features, and frame-aligned Fusion combined task-specific and semantic evidence. The final system achieved the best overall balance in our evaluation. The next step is to validate these results on fresh external data and complete long-duration field testing. Thank you. I am ready for your questions.
