# Edge AI Threat-Sound Detection — English Presentation Script

Target timing: 8 minutes presentation + 2 minutes Q&A.

## Slide 1 — Edge AI Threat-Sound Detection (0:25)

Good afternoon. Our project is an edge AI security system that listens for two threat sounds: breaking glass and screams. It runs on a Raspberry Pi, classifies audio in real time, and triggers physical and remote alerts. I will focus on the baseline, our final Fusion model, what failed, what finally worked, and what should improve next.

## Slide 2 — Why listen for threats? (0:40)

Why use sound at all? Cameras and motion sensors are useful, but they have blind spots. A microphone can detect an event even when the source is outside the camera view. Our design also keeps inference on the edge: raw audio does not need to be continuously uploaded. When a threat is confirmed, the system can activate a buzzer and LED, take a photo, write a log, and send a Telegram message.

## Slide 3 — Why YAMNet alone is not enough (0:50)

This slide asks one specific question: why not use YAMNet by itself? As a control comparison, we remove our custom head and map YAMNet's 521 generic AudioSet scores directly into glass, scream, or normal using fixed rules. On the strict test set, this produced a Macro F1 of 0.756 and detected only 19 of 38 screams, or 50 percent. The result shows that generic semantic scores alone are not sufficiently specialized for our targets. We use this comparison only to understand the limitation of YAMNet-only detection. It was not used to choose our model or thresholds. This limitation motivates the task-specific head on the next slide.

## Slide 4 — Final model: custom head + YAMNet Fusion (1:05)

Our final system uses both outputs from one YAMNet forward pass. The 1,024-dimensional embeddings enter our task-specific classifier, which uses dense layers of 512, 256, and 128 units followed by a three-class softmax. In parallel, selected YAMNet AudioSet scores provide semantic corroboration. The final rule has two paths: sustained classifier evidence can trigger directly, while a short corroborated path requires the classifier and YAMNet to agree on the same frame.

## Slide 5 — A stricter experiment, not just a larger dataset (0:55)

The largest methodological improvement was the evaluation protocol. We removed 158 broad-label clips, grouped clean and augmented files by original source, merged 325 duplicate aliases, and prevented shared YouTube IDs from crossing splits. For this rerun, we fixed only three candidates before executing the selection program: recall-oriented, balanced, and precision-oriented. Selection used validation only, and the balanced policy achieved the highest Macro F1 while keeping both event recalls above seventy percent.

## Slide 6 — Final Fusion results (0:55)

The final Fusion system provided the best overall balance between threat recall and normal false alarms, reaching a Macro F1 of 0.837. The class-level results show balanced behavior across both threat categories and normal audio. The important limitation is that, although policy selection used validation only, this test split had been inspected in earlier project iterations. We therefore treat this as a retrospective engineering result and still need fresh external validation.

## Slide 7 — Trial and error changed the system (1:00)

Three lessons changed our approach. First, we began with conventional clip-level averaging commonly used with YAMNet. This is appropriate when the goal is to summarize the dominant sound across a clip. Our task, however, is transient-event detection, so short glass impacts were diluted by surrounding normal frames. We therefore retained frame-level evidence. Second, broad AudioSet labels introduced label noise, so we kept only verified glass and shatter sources. Third, the high false-alarm ablation is not the standalone classifier result. We deliberately kept the low-threshold shortcut designed for Fusion and removed only its required YAMNet gate. The sharp increase in false alarms shows that this relaxed path is safe only when both signals agree.

## Slide 8 — Real-time Fusion (0:50)

Each YAMNet forward pass returns two outputs used by our system: the embeddings for our custom head and the 521 AudioSet scores for semantic corroboration. All cleaned clips, offline evaluation, file prediction, and live defaults now use a three-second window; the live hop is one second. On this Windows PC, YAMNet, the H5 classifier, and Fusion averaged 57 milliseconds with a 61 millisecond p95 over thirty files. On the Raspberry Pi, complete event-to-alert latency was [0000] milliseconds on average, with a [0000] millisecond p95. The bracketed values are temporary fields and must be replaced with tonight's measurement before presenting.

## Slide 9 — Where we can improve next (0:50)

There are three practical directions for improvement. For data, we need more real microphone screams and hard negatives such as impacts, speech, and footsteps from the deployment room. For the model, selective YAMNet fine-tuning, calibration, or a compact temporal network could improve context. For deployment, the current runtime is TensorFlow Hub plus H5, while the TFLite file contains only the classifier head. We therefore need fresh external validation, sustained-load Raspberry Pi latency validation, and long-duration false alarms per hour.

## Slide 10 — Conclusion (0:40)

To conclude, transfer learning made the detector practical, our custom head specialized YAMNet features, and frame-aligned Fusion combined task-specific and semantic evidence. The final Fusion system reached a Macro F1 of 0.837 on our strict-label test. The next step is to validate these results on fresh external data and complete long-duration field testing. Thank you. I am ready for your questions.
