# Edge AI Threat-Sound Detection — English Presentation Script

Target timing: approximately 9 minutes presentation + 1 minute transition or Q&A.

## Slide 1 — Edge AI Threat-Sound Detection (0:20)

Good afternoon. Our project is an edge AI security system that listens for breaking glass and screams. It runs on a Raspberry Pi and connects real-time audio classification to physical and remote alerts.

## Slide 2 — Why listen for threats? (0:30)

Cameras and motion sensors have blind spots. A microphone adds an independent signal, while edge inference avoids continuously uploading raw audio. A confirmed event can immediately trigger local and remote actions.

## Slide 3 — Project idea and objectives (0:30)

The main idea is to use sound as an additional security sensor. Our objectives were reliable three-class detection, local privacy-preserving inference, and direct IoT response. Success therefore means balancing threat recall, false alarms, and Raspberry Pi practicality.

## Slide 4 — End-to-end system design (0:30)

This is the complete system flow. Audio is captured into a rolling window, YAMNet extracts embeddings and semantic scores, the custom head produces three-class probabilities, Fusion makes the event decision, and the IoT layer activates the response.

## Slide 5 — Hardware architecture (0:30)

The Raspberry Pi 4 performs all inference. The microphone provides audio, the camera captures evidence, and the LED and buzzer provide local warning. The PC only uses PuTTY as a remote terminal; it does not run the AI.

## Slide 6 — Software architecture and implementation (0:35)

The implementation is modular. The live loop is in realtime_detect.py, audio preprocessing is separated into audio_pipeline.py, the custom H5 model performs task classification, and detection_policy.py applies the final Fusion rules. Separate modules handle GPIO, camera, Telegram, and email.

## Slide 7 — Baseline: YAMNet-only control (0:30)

Our baseline removes the custom head and maps generic YAMNet AudioSet scores directly to three classes. Its weak scream recall shows why generic semantic scores alone are not sufficiently specialized for our task.

## Slide 8 — Final architecture (0:35)

One YAMNet execution provides two outputs. The 1,024-dimensional embeddings enter the custom dense head, while selected scores from the 521 AudioSet classes provide supporting evidence. The 1,024 values are features, whereas the 521 values are generic class scores.

## Slide 9 — Frame-aligned Fusion policy (0:40)

The policy uses two paths. Strong classifier evidence sustained over consecutive frames can trigger directly. The more sensitive path triggers only when the classifier and the relevant YAMNet score agree on the same frame.

## Slide 10 — Data sources and class definition (0:30)

We used AudioSet for broad acoustic diversity, direct Raspberry Pi microphone recordings for the deployment domain, and realistic normal hard negatives such as speech, music, impacts, and footsteps.

## Slide 11 — Augmentation and final dataset usage (0:40)

Training augmentation varied gain, added noise and filtering, and mixed events with normal backgrounds at different signal-to-noise ratios and positions. The final training input contained 732 source audio files and 4,458 augmented or mixed files, for 5,190 training audio files in total.

## Slide 12 — Training and evaluation methodology (0:40)

We split by original source, froze YAMNet, cached frame-level embeddings, and trained the dense head with regularization and early stopping. Fusion candidates were selected using validation data under minimum recall constraints.

## Slide 13 — Trial and error (0:40)

Three lessons changed the system. Mean pooling diluted short glass impacts, broad labels introduced noise, and a relaxed threshold without YAMNet agreement caused many normal sounds to trigger. Each finding led directly to the final design.

## Slide 14 — Baseline vs Final Fusion (0:40)

Compared with YAMNet-only, Final Fusion substantially improved scream recall and Macro F1. The trade-off was a modest increase in normal false alarms. Fresh external validation is still required because the test split was inspected in earlier iterations.

## Slide 15 — Results and project contributions (0:35)

The main contributions are task specialization, frame-level Fusion, practical edge deployment, and a more defensible evaluation process. Together they connect an audio model to an actual IoT security response.

## Slide 16 — Real-time Fusion on Raspberry Pi (0:35)

The live system uses a three-second rolling window and a one-second hop. On Raspberry Pi, WAV loading, YAMNet, the H5 classifier, and the Fusion decision averaged about 287 milliseconds in the measured benchmark.

## Slide 17 — Future work (0:35)

Future work should prioritize fresh external validation, more microphone-domain threats and hard negatives, calibration or a compact temporal model, sustained-load testing, and long-duration false alarms per hour.

## Slide 18 — Conclusion (0:25)

To conclude, transfer learning made the detector practical, the custom head specialized YAMNet features, and frame-aligned Fusion combined task-specific and semantic evidence. The next required step is validation on fresh external data. Thank you.
