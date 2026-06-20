# Edge AI Threat-Sound Detection — English Presentation Script

Target timing: 8 minutes presentation + 2 minutes Q&A.

## Slide 1 — Edge AI Threat-Sound Detection (0:25)

Good afternoon. Our project is an edge AI security system that listens for two threat sounds: breaking glass and screams. It runs on a Raspberry Pi, classifies audio in real time, and triggers physical and remote alerts. I will focus on the baseline, our model, what failed, what finally worked, and what should improve next.

## Slide 2 — Why listen for threats? (0:40)

Why use sound at all? Cameras and motion sensors are useful, but they have blind spots. A microphone can detect an event even when the source is outside the camera view. Our design also keeps inference on the edge: raw audio does not need to be continuously uploaded. When a threat is confirmed, the system can immediately activate a buzzer and LED, take a photo, write a log, and send a Telegram message.

## Slide 3 — Reference baseline: YAMNet-only ablation (0:50)

Our architectural reference is a YAMNet-only detector. Audio is resampled to sixteen kilohertz and passed through Google’s pretrained AudioSet network. YAMNet outputs probabilities for 521 generic sound classes, and fixed rules map selected scores into glass, scream, or normal. In a post-hoc component ablation on the strict test set, Macro F1 was 0.756, glass recall was 20 out of 29, and scream recall was 19 out of 38. This comparison is exploratory and was not used to select the deployed model or its thresholds.

## Slide 4 — Retained model: YAMNet embeddings + custom head (1:05)

Our retained model keeps YAMNet as a frozen feature extractor but adds a task-specific classifier. Every audio frame becomes a 1,024-dimensional embedding. The classifier uses dense layers of 512, 256, and 128 units, followed by a three-class softmax. Batch normalization, dropout, and L2 regularization reduce overfitting. The official policy was selected on validation and uses classifier-only thresholds of 0.95 for glass and 0.40 for scream, with one and three consecutive frames respectively.

## Slide 5 — A stricter experiment, not just a larger dataset (0:50)

The largest methodological improvement was the evaluation protocol. We removed 158 clips collected from broad labels such as smash, breaking, and crushing, because many were not actually glass. We also grouped every clean and augmented file by its original source, merged 325 duplicate aliases, and prevented shared YouTube IDs from crossing splits. Candidates and thresholds were selected only on validation. The test set contains 165 independent source groups and was evaluated once after selection.

## Slide 6 — Official test results (0:50)

These are the official strict-label test results for the retained model. It correctly classified 136 of 165 source groups, for 82.4 percent accuracy and 0.801 Macro F1. Glass recall was 24 of 29, and scream recall was 27 of 38. On 98 normal groups, 13 produced a false alarm. These counts are important because the class sample sizes are modest. No new candidate passed the validation promotion rule, which required higher Macro F1 than the retained model, at most five percent normal false alarms, and at least seventy percent recall for both events.

## Slide 7 — Trial and error changed the evaluation (1:00)

Three failures changed our approach. First, a diagnostic example showed that mean pooling could hide short glass impacts inside longer windows. That example motivated frame-level decisions, but it is not an official accuracy result. Second, broad AudioSet labels introduced label noise, so we retained only verified glass and shatter sources. Third, Candidate C reduced validation false alarms to 3.2 percent, but scream recall dropped to 60.5 percent. Because the promotion rule required at least 70 percent event recall and higher Macro F1, we rejected it.

## Slide 8 — Official vs. original demo (0:45)

The official and original demo policies optimize different risks. The official classifier-only policy achieved Macro F1 0.801, normal false alarms of 13.3 percent, glass recall of 82.8 percent, and scream recall of 71.1 percent. The original demo used higher thresholds, YAMNet corroboration, scream mean and margin filters, and an RMS gate in the live microphone loop. The historical offline ablation did not include that RMS gate. In that ablation, Macro F1 was 0.804, false alarms fell to 7.1 percent, glass recall stayed at 82.8 percent, and scream recall fell to 57.9 percent. This comparison was not used for model selection.

## Slide 9 — Where we can improve next (0:55)

There are three practical directions for improvement. For data, we need more real microphone screams and hard negatives such as impacts, speech, and footsteps from the deployment room. For the model, selective YAMNet fine-tuning, better calibration, or a compact temporal network could improve context. For deployment, the current runtime is TensorFlow Hub plus H5; the TFLite file contains only the classifier head. We therefore need an end-to-end Raspberry Pi benchmark and long-duration monitoring that reports false alarms per hour.

## Slide 10 — Conclusion (0:40)

To conclude, transfer learning made the detector practical, while source-group splitting and explicit validation gates made the evaluation more defensible. Comparing the official and original demo policies exposes the real trade-off: fewer false alarms versus more missed screams. The next engineering goal is to reduce false alarms without sacrificing threat recall. Thank you. I am ready for your questions.
