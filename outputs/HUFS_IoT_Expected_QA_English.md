# Edge AI Threat-Sound Detection — Expected Q&A

## 1. What is the baseline model?

The baseline is a YAMNet-only control. It removes our custom classifier and maps YAMNet's generic AudioSet scores directly to glass, scream, or normal using fixed rules. It is useful for showing why a task-specific classifier is needed.

## 2. Why is YAMNet alone insufficient?

YAMNet was trained to recognize 521 general AudioSet classes, not specifically to separate our three classes under our microphone and deployment conditions. Its semantic scores are useful, but they are not sufficiently specialized for reliable threat detection.

## 3. What does the number 521 mean?

YAMNet predicts scores for 521 general sound classes, such as speech, screaming, music, and shatter. These are class probabilities, not learned feature dimensions. Our Fusion policy uses only the scores related to our target events.

## 4. What is a 1,024-dimensional embedding?

It is a vector of 1,024 learned audio features produced by YAMNet for each frame. The values do not correspond to 1,024 named classes. They encode acoustic patterns that our custom classifier can use to distinguish glass, normal, and scream.

## 5. Does the system run YAMNet twice?

No. One YAMNet forward pass produces both the 1,024-dimensional embeddings and the 521 AudioSet scores. The embeddings go to our custom classifier, while selected AudioSet scores provide semantic corroboration.

## 6. What is the custom classifier?

It is a task-specific dense neural network attached to frozen YAMNet embeddings. Its structure is 1,024 input features followed by dense layers of 512, 256, and 128 units, and a three-class softmax output. Batch normalization, dropout, and L2 regularization are used to reduce overfitting.

## 7. Why freeze YAMNet?

YAMNet already provides useful general audio features. Freezing it reduces training cost and the risk of overfitting on our limited dataset. Fine-tuning selected layers is a possible future improvement when more deployment data are available.

## 8. What is the final system called?

The overall system is a YAMNet–custom classifier Fusion system. Frame-aligned Fusion is the decision policy, not the name of a separate neural network.

## 9. What does “frame-aligned” mean?

It means the custom classifier and YAMNet must support the same event in the same time frame. Evidence from unrelated times is not combined. This makes the corroboration temporally meaningful.

## 10. How does the final decision policy work?

There are two paths. Strong classifier evidence sustained over consecutive frames can trigger directly. A more sensitive path uses a relaxed classifier threshold, but it triggers only when the relevant YAMNet score agrees on the same frame.

## 11. Why not require YAMNet agreement for every detection?

YAMNet's generic scores can miss target sounds that our specialized classifier detects correctly. Requiring agreement in every case would reduce recall. Therefore, strong sustained custom-classifier evidence can trigger directly, while weaker evidence requires corroboration.

## 12. What is a decision policy?

A decision policy is the set of rules applied after the neural network produces probabilities. It includes class thresholds, required consecutive frames, and the conditions for combining custom-classifier and YAMNet evidence.

## 13. What was the purpose of the gating check?

We tested the relaxed classifier path without YAMNet corroboration. It produced many false alarms on normal sounds. This showed that the relaxed threshold should be used only when both signals agree; it was not evidence that the entire custom classifier was unusable.

## 14. Why did you stop using mean pooling?

Mean pooling summarizes the dominant sound across an entire clip and is reasonable for clip tagging. Glass break is a short transient event, so averaging all frames can dilute its evidence. We therefore retained frame-level predictions for the final decision.

## 15. Where did the data come from?

We used two complementary sources. AudioSet provided labeled YouTube audio with broad acoustic diversity, and direct Raspberry Pi microphone recordings represented the actual deployment domain and useful failure cases.

## 16. How was the audio augmented?

For training, we varied gain, added noise and filtering, and mixed threat events with normal backgrounds at different signal-to-noise ratios and temporal positions. This simulated noisy and varied deployment conditions. Pitch shift and time stretching were not part of the default final augmentation run.

## 17. How did you prevent data leakage?

We split the data by original source rather than by generated clip. Duplicate files, clips sharing a YouTube source, and augmented versions of the same recording were kept in the same group, so related audio could not cross the train, validation, and test splits.

## 18. What are Precision, Recall, and F1 score?

Precision is the proportion of predicted threats that are real threats. Recall is the proportion of real threats that the system detects. F1 is their harmonic mean, so it becomes high only when both false alarms and missed events are controlled.

## 19. What is Macro F1?

Macro F1 calculates F1 separately for glass, normal, and scream, then gives the three classes equal weight. It is appropriate here because high performance on the larger normal class should not hide weak threat detection.

## 20. What does Macro F1 0.837 mean?

It means the average of the three class-specific F1 scores is 0.837. It does not mean that every class individually achieved 83.7 percent accuracy.

## 21. What is the main limitation of the reported result?

The final policy was selected using validation data, but the current test split had been inspected during earlier project iterations. Therefore, the result is a retrospective engineering evaluation rather than proof of generalization to completely unseen external data.

## 22. Where does the AI inference run?

The AI runs on the Raspberry Pi. PuTTY on the PC only provides a remote terminal for starting and monitoring the Raspberry Pi process; it does not perform the model inference.

## 23. What does the reported Raspberry Pi latency include?

The measured model-processing time includes loading one WAV file, running YAMNet, running the H5 custom classifier, and applying the Fusion decision. The mean was about 287 milliseconds for the measured files.

## 24. Is 287 milliseconds the complete event-to-alert latency?

No. It is the measured model-processing scope. The live system also uses a rolling audio window and a hop interval, so the time at which an event occurs inside the window affects the complete response time.

## 25. Why is edge inference important?

It avoids continuously uploading raw audio, reduces dependence on network connectivity, and allows local physical alerts. Only detected events and associated evidence need to be transmitted remotely.

## 26. What happens after a threat is detected?

The system can activate the buzzer and LED, capture a camera image, write an event log, and send Telegram and email notifications.

## 27. What should be improved next?

The priorities are fresh external validation, more real microphone threats and hard negatives, long-duration false-alarm measurement, and sustained-load Raspberry Pi testing. Model improvements could include calibration, selective YAMNet fine-tuning, or a compact temporal network.

## 28. How large was the final dataset split?

The final manifest contained 1,053 independent source groups: 732 for training, 156 for validation, and 165 for testing. Source-capped augmentation added 4,458 training audio files, so the final training input contained 5,190 audio files: 1,808 glass, 1,732 normal, and 1,650 scream. These produced 12,919 selected YAMNet embedding frames.

## Short summary answer

Our system runs YAMNet once to obtain two outputs: 1,024-dimensional embeddings and 521 AudioSet scores. A custom classifier specializes the embeddings for glass, normal, and scream. The final decision policy combines strong sustained classifier evidence with a frame-aligned corroboration path. This improves the balance between threat recall and false alarms while remaining practical on a Raspberry Pi.
