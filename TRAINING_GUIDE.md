# Handspeak model training guide

## What the current model recognises

The in-browser recogniser is a **static, single-hand fingerspelling classifier**. It uses MediaPipe hand landmarks and a nearest-neighbour model. It can be trained on A–Z hand poses and a space pose.

The camera can now track and draw **two hands at once**. However, the current classifier does not yet understand two-hand relationships, motion sequences, facial expressions, or ASL grammar. When two hands are visible, the UI reports both hand candidates but deliberately does not append guessed letters to the sentence. This avoids presenting unrelated hand predictions as a valid two-hand sign.

## Train the alphabet from an image dataset

Use a labelled ASL alphabet image dataset with folders named `A` through `Z` and optionally `space`. Do not mix unrelated gestures or words into the alphabet classes.

1. Install Python dependencies:

   ```bash
   pip install opencv-python mediapipe
   ```

2. Download the MediaPipe hand landmark model into the repository root as `hand_landmarker.task`:

   ```powershell
   Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" -OutFile "hand_landmarker.task"
   ```

3. Run the feature extraction script, passing the directory that directly contains the class folders:

   ```bash
   python build_dataset.py "C:\\datasets\\asl_alphabet_train" 400
   ```

4. The script creates `handspeak-samples.json`. Open Handspeak, switch to **Signs → text**, expand **Train on your own hand**, and import that JSON file. The imported samples are stored in this browser.

## Improve the quality of your own camera samples

- Record at least 25 samples for each class; 100–300 varied samples per letter is a better starting point.
- Record in more than one session, with different lighting, camera distances, hand rotations, and backgrounds.
- Keep the whole hand in frame. Avoid sleeves or objects hiding the fingers.
- Record each class separately and check the sample counts before testing.
- Test with a different person or recording session than the one used for training.
- If the model confuses two letters, collect more examples for both letters rather than only increasing the total dataset size.
- Export your browser-collected samples periodically so they are not lost when browser storage is cleared.

## Important accuracy boundary

The bundled sample file currently contains a custom-word class rather than a balanced A–Z training set. The page now ignores bundled custom-word samples by default. Train or import actual alphabet classes before expecting the sample-based classifier to replace the built-in pose rules.

A static image dataset cannot train a full sign-language translator. A future two-hand sign model needs paired-hand landmarks and relative positions; a continuous sign model additionally needs timestamped landmark sequences and labelled movement clips. Facial expression and non-manual grammar also matter. Do not describe the current prototype as a complete ASL translator.
