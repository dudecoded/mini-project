"""Turn a folder-per-letter image dataset (e.g. Kaggle "ASL Alphabet") into handspeak-samples.json.

Usage:
    pip install opencv-python mediapipe
    curl.exe -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    python build_dataset.py C:\\path\\to\\asl_alphabet_train 200      (the number is optional; do not type brackets)

Then open Handspeak > Signs -> text > "Train on your own hand" > Import samples.
Folders named A-Z are used, 'space' becomes SPACE; 'del' and 'nothing' are skipped.
This pipeline is for static, single-hand ASL fingerspelling poses only. It is not a
continuous-sign or two-hand ASL sentence model. Use diverse images and balanced
class counts; validate with a separate held-out set before relying on predictions.
The feature function must match feat() in index.html.
"""
import json, math, random, sys, pathlib


def feat(lm, W, H):
    P = [(p.x * W, p.y * H) for p in lm]
    s = math.hypot(P[9][0] - P[0][0], P[9][1] - P[0][1]) or 1
    ux, uy = (P[9][0] - P[0][0]) / s, (P[9][1] - P[0][1]) / s
    F = []
    for x, y in P:
        vx, vy = (x - P[0][0]) / s, (y - P[0][1]) / s
        F.append((-vx * uy + vy * ux, vx * ux + vy * uy))
    flip = -1 if F[17][0] - F[5][0] < 0 else 1
    return [round(c, 3) for k in range(1, 21) for c in (F[k][0] * flip, F[k][1])]


def main():
    if len(sys.argv) < 2 or not pathlib.Path(sys.argv[1]).is_dir():
        sys.exit("Give the dataset folder (the one that directly contains the A, B, C... folders), e.g.\n"
                 "  python build_dataset.py C:\\data\\asl_alphabet_train 200")
    if not pathlib.Path("hand_landmarker.task").exists():
        sys.exit("hand_landmarker.task not found in this folder. Download it (see the top of this file).")
    digits = "".join(c for c in sys.argv[2] if c.isdigit()) if len(sys.argv) > 2 else ""
    per = int(digits) if digits else 200

    import cv2, mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision

    root = pathlib.Path(sys.argv[1])
    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=1, min_hand_detection_confidence=0.5))
    out, rng = {}, random.Random(0)
    for d in sorted(root.iterdir()):
        name = {"space": "SPACE"}.get(d.name.lower(), d.name.upper())
        if not d.is_dir() or not (len(name) == 1 and name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" or name == "SPACE"):
            continue
        files = [p for p in d.iterdir() if p.is_file()]
        rng.shuffle(files)
        rows = []
        for f in files:
            if len(rows) >= per:
                break
            img = cv2.imread(str(f))
            if img is None:
                continue
            h, w = img.shape[:2]
            if min(h, w) < 80:
                continue
            # Ignore severely blurred samples; they tend to produce unstable landmarks.
            if cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 12:
                continue
            result = det.detect(mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if result.hand_landmarks:
                rows.append(feat(result.hand_landmarks[0], w, h))
        out[name] = rows
        print(f"{name}: {len(rows)} usable samples")
    det.close()
    out = {label: rows for label, rows in out.items() if rows}
    missing = [label for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if len(out.get(label, [])) < 25]
    if missing:
        print("Warning: fewer than 25 usable samples for:", ", ".join(missing))
    json.dump(out, open("handspeak-samples.json", "w"))
    print("Saved handspeak-samples.json. Import it in Handspeak > Signs → text > Train on your own hand.")


if __name__ == "__main__":
    main()
