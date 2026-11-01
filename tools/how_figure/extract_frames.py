"""One-off: cut the real frames the How it works figure shows from the sample clip and score them with the app's
own frame_scoring functions. Run with the repo venv: .venv/bin/python tools/how_figure/extract_frames.py <clip.mp4>
Writes static/ui/media/how/*.webp and prints the TOML rows for config/how_it_works.toml ([[example.candidates]], [[example.strip]]).
The clip is private footage and is not in the repo; the outputs are git-ignored."""
import subprocess
import sys
from pathlib import Path

import cv2

from app import frame_scoring as F

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "static" / "ui" / "media" / "how"
WINDOW = (124.0, 132.0)
STRIP = (3.0, 40.0, 89.0, 102.0, 150.0)


def webp(frame, name, width, quality=72):
    h = round(frame.shape[0] * width / frame.shape[1])
    small = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
    tmp = OUT / (name + ".png")
    cv2.imwrite(str(tmp), small)
    subprocess.run(["cwebp", "-quiet", "-q", str(quality), str(tmp), "-o", str(OUT / (name + ".webp"))], check=True)
    tmp.unlink()


def main(clip):
    OUT.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(clip)
    prev, scored = None, []
    for t in F.sample_times(*WINDOW):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scored.append(F.ScoredFrame(t, F.laplacian_sharpness(gray), 0.0 if prev is None else F.frame_motion(prev, gray)))
        prev = gray
        webp(frame, f"c{t:g}", 192, 60)
    kept = {f.timestamp_s for f in F.pick_best_two(scored)}
    for t in sorted(kept):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        webp(cap.read()[1], f"k{t:g}", 960, 74)
    for t in STRIP:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        webp(cap.read()[1], f"s{t:g}", 192, 60)
    for f in scored:
        print(f'[[example.candidates]]\nt = {f.timestamp_s:g}\nsharpness = {f.sharpness:.1f}\nmotion = {f.motion:.1f}\n')


main(sys.argv[1])
