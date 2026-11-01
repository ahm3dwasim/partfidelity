"""Calibrate once, then evaluate labelled audit edits with saved re-find masks."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from app.image_match import compare
from app.sam import SamMask, SamStillResult
from app.settings import ImageMatchSettings, load_settings


class SavedMask:
    def __init__(self, path: Path):
        self.path = path

    def segment_still(self, image_path: Path, phrase: str, *, include_confidence: bool = True) -> SamStillResult:
        if not self.path.is_file():
            return SamStillResult(())
        return SamStillResult((SamMask((0, 0, 0, 0), None, self.path.read_bytes()),))


def read_cases(folders: list[Path], filename: str) -> list[dict]:
    cases = []
    for folder in folders:
        with (folder / filename).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("label") not in ("altered", "faithful") or not row.get("clip") or not row.get("phrase"):
                    raise ValueError("each case needs a clip, phrase and owner label")
                paths = {}
                root = folder.resolve()
                for name in ("source", "mask", "candidate", "refound"):
                    if not row.get(name):
                        raise ValueError(f"missing {name}")
                    relative = Path(row[name])
                    path = (root / relative).resolve()
                    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
                        raise ValueError(f"{name} must stay under its folder")
                    paths[name] = str(path)
                row["_paths"] = paths
                cases.append(row)
    return cases


def split(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    clips = sorted({case["clip"] for case in cases})
    if len(clips) < 2:
        raise ValueError("at least two source clips are required")
    calibration = set(clips[:len(clips) // 2])
    return ([case for case in cases if case["clip"] in calibration],
            [case for case in cases if case["clip"] not in calibration])


def measure(case: dict, cfg: ImageMatchSettings):
    paths = case["_paths"]
    return compare(Path(paths["source"]), Path(paths["mask"]), Path(paths["candidate"]),
                   case["phrase"], SavedMask(Path(paths["refound"])), cfg)


def calibrate(cases: list[dict], cfg: ImageMatchSettings) -> float:
    rows = [(case["label"], measure(case, cfg).ssim) for case in cases]
    if {label for label, score in rows if score is not None} != {"altered", "faithful"}:
        raise ValueError("calibration needs measurable altered and faithful cases")
    scores = sorted({0.0, 1.0, *(score for _, score in rows if score is not None and 0 <= score <= 1)})
    # Tie-break towards the stricter cutoff; thresholds are the observed scores.
    def rank(cutoff: float):
        altered = [score for label, score in rows if label == "altered" and score is not None]
        faithful = [score for label, score in rows if label == "faithful" and score is not None]
        false_accept = sum(score >= cutoff for score in altered) / len(altered)
        false_reject = sum(score < cutoff for score in faithful) / len(faithful)
        return (abs(false_accept - false_reject), false_accept + false_reject, -cutoff)
    return min(scores, key=rank)


def fingerprint(cases: list[dict]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(json.dumps({k: v for k, v in case.items() if k != "_paths"}, sort_keys=True).encode())
        for name in ("source", "mask", "candidate", "refound"):
            try:
                data = Path(case["_paths"][name]).read_bytes()
            except OSError:
                digest.update(b"missing")
            else:
                digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "evaluate"))
    parser.add_argument("frozen", type=Path)
    parser.add_argument("folders", type=Path, nargs="+")
    args = parser.parse_args()
    current = load_settings().image_match
    if args.mode == "evaluate":
        frozen = json.loads(args.frozen.read_text(encoding="utf-8"))
    cases = read_cases(args.folders, current.cases_filename)
    calibration, heldout = split(cases)
    dataset_sha256 = fingerprint(cases)
    if args.mode == "calibrate":
        cfg = replace(current, ssim_cutoff=calibrate(calibration, current))
        if dataset_sha256 != fingerprint(cases):
            raise ValueError("labelled dataset changed during calibration")
        with args.frozen.open("x", encoding="utf-8") as stream:
            json.dump({"settings": asdict(cfg), "dataset_sha256": dataset_sha256, "calibration_clips": sorted({c["clip"] for c in calibration}),
                       "heldout_clips": sorted({c["clip"] for c in heldout})}, stream, indent=2)
        print(f"Frozen SSIM cutoff: {cfg.ssim_cutoff}")
        return
    if frozen["dataset_sha256"] != dataset_sha256:
        raise ValueError("labelled dataset differs from frozen calibration")
    if (frozen["calibration_clips"] != sorted({c["clip"] for c in calibration}) or
            frozen["heldout_clips"] != sorted({c["clip"] for c in heldout})):
        raise ValueError("clip split differs from frozen calibration")
    cfg = replace(current, ssim_cutoff=calibrate(calibration, current))
    if dataset_sha256 != fingerprint(cases):
        raise ValueError("labelled dataset changed during calibration")
    if asdict(cfg) != frozen["settings"]:
        raise ValueError("settings differ from frozen calibration")
    rows = []
    for part, items in (("calibration", calibration), ("heldout", heldout)):
        for case in items:
            match = measure(case, cfg)
            rows.append({"split": part, "clip": case["clip"], "candidate": case["candidate"],
                         "recipe": case.get("recipe", ""), "label": case["label"],
                         **asdict(match), "verdict": match.verdict})
    if dataset_sha256 != fingerprint(cases):
        raise ValueError("labelled dataset changed during evaluation")
    altered = sum(row["label"] == "altered" and row["split"] == "heldout" for row in rows)
    faithful = sum(row["label"] == "faithful" and row["split"] == "heldout" for row in rows)
    caption = (f"Controlled demonstration — {altered} altered / {faithful} faithful held out; not accuracy or precision."
               if min(altered, faithful) < cfg.small_sample_count else None)
    groups = {}
    for row in rows:
        key = (row["split"], row["recipe"], row["label"])
        group = groups.setdefault(key, {"split": key[0], "recipe": key[1], "label": key[2], "accepted": 0, "rejected": 0})
        group[row["verdict"].lower()] += 1
    print(json.dumps({"ssim_cutoff": cfg.ssim_cutoff, "rows": rows, "caption": caption,
                      "counts": list(groups.values()),
                      "eval_limits": "Hand-labelled edits split by source clip; results describe only these cases."}, indent=2))


if __name__ == "__main__":
    main()
