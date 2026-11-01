"""Validated destinations for the recorded sample's proof links."""
from pathlib import Path
import re
import tomllib

KEYS = {"sample_calls", "hard_stop", "catch_cta", "meta_roles", "meta_roles_link", "meta_roles_catch_link", "meta_roles_built_link"}
# Each role is a distinct phrase that must stand alone: not followed by a word character or ".digit", so
# "SAM 3" is not satisfied by "SAM 3.1" and "Muse Spark" is not satisfied by "Muse Spark Open".
ROLE_PATTERNS = tuple(re.compile(r"(?<![\w.])" + re.escape(t) + r"(?![\w]|\.\d)") for t in (
    "Muse Spark steps", "SAM 3.1 part boxes", "SAM 3 contours", "Muse Spark Open", "Muse Image"))


def load():
    with (Path(__file__).resolve().parents[1] / "config/proof_links.toml").open("rb") as source:
        data = tomllib.load(source)
    if set(data) != KEYS or any(not isinstance(v, str) or not v.strip() for v in data.values()):
        raise ValueError("invalid sample proof link configuration")
    if not re.fullmatch(r"/calls/[0-9a-f]{32}", data["sample_calls"]) or not re.fullmatch(r"/guide/sample#step-[1-9][0-9]*", data["hard_stop"]):
        raise ValueError("invalid sample proof link configuration")
    if not all(p.search(data["meta_roles"]) for p in ROLE_PATTERNS) or "verified" in data["meta_roles"].lower():
        raise ValueError("invalid sample proof link configuration: meta_roles must name every model and never say verified")
    return data
