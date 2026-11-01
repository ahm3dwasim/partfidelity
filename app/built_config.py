"""Load the public build notes and their generated feature index."""
from pathlib import Path
import re
import tomllib
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "built.toml"
FEATURES = ROOT / "docs" / "FEATURES.md"
PAGE_KEYS = frozenset("title kicker intro summary_heading window_heading window repository_heading repository_url repository_pending features_heading features_intro replay_link_text history_summary history_intro current_features replaced_label history_label tools_heading input_heading input licences_heading footer".split())


def load(path: Path = CONFIG, features: Path = FEATURES) -> dict:
    try:
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
        source = features.read_text(encoding="utf-8")
        if set(cfg) != {"page", "summary", "tools", "licences", "superseded", "retired_names"} or set(cfg["page"]) != PAGE_KEYS:
            raise ValueError("unexpected build notes keys")
        featured = cfg["page"]["current_features"]
        if not isinstance(featured, list) or not 6 <= len(featured) <= 8 or len(set(featured)) != len(featured) or any(not isinstance(n, str) for n in featured):
            raise ValueError("current_features must list 6 to 8 distinct feature names")
        for key, value in cfg["page"].items():
            if key == "current_features":
                continue
            if not isinstance(value, str) or (key != "repository_url" and not value.strip()):
                raise ValueError(f"invalid page.{key}")
        url = cfg["page"]["repository_url"]
        if url and (urlsplit(url).scheme != "https" or not urlsplit(url).netloc or urlsplit(url).username or urlsplit(url).password):
            raise ValueError("repository URL must be a public HTTPS URL")
        for group in ("summary", "tools", "licences"):
            if not isinstance(cfg[group], list) or not cfg[group]:
                raise ValueError(f"{group} must be non-empty")
            for item in cfg[group]:
                if set(item) != {"name", "text"} or any(not isinstance(v, str) or not v.strip() for v in item.values()):
                    raise ValueError(f"invalid {group} entry")
        superseded = cfg["superseded"]
        retired = cfg["retired_names"]
        if not isinstance(superseded, dict) or set(retired) != {"fonts", "icons"}:
            raise ValueError("invalid superseded map or retired names")
        if any(not isinstance(v, str) for v in superseded.values()) or any(not isinstance(v, list) for v in retired.values()):
            raise ValueError("invalid superseded map or retired names")
        sections = re.split(r"^## (.+)$", source, flags=re.M)[1:]
        if not sections:
            raise ValueError("feature index is empty")
        names = sections[::2]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f"duplicate feature names: {', '.join(duplicated)}")
        for old, new in superseded.items():
            if old not in names or new not in names or new in superseded or old == new:
                raise ValueError(f"superseded map names an unknown or retired feature: {old}")
        for name in cfg["page"]["current_features"]:
            if name not in names or name in superseded:
                raise ValueError(f"current_features names an unknown or replaced feature: {name}")
        current = (ROOT / "static/ui/tokens.css").read_text(encoding="utf-8") + (ROOT / "static/ui/icon.svg").read_text(encoding="utf-8")
        rows = []
        for name, body in zip(names, sections[1::2]):
            match = re.search(r"^- What it does: (.+)$", body, re.M)
            description = match.group(1) if match else body.strip().split("\n\n", 1)[0].replace("\n", " ")
            if not description or description.startswith("- "):
                raise ValueError("feature index has missing descriptions")
            if name in superseded:
                rows.append({"name": name, "text": f"{cfg['page']['replaced_label']} {superseded[name]}.", "history": True})
                continue
            for font in retired["fonts"]:
                if font not in current and re.search(r"\b" + re.escape(font) + r"\b", body):
                    raise ValueError(f"{name} names retired font {font}; mark it superseded")
            for icon in retired["icons"]:
                if icon not in current and icon.lower() in body.lower():
                    raise ValueError(f"{name} names retired icon {icon}; mark it superseded")
            rows.append({"name": name, "text": re.split(r"(?<=\.)\s+(?=[`*_\[]*[A-Za-z])", description, maxsplit=1)[0].replace("`", ""), "history": False, "current": name in cfg["page"]["current_features"]})
        cfg["features"] = rows
        return cfg
    except (OSError, tomllib.TOMLDecodeError, ValueError, TypeError, KeyError) as error:
        raise ValueError(f"Invalid build notes: {error}") from error
