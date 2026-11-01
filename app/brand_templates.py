"""Jinja environment whose templates take the marks and head icons from config/brand.toml (see brand_config.fill)."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.brand_config import fill


class _BrandLoader(FileSystemLoader):
    def get_source(self, environment, template):
        source, filename, uptodate = super().get_source(environment, template)
        return fill(source), filename, uptodate


def make_env(templates_dir: Path) -> Environment:
    return Environment(loader=_BrandLoader(str(templates_dir)), autoescape=select_autoescape(["html"]))
