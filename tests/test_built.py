from pathlib import Path
import re

import pytest
from fastapi.testclient import TestClient

from app import built_config
from app.routes.site import create_site_app


def _index_names():
    source = (built_config.ROOT / 'docs/FEATURES.md').read_text(encoding='utf-8')
    return re.findall(r'^## (.+)$', source, flags=re.M)


def test_built_contains_every_feature_and_release_notes():
    client = TestClient(create_site_app())
    page = client.get('/built')
    assert page.status_code == 200
    from html import escape
    names = _index_names()
    assert len(names) > 3 and len(set(names)) == len(names)
    for name in names:
        assert page.text.count('<dt>' + escape(name) + '</dt>') == 1, name
    for old, new in built_config.load()['superseded'].items():
        assert 'Replaced by ' + escape(new) in page.text and old in names
    assert 'Repository link added at release' in page.text
    assert 'live app makes no Muse Image call' in page.text
    assert 'made offline and frozen as a test set for the edit check' in page.text
    assert 'Oct 26–Nov 4, 2026' in page.text
    assert '<a href="/built">How this was built</a>' in client.get('/').text


def _featured(cfg: str, *names: str) -> str:
    start = cfg.index('current_features = [')
    return cfg[:start] + 'current_features = ' + str(list(names)).replace("'", '"') + cfg[cfg.index(']\n', start) + 1:]


FILLER = ['Filler one', 'Filler two', 'Filler three', 'Filler four', 'Filler five']
FILLER_MD = ''.join(f'## {n}\n- What it does: Filler {n}.\n\n' for n in FILLER)


def _no_superseded(tmp_path: Path, *names: str) -> Path:
    cfg = built_config.CONFIG.read_text()
    config = tmp_path / 'plain.toml'
    config.write_text(_featured(cfg[:cfg.index('[superseded]')] + '[superseded]\n\n' + cfg[cfg.index('[retired_names]'):], *names))
    return config


def test_loader_rejects_missing_and_invalid_content(tmp_path: Path):
    with pytest.raises(ValueError, match='Invalid build notes'):
        built_config.load(tmp_path / 'missing.toml')
    broken = tmp_path / 'broken.toml'
    broken.write_text(built_config.CONFIG.read_text().replace('repository_url = ""', 'repository_url = "javascript:alert(1)"'))
    with pytest.raises(ValueError, match='repository URL'):
        built_config.load(broken)
    missing_feature = tmp_path / 'features.md'
    missing_feature.write_text('## Missing feature\n\n' + FILLER_MD)
    with pytest.raises(ValueError, match='missing descriptions'):
        built_config.load(_no_superseded(tmp_path, 'Missing feature', *FILLER), missing_feature)


def _config_with(tmp_path: Path, old: str, new: str) -> Path:
    config = tmp_path / 'built.toml'
    config.write_text(built_config.CONFIG.read_text().replace(old, new))
    return config


def test_superseded_map_names_every_feature_with_a_retired_font_or_icon():
    cfg = built_config.load()
    tokens = (built_config.ROOT / 'static/ui/tokens.css').read_text()
    icons = (built_config.ROOT / 'static/ui/icon.svg').read_text()
    assert 'David Libre' in tokens and 'Roboto' in tokens and 'pf-mark' in icons
    source = (built_config.ROOT / 'docs/FEATURES.md').read_text(encoding='utf-8')
    sections = re.split(r'^## (.+)$', source, flags=re.M)[1:]
    for name, body in zip(sections[::2], sections[1::2]):
        stale = any(re.search(r'\b' + f + r'\b', body) for f in ('Oswald', 'Fraunces', 'Inter')) or 'screw-head-check' in body
        assert not stale or name in cfg['superseded'], name
    for row in cfg['features']:
        assert row['history'] == (row['name'] in cfg['superseded'])
        if row['history']:
            assert row['text'].startswith('Replaced by ')


def test_unmarked_retired_font_is_rejected(tmp_path: Path):
    config = _config_with(tmp_path, '"Workshop upload page" = ', '"Unrelated" = ')
    with pytest.raises(ValueError, match='Workshop upload page|unknown'):
        built_config.load(config)
    features = tmp_path / 'features.md'
    features.write_text('## A\n- What it does: Uses Oswald caps.\n\n## B\n- What it does: New.\n\n' + FILLER_MD)
    config = _config_with(tmp_path, '"Page shell: tokens, header, footer, themes" = "Site-wide look: guide, upload, wait, preview and shell"\n"Workshop upload page" = "Site-wide look: guide, upload, wait, preview and shell"\n"Waiting screen: your own video, parts outlined live" = "Waiting screen: named stages, real times, honest clock"', '"X" = "B"')
    config.write_text(_featured(config.read_text(), 'A', *FILLER))
    with pytest.raises(ValueError):
        built_config.load(config, features)
    ok = _config_with(tmp_path, '"Page shell: tokens, header, footer, themes" = "Site-wide look: guide, upload, wait, preview and shell"\n"Workshop upload page" = "Site-wide look: guide, upload, wait, preview and shell"\n"Waiting screen: your own video, parts outlined live" = "Waiting screen: named stages, real times, honest clock"', '"A" = "B"')
    ok.write_text(_featured(ok.read_text(), 'B', *FILLER))
    assert built_config.load(ok, features)['features'][0]['history'] is True


def test_duplicate_and_unknown_names_are_rejected(tmp_path: Path):
    """The audit probe: two copies of one guide section and a missing receipts section."""
    source = built_config.FEATURES.read_text(encoding='utf-8')
    duplicated = tmp_path / 'dup.md'
    duplicated.write_text(source.replace('## Model call receipts', '## Real-frame maker guide'))
    with pytest.raises(ValueError, match='duplicate feature names: Real-frame maker guide'):
        built_config.load(features=duplicated)
    renamed = tmp_path / 'renamed.md'
    renamed.write_text(source.replace('## Waiting screen: named stages, real times, honest clock', '## Waiting screen E'))
    with pytest.raises(ValueError, match='unknown'):
        built_config.load(features=renamed)


def test_first_sentence_handles_opening_markdown(tmp_path: Path):
    source = tmp_path / 'features.md'
    source.write_text('## A\n- What it does: First sentence. `finish` starts the next. More.\n\n' + FILLER_MD)
    config = tmp_path / 'built.toml'
    cfg = built_config.CONFIG.read_text()
    start, end = cfg.index('[superseded]'), cfg.index('[retired_names]')
    config.write_text(_featured(cfg[:start] + '[superseded]\n\n' + cfg[end:], 'A', *FILLER))
    assert built_config.load(config, source)['features'][0]['text'] == 'First sentence.'


def test_guide_end_claim_matches_controls_the_reader_renders():
    from app.routes.site import share_html
    row = next(r for r in built_config.load()['features'] if r['name'].startswith('Guide end:'))
    assert not row['history']
    unshared = share_html(None)  # no caller of _render_reader passes share_url
    assert 'data-ge-download' in unshared and 'data-ge-copy' not in unshared and 'data-ge-consent' not in unshared
    assert 'image download' in row['text'] and 'no share link' in row['text']
    assert 'consent tick gating' not in row['text']


def test_history_labels_come_from_config_and_are_escaped(tmp_path: Path):
    config = _config_with(tmp_path, 'history_label = "History ·"', 'history_label = "<b>Old</b>"')
    text = config.read_text().replace('replaced_label = "Replaced by"', 'replaced_label = "Now:"')
    config.write_text(text)
    rows = built_config.load(config)['features']
    assert any(r['text'].startswith('Now: ') for r in rows if r['history'])
    assert 'content:' not in (built_config.ROOT / 'static/ui/built.css').read_text()
    from unittest.mock import patch
    real = built_config.load
    with patch.object(built_config, 'load', lambda: real(config)):
        page = TestClient(create_site_app()).get('/built').text
    assert '<span class="built-label">&lt;b&gt;Old&lt;/b&gt;</span>' in page and '<b>Old</b>' not in page
    missing = tmp_path / 'missing.toml'
    missing.write_text(built_config.CONFIG.read_text().replace('history_label = "History ·"\n', ''))
    with pytest.raises(ValueError, match='Invalid build notes'):
        built_config.load(missing)


def test_built_lists_spark_open_up_top_and_keeps_its_feature_row_plain():
    text = TestClient(create_site_app()).get('/built').text
    assert '<dt>Muse Spark Open</dt>' in text
    assert 'second opinion on each part label' in text
    assert 'flags the label for a person to review' in text
    row = next(r for r in built_config.load()['features'] if r['name'] == 'Open-weights second check of part labels')
    assert '`' not in row['text'] and 'reasoning_effort' not in row['text']


def test_built_renders_the_configured_badge_and_no_brand_token():
    from app import brand_config
    text = TestClient(create_site_app()).get('/built').text
    assert brand_config.mark_badge(brand_config.load()) in text
    assert '@@brand_' not in text


def test_built_has_no_unfinished_wording_and_leads_with_a_summary():
    import re
    text = TestClient(create_site_app()).get('/built').text
    assert 'to confirm' not in text.lower() and '`' not in text
    assert text.index('In short') < text.index('Current features') < text.index('Build window') < text.index('Meta tooling')
    summary = text[text.index('In short'):text.index('Current features')]
    assert 'Spark Open' in summary and 'Honest limits' in summary
    assert 'raw input only' in text and 'original project work' in text
    assert len(re.findall(r'<dt>', summary)) == 4


LEAKS = {
    'local path': r'(?<![\w.])(?:/Users/|/private/|/home/|/tmp/|~/\.)',
    'email': r'[\w.+-]+@[\w-]+\.[\w.-]+',
    'credential': r'\bsk-[A-Za-z0-9]|\bAKIA[0-9A-Z]{8}|BEGIN [A-Z ]*PRIVATE KEY|\b(?:api[_-]?)?(?:key|token)\s*=\s*\S',
    'ai credit': r'co-authored-by|generated (?:by|with) (?:ai|claude|chatgpt|gpt|codex)|written (?:by|with) (?:ai|claude|chatgpt|gpt)',
}


def test_readme_and_built_page_have_no_private_paths_or_placeholders():
    page = TestClient(create_site_app()).get('/built').text
    readme = (built_config.ROOT / 'README.md').read_text(encoding='utf-8')
    for banned in ("owner's own footage", 'to confirm'):
        assert banned not in page, banned
        assert banned not in readme, banned
    for where, text in (('/built', page), ('README.md', readme)):
        for kind, pattern in LEAKS.items():
            hit = re.search(pattern, text, flags=re.I)
            assert not hit, (where, kind, hit and text[max(0, hit.start() - 30):hit.end() + 30])


def test_built_shows_current_features_first_and_history_in_a_closed_disclosure():
    page = TestClient(create_site_app()).get('/built').text
    cfg = built_config.load()
    head, _, tail = page.partition('<details class="built-all">')
    assert '<details class="built-all" open' not in page and tail
    featured = cfg['page']['current_features']
    assert 6 <= len(featured) <= 8  # the shipped page shows 6 to 8
    from html import escape
    for name in featured:
        assert head.count('<dt>' + escape(name) + '</dt>') == 1 and escape(name) not in tail.split('<dt>')[0]
    assert head.count('<dt>') < len(_index_names()) and page.count('<dt>') > len(featured)
    assert page.index('Current features') < page.index('Meta tooling')
    for kept in ('held privately by the author', 'Muse Spark Open', 'every part is modelled', '26 Oct 2026', '29 Oct 2026'):
        assert kept in page, kept


def test_current_features_must_be_live_and_at_most_eight(tmp_path: Path):
    cfg = built_config.CONFIG.read_text()
    start = cfg.index('current_features = [')
    block = cfg[start:cfg.index(']\n', start) + 1]
    with pytest.raises(ValueError, match='6 to 8'):
        built_config.load(_config_with(tmp_path, block, 'current_features = []'))
    with pytest.raises(ValueError, match='unknown or replaced'):
        built_config.load(_config_with(tmp_path, '"Real-frame maker guide"', '"Workshop upload page"'))


def test_current_feature_count_must_be_six_to_eight(tmp_path: Path):
    names = built_config.load()['page']['current_features']
    for count in (5, 9):
        bad = tmp_path / f'n{count}.toml'
        bad.write_text(_featured(built_config.CONFIG.read_text(), *(names + names)[:count]))
        with pytest.raises(ValueError, match='6 to 8'):
            built_config.load(bad)
    ok = tmp_path / 'n6.toml'
    ok.write_text(_featured(built_config.CONFIG.read_text(), *names[:6]))
    assert built_config.load(ok)['page']['current_features'] == names[:6]


def test_built_disclosure_count_matches_the_rows_inside_it():
    import re
    from fastapi.testclient import TestClient
    from app.routes.site import create_site_app
    html = TestClient(create_site_app()).get('/built').text
    summary = re.search(r'<summary>[^<]*\((\d+) earlier entries\)</summary>(.*?)</details>', html, re.S)
    assert int(summary.group(1)) == summary.group(2).count('<dt>')
