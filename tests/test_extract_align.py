"""Replay fixture, aligned consensus, timing scorer.

Proves the redacted replay fixture replays through the exact (pre-change)
consensus plus stitch into guide.json's six steps, pins the aligned
consensus behavior (paraphrase merge, one-to-many splits, adjacent-phase
split, exact rollback), and checks the canonical timing scorer aligns the
current guide by time overlap (1/9 strict on the exact replay, 2/9 on the
aligned replay) while still failing the 7/9 bar.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.chunks import ChunkWindow, StitchedSteps, plan_chunks, stitch_bundles
from app.contracts import PartRequest, Step, StepBundle
from app.scorer_time import score_guide
from app.settings import (
    STEP_ALIGNMENT_GENERIC_TOKENS_ENV_VAR,
    STEP_ALIGNMENT_MODE_ENV_VAR,
    STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR,
    STEP_ALIGNMENT_PART_SYNONYMS_ENV_VAR,
    STEP_ALIGNMENT_QUALIFIERS_ENV_VAR,
    STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR,
    load_provider_settings,
)
from app.spark import (
    SparkStageConfig,
    StepsDisagreement,
    bundle_to_json,
    consensus_bundle,
    consensus_digest,
    validate_legacy_spark_payload,
    validate_spark_payload,
)
from app.step_alignment import (
    GENERIC_TOKENS,
    NEGATION_WORDS,
    PART_SYNONYM_GROUPS,
    QUALIFIERS,
    VERB_SYNONYM_GROUPS,
    Window,
    cluster_windows,
    majority_of,
    same_action,
    signature,
)

from test_spark_schema import make_payload, make_step

FIXTURES = Path(__file__).parent / "fixtures"
REPLAY = FIXTURES / "run4_replay.json"
GT_PROPOSED = FIXTURES / "sample_steps_proposed.json"

EXACT = SparkStageConfig(alignment_mode="exact")
ALIGN = SparkStageConfig()  # aligned consensus is the default
LEGACY = SparkStageConfig(span_mode="legacy")  # sparse wording consensus


def _replay_fixture() -> dict:
    return json.loads(REPLAY.read_text(encoding="utf-8"))


def _gt_proposed() -> list[dict]:
    data = json.loads(GT_PROPOSED.read_text(encoding="utf-8"))
    assert data["status"] == "proposed"
    return data["steps"]


def _gt_excluded() -> list[tuple[float, float]]:
    data = json.loads(GT_PROPOSED.read_text(encoding="utf-8"))
    return [(span["start_s"], span["end_s"]) for span in data["excluded_spans"]]


def _replay(mode: str) -> StitchedSteps:
    fixture = _replay_fixture()
    config = SparkStageConfig(alignment_mode=mode)
    bundles = []
    for chunk in fixture["chunks"]:
        runs = [validate_legacy_spark_payload(payload) for payload in chunk["runs"]]
        bundles.append(
            (
                ChunkWindow(chunk["window"][0], chunk["window"][1]),
                consensus_bundle(runs, config),
            )
        )
    return stitch_bundles(bundles, overlap_s=2.0, alignment_mode=mode)


def _consensus(step_sets: list[tuple], config: SparkStageConfig = ALIGN):
    runs = [
        validate_spark_payload(make_payload(steps=steps)) for steps in step_sets
    ]
    return consensus_bundle(runs, config)


def _stitched_step(step: Step) -> dict:
    return {
        "number": step.number,
        "start_s": step.start_s,
        "end_s": step.end_s,
        "action": step.action,
        "tools": list(step.tools),
        "parts": [
            {"phrase": part.phrase, "position_hint": part.position_hint}
            for part in step.parts
        ],
        "hazards": list(step.hazards),
        "time_spread_s": step.time_spread_s,
        "time_needs_check": step.time_needs_check,
    }


# --- replay fixture ----------------------------------------------------------


def test_fixture_windows_match_plan_chunks() -> None:
    fixture = _replay_fixture()
    windows = plan_chunks(
        fixture["video_duration_s"], fixture["chunk_s"], fixture["overlap_s"]
    )
    assert [(w.start_s, w.end_s) for w in windows] == [
        (0.0, 60.0),
        (58.0, 118.0),
        (116.0, 176.0),
        (174.0, 214.167),
    ]
    assert [chunk["window"] for chunk in fixture["chunks"]] == [
        [w.start_s, w.end_s] for w in windows
    ]


def test_fixture_is_redacted_step_payloads_only() -> None:
    fixture = _replay_fixture()
    assert [chunk["window"] for chunk in fixture["chunks"]] is not None
    for chunk in fixture["chunks"]:
        assert len(chunk["runs"]) == 3
        for payload in chunk["runs"]:
            assert set(payload) == {"device_guess", "confidence", "steps"}
            for step in payload["steps"]:
                assert set(step) == {
                    "number", "start_s", "end_s", "action",
                    "tools", "parts", "hazards",
                }
                for part in step["parts"]:
                    assert set(part) == {"phrase", "position_hint"}
    text = REPLAY.read_text(encoding="utf-8")
    for marker in (
        "file-", "/tmp/", "/Users/", "Bearer", "META_API",
        ".png", ".mp4", "job_id", "file_id",
    ):
        assert marker not in text


def test_replay_exact_reproduces_guide_json_steps() -> None:
    fixture = _replay_fixture()
    counts = []
    bundles = []
    for chunk in fixture["chunks"]:
        runs = [validate_legacy_spark_payload(payload) for payload in chunk["runs"]]
        bundle = consensus_bundle(runs, EXACT)
        counts.append(len(bundle.steps))
        assert bundle_to_json(bundle) == chunk["expected_consensus"]
        bundles.append(
            (ChunkWindow(chunk["window"][0], chunk["window"][1]), bundle)
        )
    assert counts == [2, 1, 2, 1]
    stitched = stitch_bundles(bundles, overlap_s=2.0, alignment_mode="exact")
    assert [_stitched_step(step) for step in stitched.steps] == fixture["expected_guide"]
    assert [(s.start_s, s.end_s, s.action) for s in stitched.steps] == [
        (11.5, 17.0, "Flip over and remove external battery"),
        (17.0, 31.0, "Disable internal battery in BIOS"),
        (64.0, 67.0, "Remove bottom screws"),
        (126.0, 135.5, "Screw down and connect battery"),
        (141.5, 160.5, "Refit bottom cover"),
        (183.5, 192.5, "Check Device Manager"),
    ]


def test_replay_align_restores_dropped_phases() -> None:
    fixture = _replay_fixture()
    counts = []
    for chunk in fixture["chunks"]:
        runs = [validate_legacy_spark_payload(payload) for payload in chunk["runs"]]
        steps = consensus_bundle(runs, ALIGN).steps
        counts.append(len(steps))
        for left, right in zip(steps, steps[1:]):
            assert left.end_s == right.start_s  # tiled by construction
    assert counts == [6, 7, 7, 5]
    stitched = _replay("align")
    # contiguous consensus matches runs by time,
    # so nothing drops to a single run anymore -- the formerly
    # verb-less ("Power down and turn over", "Close lid and test
    # charging") and qualifier-split wordings survive as kept,
    # flagged segments, and each chunk tiles its observed span.
    # Labels follow the majority wording per segment (the intro
    # reads "Show laptop overview", not the old median-start
    # "Introduce laptop on table"). One boundary per run per cluster,
    # so short phases (7-9, 4-8) survive instead of merging.
    assert [(s.start_s, s.end_s, s.action) for s in stitched.steps] == [
        (0.0, 9.0, "Show laptop overview"),
        (9.0, 17.0, "Flip over and remove battery"),
        (17.0, 30.0, "Disable internal battery in BIOS"),
        (30.0, 36.5, "Power down and turn over"),
        (36.5, 51.0, "Remove bottom cover screws"),
        (51.0, 66.0, "Inspect screw and open cover"),
        (58.0, 65.0, "Opening intro overview"),
        (65.0, 67.0, "Remove bottom screws"),
        (67.0, 86.0, "Pry off bottom cover"),
        (86.0, 93.0, "Lift off bottom cover"),
        (93.0, 106.0, "Disconnect battery cable"),
        (106.0, 118.0, "Remove internal battery"),
        (116.0, 120.0, "Opening intro"),
        (118.0, 125.0, "Inspect replacement battery"),
        (120.0, 126.0, "Position internal battery"),
        (126.0, 136.0, "Screw down and connect battery"),
        (136.0, 141.0, "Inspect cooling assembly"),
        (141.0, 161.0, "Reattach bottom cover"),
        (161.0, 174.0, "Secure cover and fit external battery"),
        (174.0, 180.0, "Close lid and test charging"),
        (174.0, 180.0, "Show closed laptop intro"),
        (180.0, 182.0, "Show closed laptop intro"),
        (182.0, 193.0, "Check Device Manager specs"),
        (193.0, 204.5, "Display upgrade RAM kits"),
        (204.5, 217.0, "Present RAM upgrade kit"),
    ]
    actions = [step.action for step in stitched.steps]
    assert stitched.steps[0].start_s == 0.0  # GT 1 phase restored
    assert "Remove internal battery" in actions  # GT 6 phase restored
    assert "Position internal battery" in actions  # GT 7 phase restored


# --- strict signature table --------------------------------------------------
# One strict signature, used by the aligner and reported per pair by the
# timing scorer. Every row is checked at all three levels: the shared
# predicate (both directions), two temporally linkable windows, and a
# single-row scorer alignment (paired by time; the wording verdict is
# reported, never the gate). Red-before rows failed on the earlier matcher
# (verified by replaying this table against 93afd49: internal-external,
# compound, negated, noun-only, and pos-cable); kept rows
# passed then and still pass. generic-only failed on a later
# matcher (verified against c421505 before the fix).


SAME_ACTION_TABLE = [
    # (case id, action a, action b, expected same_action)
    ("direct", "Disconnect battery cable", "Remove internal battery", False),
    ("bridge-ab", "Unscrew base panel", "Unscrew base cover", True),
    ("bridge-bc", "Unscrew base cover", "Pry off base cover", False),
    ("bridge-ac", "Unscrew base panel", "Pry off base cover", False),
    ("nested", "Inspect battery", "Remove battery", False),
    ("keyboard", "Inspect keyboard", "Remove keyboard", False),
    (
        "internal-external",
        "Remove internal battery",
        "Remove external battery",
        False,
    ),
    (
        "compound",
        "Remove and install internal battery",
        "Install internal battery",
        False,
    ),
    ("negated", "Do not remove battery", "Remove battery", False),
    ("noun-only", "Battery", "Battery", False),
    ("generic-only", "Remove battery", "Remove device", False),
    ("pos-pry-cover", "Pry off cover", "Pry open the cover", True),
    (
        "pos-screws",
        "Remove bottom cover screws",
        "Unscrew bottom cover screws",
        True,
    ),
    (
        "pos-cable",
        "Disconnect battery cable",
        "Unplug battery connector",
        True,
    ),
    # opposite or merely different transition verbs
    # never merge, in either mode.
    (
        "neg-power-direction",
        "Power on the unit",
        "Power off the unit",
        False,
    ),
    (
        "neg-transition-mismatch",
        "Boot the unit",
        "Shut down the unit",
        False,
    ),
]


# Rows that hold only under the contiguous/phases rules: a
# transition-only extra never splits an action, and identical
# verb-less wordings match on shared part evidence. Sparse/legacy
# mode keeps the old predicate (strictly equal verb sets, noun-only
# never matches), so these rows run with phases_mode=True.
PHASES_ACTION_TABLE = [
    # (case id, action a, action b, expected same_action)
    (
        "pos-transition-extra",
        "Flip over and loosen lid screws",
        "Loosen lid screws",
        True,
    ),
    ("pos-identical-verbless", "Alpha phase", "Alpha phase", True),
]


@pytest.mark.parametrize(
    "case_id,first,second,expected",
    SAME_ACTION_TABLE,
    ids=[row[0] for row in SAME_ACTION_TABLE],
)
def test_same_action_table(
    case_id: str, first: str, second: str, expected: bool
) -> None:
    assert case_id  # the id names the case the row covers
    assert same_action(first, second) is expected
    assert same_action(second, first) is expected  # symmetric
    windows = [Window(0.0, 10.0, first), Window(5.0, 15.0, second)]
    assert cluster_windows(windows, tolerance_s=5.0) == (
        [[0, 1]] if expected else [[0], [1]]
    )
    guide = [{"number": 1, "start_s": 0.0, "end_s": 10.0, "action": first}]
    gt = [{"number": 1, "start_s": 0.0, "end_s": 10.0, "action": second}]
    report = score_guide(guide, gt)
    # Time rules pair identical windows regardless of wording; the wording
    # verdict is reported via the contiguous rules instead of gating.
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(1, 1)]
    assert report.missing_gt == ()
    assert report.extra_guide == ()
    assert report.matches[0].wording_same_action is same_action(
        first, second, phases_mode=True
    )


@pytest.mark.parametrize(
    "case_id,first,second,expected",
    PHASES_ACTION_TABLE,
    ids=[row[0] for row in PHASES_ACTION_TABLE],
)
def test_phases_action_table(
    case_id: str, first: str, second: str, expected: bool
) -> None:
    assert case_id  # the id names the case the row covers
    assert same_action(first, second, phases_mode=True) is expected
    assert same_action(second, first, phases_mode=True) is expected
    windows = [Window(0.0, 10.0, first), Window(5.0, 15.0, second)]
    assert cluster_windows(windows, tolerance_s=5.0, phases_mode=True) == (
        [[0, 1]] if expected else [[0], [1]]
    )
    guide = [{"number": 1, "start_s": 0.0, "end_s": 10.0, "action": first}]
    gt = [{"number": 1, "start_s": 0.0, "end_s": 10.0, "action": second}]
    report = score_guide(guide, gt)  # the scorer reports the contiguous rules
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(1, 1)]
    assert report.missing_gt == ()
    assert report.extra_guide == ()
    assert report.matches[0].wording_same_action is expected


def test_align_rejects_noun_only_contained_windows() -> None:
    """identical noun-only text splits."""
    windows = [
        Window(0.0, 20.0, "Battery"),
        Window(8.0, 12.0, "Battery"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0], [1]]


def test_signature_components() -> None:
    """The signature exposes verbs, negation, parts, and qualifiers."""
    compound = signature("Remove and install internal battery")
    assert len(compound.verbs) == 2  # one verb per clause
    assert compound.negated is False
    assert compound.qualifiers == frozenset({"internal"})
    assert (
        signature("Pry open the cover", phases_mode=True).verbs
        == signature("Pry off cover", phases_mode=True).verbs
    )  # mid-clause "open" counts but normalises to the pry group
    assert signature("Battery").verbs == frozenset()  # noun-only
    for wording in (
        "Do not remove battery",
        "Don't remove battery",
        "Never remove battery",
        "Avoid removing battery",
        "Remove battery without tools",
    ):
        assert signature(wording).negated is True, wording
    assert signature("Remove battery").negated is False
    assert signature("Knot the cable").negated is False  # a substring, not "not"


# --- aligned consensus -------------------------------------------------------


def test_align_merges_intro_paraphrase() -> None:
    bundle = _consensus(
        [
            (make_step(start=0.0, end=9.0, action="Introduce laptop on table",
                       parts=("laptop lid",)),),
            (make_step(start=0.0, end=3.0, action="Introduce closed laptop",
                       parts=("laptop lid",)),),
            (make_step(start=0.0, end=9.0, action="Present laptop overview",
                       parts=("laptop lid", "table")),),
        ]
    )
    (step,) = bundle.steps
    assert (step.start_s, step.end_s) == (0.0, 9.0)
    assert step.action == "Introduce laptop on table"
    assert step.time_needs_check is True  # end spread 6 s exceeds the line


def test_align_show_closeup_does_not_join_intro() -> None:
    """``show`` is no introduction synonym: shared parts cannot bridge verbs.

    A screwdriver close-up that lists the laptop lid still overlaps the
    intro, but the verbs differ, so it phases apart (a shared noun is not same-action evidence).
    """
    windows = [
        Window(0.0, 9.0, "Introduce laptop on table", ("laptop lid",)),
        Window(3.0, 10.0, "Show screwdriver tool", ("laptop lid", "screwdriver")),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0], [1]]


def test_align_identical_show_wordings_link_with_part_evidence() -> None:
    """Identical ``show`` verbs still link when their parts agree."""
    windows = [
        Window(9.0, 13.0, "Show screwdriver tool", ("screwdriver",)),
        Window(3.0, 10.0, "Show screwdriver tool", ("laptop lid", "screwdriver")),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0, 1]]


def test_align_merges_cover_pry_with_differing_part_tuples() -> None:
    bundle = _consensus(
        [
            (make_step(start=10.0, end=28.0, action="Pry open bottom cover",
                       parts=("bottom cover", "side ports")),),
            (make_step(start=9.0, end=28.0, action="Pry off bottom cover",
                       parts=("side ports", "vent grille", "bottom chassis")),),
            (make_step(start=8.0, end=28.0, action="Pry open bottom cover",
                       parts=("bottom cover", "side vent")),),
        ]
    )
    (step,) = bundle.steps
    assert (step.start_s, step.end_s) == (8.0, 28.0)  # observed span
    assert step.action == "Pry open bottom cover"  # largest-overlap sample
    assert step.time_needs_check is False


def test_align_merges_one_to_many_battery_removal() -> None:
    # every sample carries the same strict signature (one
    # removal verb, the "internal" qualifier, the shared "fan" part), so
    # the split run still phases many-to-one. A same-group compound
    # ("Unscrew and detach ...") collapses to the one removal verb; an
    # ambiguous verb ("Lift ...") or a missing qualifier would split --
    # see the table's compound and internal-external rows.
    bundle = _consensus(
        [
            (make_step(start=48.0, end=66.0, action="Remove internal battery",
                       parts=("battery", "fan")),),
            (
                make_step(start=47.0, end=55.0, action="Unscrew internal battery",
                          parts=("battery", "fan")),
                make_step(start=55.0, end=60.0, action="Detach internal battery",
                          parts=("battery", "fan")),
            ),
            (make_step(start=48.0, end=60.0, action="Unscrew and detach internal battery",
                       parts=("battery", "fan", "motherboard")),),
        ]
    )
    (step,) = bundle.steps
    # Time consensus: the lone 55.0 split edge drops (one run only),
    # so the three runs merge many-to-one over the observed span.
    assert (step.start_s, step.end_s) == (47.0, 66.0)
    assert step.action == "Remove internal battery"
    assert step.time_spread_s == 6.0
    assert step.time_needs_check is True


def test_align_keeps_adjacent_disconnect_and_remove_split() -> None:
    pair = (
        make_step(start=10.0, end=20.0, action="Disconnect battery cable",
                  parts=("battery", "cable connector")),
        make_step(start=20.0, end=40.0, action="Remove internal battery",
                  parts=("battery", "fan")),
    )
    bundle = _consensus([pair, pair, pair])
    assert [s.action for s in bundle.steps] == [
        "Disconnect battery cable",
        "Remove internal battery",
    ]
    assert [(s.start_s, s.end_s) for s in bundle.steps] == [
        (10.0, 20.0),
        (20.0, 40.0),
    ]
    assert [s.time_needs_check for s in bundle.steps] == [False, False]


def test_align_rejects_generic_token_link_between_adjacent_actions() -> None:
    """one shared generic noun is not a link."""
    windows = [
        Window(0.0, 10.0, "Disconnect battery cable"),
        Window(5.0, 15.0, "Remove internal battery"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0], [1]]


def test_align_blocks_transitive_bridge_between_unlinked_endpoints() -> None:
    """B links A and C, A/C stay separate."""
    windows = [
        Window(0.0, 10.0, "Unscrew base panel"),
        Window(5.0, 15.0, "Unscrew base cover"),
        Window(10.0, 20.0, "Pry off base cover"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0, 1], [2]]


def test_align_touching_identical_signatures_link() -> None:
    """Branch (a): fully equal signatures link on tolerance alone, even
    merely touching -- a boundary dispute over identical wordings still
    merges with its spread flagged."""
    windows = [
        Window(0.0, 10.0, "Remove bottom cover"),
        Window(10.0, 20.0, "Remove bottom cover"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0, 1]]


def test_align_touching_paraphrases_stay_split() -> None:
    """Branch (b): same-action paraphrases still need a strict overlap."""
    windows = [
        Window(0.0, 10.0, "Unscrew base panel"),
        Window(10.0, 20.0, "Unscrew base cover"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0], [1]]


def test_align_rejects_nested_inspect_remove_stays_split() -> None:
    """containment plus a shared noun is not a link."""
    windows = [
        Window(0.0, 20.0, "Inspect battery"),
        Window(8.0, 12.0, "Remove battery"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0], [1]]


def test_align_extra_step_does_not_strand_later_steps() -> None:
    """A finer run's unaligned extra leaves the run's floor where it was."""
    windows = [
        Window(0.0, 10.0, "Unscrew base panel"),
        Window(20.0, 30.0, "Pry off base cover"),
        Window(0.0, 10.0, "Unscrew base panel"),
        Window(25.0, 30.0, "Wipe workbench surface"),
        Window(20.0, 36.0, "Pry off base cover"),
        Window(0.0, 10.0, "Unscrew base panel"),
        Window(20.0, 30.0, "Pry off base cover"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0, run_of=[0, 0, 1, 1, 1, 2, 2]) == [
        [0, 2, 5],
        [1, 4, 6],
        [3],
    ]


def test_align_refuses_crossing_match_between_runs() -> None:
    """Time consensus: a reordered run is excluded and cannot corrupt.

    Run 1 lists the phases backwards, so it is not a contiguous
    partition and drops; consensus comes from the two ordered runs
    (monotonic stranding still rules the legacy path and the shared
    clusterer).
    """
    run0 = (
        make_step(start=0.0, end=10.0, action="Unscrew base panel"),
        make_step(start=10.0, end=30.0, action="Pry off base cover"),
    )
    run1 = (
        make_step(start=20.0, end=30.0, action="Pry off base cover"),
        make_step(start=0.0, end=16.0, action="Unscrew base panel"),
    )
    bundle = _consensus([run0, run1, run0])
    assert [(s.action, s.start_s, s.end_s) for s in bundle.steps] == [
        ("Unscrew base panel", 0.0, 10.0),
        ("Pry off base cover", 10.0, 30.0),
    ]
    assert [s.time_needs_check for s in bundle.steps] == [False, False]


def test_align_flags_same_action_rivals_with_disjoint_parts() -> None:
    """same action + disjoint parts + overlap = flagged.

    A legacy-path pin: contiguous phases mode separates the two
    wordings by time instead of flagging rivals.
    """
    pair = (
        make_step(start=10.0, end=20.0, action="Inspect device",
                  parts=("cooling fan",)),
        make_step(start=12.0, end=22.0, action="Inspect device",
                  parts=("display panel",)),
    )
    bundle = _consensus([pair, pair, pair], LEGACY)
    assert [s.action for s in bundle.steps] == ["Inspect device", "Inspect device"]
    assert [s.time_needs_check for s in bundle.steps] == [True, True]


def test_align_merges_identical_steps_without_known_verbs() -> None:
    """identical wordings match on shared part evidence.

    "Seal the case" is imperative English but "seal" is outside the
    configured verb lexicon; three runs agreeing word-for-word still
    form a phase (the live spark3 failure was verb-less transition
    phases dropping to singles). Vetoes still win: negated or
    disjoint-part identicals split (see the table and the rivals
    tests).
    """
    assert signature("Seal the case").verbs == frozenset()
    step = make_step(
        start=50.0, end=60.0, action="Seal the case", parts=(),
    )
    (merged,) = _consensus([(dict(step),)] * 3).steps
    assert (merged.start_s, merged.end_s) == (50.0, 60.0)
    assert merged.action == "Seal the case"


def test_exact_mode_is_the_red_mutant_for_paraphrases() -> None:
    """The plan's mutant check: exact keys drop the pry paraphrase (red);
    the aligned implementation merges it (green)."""
    steps = [
        (make_step(start=10.0, end=28.0, action="Pry open bottom cover",
                   parts=("bottom cover", "side ports")),),
        (make_step(start=9.0, end=28.0, action="Pry off bottom cover",
                   parts=("side ports", "vent grille", "bottom chassis")),),
        (make_step(start=8.0, end=28.0, action="Pry open bottom cover",
                   parts=("bottom cover", "side vent")),),
    ]
    with pytest.raises(StepsDisagreement):
        _consensus(steps, EXACT)
    (step,) = _consensus(steps, ALIGN).steps
    assert step.action == "Pry open bottom cover"  # majority wording wins


def _bundle(steps: list[Step]) -> StepBundle:
    return StepBundle(
        device_guess="laptop",
        confidence=0.9,
        steps=tuple(steps),
        source_runs=3,
        timestamps_approximate=True,
    )


def _step(number: int, start: float, end: float, action: str, phrase: str) -> Step:
    return Step(
        number=number,
        start_s=start,
        end_s=end,
        action=action,
        tools=(),
        parts=(PartRequest(phrase=phrase, position_hint="here"),),
        hazards=(),
        time_spread_s=0.0,
        time_needs_check=False,
    )


def test_stitch_align_merges_paraphrase_sightings() -> None:
    bundles = [
        (ChunkWindow(0.0, 60.0),
         _bundle([_step(1, 52.0, 60.0, "Pry open bottom cover", "bottom cover")])),
        (ChunkWindow(58.0, 118.0),
         _bundle([_step(1, 0.0, 6.0, "Pry off bottom cover", "bottom cover")])),
    ]
    stitched = stitch_bundles(bundles, overlap_s=2.0, alignment_mode="align")
    (merged,) = stitched.steps
    assert (merged.start_s, merged.end_s) == (52.0, 64.0)
    assert merged.action == "Pry open bottom cover"  # earlier sighting wins
    rival = stitch_bundles(bundles, overlap_s=2.0, alignment_mode="exact")
    assert len(rival.steps) == 2  # rollback keeps both wordings


def test_stitch_align_flags_same_action_rivals_with_disjoint_parts() -> None:
    """component identity keeps part evidence."""
    bundles = [
        (ChunkWindow(0.0, 60.0),
         _bundle([_step(1, 10.0, 20.0, "Inspect device", "cooling fan")])),
        (ChunkWindow(0.0, 60.0),
         _bundle([_step(1, 12.0, 22.0, "Inspect device", "display panel")])),
    ]
    stitched = stitch_bundles(bundles, overlap_s=2.0, alignment_mode="align")
    assert [(s.start_s, s.end_s) for s in stitched.steps] == [
        (10.0, 20.0),
        (12.0, 22.0),
    ]
    assert [s.time_needs_check for s in stitched.steps] == [True, True]


def test_invalid_alignment_mode_rejected() -> None:
    with pytest.raises(ValueError):
        SparkStageConfig(alignment_mode="bogus")
    bundles = [
        (ChunkWindow(0.0, 60.0),
         _bundle([_step(1, 1.0, 2.0, "First", "fan")])),
    ]
    with pytest.raises(ValueError):
        stitch_bundles(bundles, overlap_s=2.0, alignment_mode="bogus")


def test_step_alignment_mode_setting() -> None:
    assert load_provider_settings({}).step_alignment_mode == "align"
    tuned = load_provider_settings({STEP_ALIGNMENT_MODE_ENV_VAR: "exact"})
    assert tuned.step_alignment_mode == "exact"
    with pytest.raises(RuntimeError):
        load_provider_settings({STEP_ALIGNMENT_MODE_ENV_VAR: "bogus"})


def test_step_alignment_generic_tokens_setting() -> None:
    assert load_provider_settings({}).step_alignment_generic_tokens == GENERIC_TOKENS
    tuned = load_provider_settings(
        {STEP_ALIGNMENT_GENERIC_TOKENS_ENV_VAR: "Battery, cell"}
    )
    assert tuned.step_alignment_generic_tokens == frozenset({"battery", "cell"})
    for bad in ("battery cell", "battery,,cover"):
        with pytest.raises(RuntimeError):
            load_provider_settings({STEP_ALIGNMENT_GENERIC_TOKENS_ENV_VAR: bad})
    blank = load_provider_settings({STEP_ALIGNMENT_GENERIC_TOKENS_ENV_VAR: "  "})
    assert blank.step_alignment_generic_tokens == GENERIC_TOKENS


def test_same_action_needs_verb_evidence_plus_parts() -> None:
    """pinned at the shared predicate itself.

    qualifier-only evidence no longer
    suffices -- an empty part set on either side never matches, so the
    internal-battery paraphrases fail closed.
    """
    assert same_action("Detach internal battery", "Remove internal battery") is False
    assert same_action("Take out the internal battery", "Remove internal battery") is False
    assert same_action("Inspect keyboard", "Remove keyboard") is False
    assert same_action("Remove bottom screws", "Remove internal battery") is False
    assert same_action("Pry off cover", "Pry off cover") is True
    assert (
        same_action(
            "Inspect device", "Inspect device",
            frozenset({"fan"}), frozenset({"display"}),
        )
        is False
    )


def test_step_alignment_verb_synonyms_setting() -> None:
    assert load_provider_settings({}).step_alignment_verb_synonyms == VERB_SYNONYM_GROUPS
    tuned = load_provider_settings(
        {STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR: "Remove, detach;Disconnect, unplug"}
    )
    assert tuned.step_alignment_verb_synonyms == (
        frozenset({"remove", "detach"}),
        frozenset({"disconnect", "unplug"}),
    )
    phrased = load_provider_settings(
        {STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR: "take out, remove"}
    )
    assert phrased.step_alignment_verb_synonyms == (
        frozenset({"take out", "remove"}),
    )
    for bad in ("remove,,detach", "remove;", ";disconnect", "re-move,detach", "remove,"):
        with pytest.raises(RuntimeError):
            load_provider_settings({STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR: bad})
    blank = load_provider_settings({STEP_ALIGNMENT_VERB_SYNONYMS_ENV_VAR: "  "})
    assert blank.step_alignment_verb_synonyms == VERB_SYNONYM_GROUPS


def test_step_alignment_negation_words_setting() -> None:
    assert load_provider_settings({}).step_alignment_negation_words == NEGATION_WORDS
    tuned = load_provider_settings(
        {STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR: "Not, never"}
    )
    assert tuned.step_alignment_negation_words == frozenset({"not", "never"})
    phrased = load_provider_settings(
        {STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR: "do not, don't"}
    )
    assert phrased.step_alignment_negation_words == frozenset({"do not", "dont"})
    for bad in ("not,,never", "do not!"):
        with pytest.raises(RuntimeError):
            load_provider_settings({STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR: bad})
    blank = load_provider_settings({STEP_ALIGNMENT_NEGATION_WORDS_ENV_VAR: "  "})
    assert blank.step_alignment_negation_words == NEGATION_WORDS


def test_step_alignment_qualifiers_setting() -> None:
    assert load_provider_settings({}).step_alignment_qualifiers == QUALIFIERS
    tuned = load_provider_settings(
        {STEP_ALIGNMENT_QUALIFIERS_ENV_VAR: "Internal, external"}
    )
    assert tuned.step_alignment_qualifiers == frozenset({"internal", "external"})
    for bad in ("internal external", "internal,,external"):
        with pytest.raises(RuntimeError):
            load_provider_settings({STEP_ALIGNMENT_QUALIFIERS_ENV_VAR: bad})
    blank = load_provider_settings({STEP_ALIGNMENT_QUALIFIERS_ENV_VAR: "  "})
    assert blank.step_alignment_qualifiers == QUALIFIERS


def test_step_alignment_part_synonyms_setting() -> None:
    assert load_provider_settings({}).step_alignment_part_synonyms == PART_SYNONYM_GROUPS
    tuned = load_provider_settings(
        {STEP_ALIGNMENT_PART_SYNONYMS_ENV_VAR: "Cable, connector; screw, fastener"}
    )
    assert tuned.step_alignment_part_synonyms == (
        frozenset({"cable", "connector"}),
        frozenset({"screw", "fastener"}),
    )
    for bad in ("cable,,connector", "cable;", ";connector", "cable connector,screw"):
        with pytest.raises(RuntimeError):
            load_provider_settings({STEP_ALIGNMENT_PART_SYNONYMS_ENV_VAR: bad})
    blank = load_provider_settings({STEP_ALIGNMENT_PART_SYNONYMS_ENV_VAR: "  "})
    assert blank.step_alignment_part_synonyms == PART_SYNONYM_GROUPS


def test_consensus_digest_keys_verb_synonyms() -> None:
    tuned = SparkStageConfig(
        verb_synonyms=(frozenset({"remove"}), frozenset({"detach"}))
    )
    assert consensus_digest("j", "v", ALIGN) != consensus_digest("j", "v", tuned)
    assert consensus_digest("j", "v", tuned) == consensus_digest("j", "v", tuned)
    with pytest.raises(ValueError):
        SparkStageConfig(verb_synonyms=(frozenset({"not a verb!"}),))
    with pytest.raises(ValueError):
        SparkStageConfig(verb_synonyms=[frozenset({"remove"})])  # type: ignore[arg-type]


def test_consensus_digest_keys_alignment_mode() -> None:
    assert consensus_digest("j", "v", EXACT) != consensus_digest("j", "v", ALIGN)
    assert consensus_digest("j", "v", ALIGN) == consensus_digest("j", "v", ALIGN)


def test_consensus_digest_keys_generic_tokens() -> None:
    tuned = SparkStageConfig(generic_tokens=frozenset({"battery"}))
    assert consensus_digest("j", "v", ALIGN) != consensus_digest("j", "v", tuned)
    assert consensus_digest("j", "v", tuned) == consensus_digest("j", "v", tuned)
    with pytest.raises(ValueError):
        SparkStageConfig(generic_tokens=frozenset({"not a token"}))


def test_consensus_digest_keys_negation_words() -> None:
    tuned = SparkStageConfig(negation_words=frozenset({"not"}))
    assert consensus_digest("j", "v", ALIGN) != consensus_digest("j", "v", tuned)
    assert consensus_digest("j", "v", tuned) == consensus_digest("j", "v", tuned)
    with pytest.raises(ValueError):
        SparkStageConfig(negation_words=frozenset({"not a word!"}))
    with pytest.raises(ValueError):
        SparkStageConfig(negation_words=["not"])  # type: ignore[arg-type]


def test_consensus_digest_keys_qualifiers() -> None:
    tuned = SparkStageConfig(qualifiers=frozenset({"internal"}))
    assert consensus_digest("j", "v", ALIGN) != consensus_digest("j", "v", tuned)
    assert consensus_digest("j", "v", tuned) == consensus_digest("j", "v", tuned)
    with pytest.raises(ValueError):
        SparkStageConfig(qualifiers=frozenset({"not a word"}))
    with pytest.raises(ValueError):
        SparkStageConfig(qualifiers=["internal"])  # type: ignore[arg-type]


def test_consensus_digest_keys_part_synonyms() -> None:
    tuned = SparkStageConfig(part_synonyms=(frozenset({"cable"}),))
    assert consensus_digest("j", "v", ALIGN) != consensus_digest("j", "v", tuned)
    assert consensus_digest("j", "v", tuned) == consensus_digest("j", "v", tuned)
    with pytest.raises(ValueError):
        SparkStageConfig(part_synonyms=(frozenset({"not a word"}),))
    with pytest.raises(ValueError):
        SparkStageConfig(part_synonyms=[frozenset({"cable"})])  # type: ignore[arg-type]


def test_majority_of() -> None:
    assert [majority_of(runs) for runs in (2, 3, 4, 5)] == [2, 2, 3, 3]
    for bad in (0, -1, True, "3"):
        with pytest.raises(ValueError):
            majority_of(bad)


# --- timing scorer --------------------------------------------------------------


def test_scorer_shared_noun_without_action_evidence_still_pairs_by_time() -> None:
    """Time alignment replaced the wording gate:
    one shared generic noun is not action evidence, but the identical
    windows pair anyway and the wording verdict is only reported."""
    guide = [
        {"number": 1, "start_s": 0.0, "end_s": 10.0, "action": "Inspect battery"}
    ]
    gt = [
        {
            "number": 1,
            "start_s": 0.0,
            "end_s": 10.0,
            "action": "Remove internal battery",
        }
    ]
    report = score_guide(guide, gt)
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(1, 1)]
    assert report.matches[0].passed is True
    assert report.matches[0].wording_same_action is False
    assert report.missing_gt == ()
    assert report.extra_guide == ()
    assert report.coverage == pytest.approx(1.0)
    assert report.strict_passes == 1
    assert report.passed is True  # 1/1 meets the 7-of-9 ratio


def test_scorer_shared_non_generic_noun_without_verb_still_pairs_by_time() -> None:
    """Time alignment replaced the wording gate:
    a shared noun alone is not action evidence, but the identical windows
    pair anyway and the wording verdict is only reported."""
    guide = [
        {"number": 1, "start_s": 0.0, "end_s": 10.0, "action": "Inspect keyboard"}
    ]
    gt = [
        {"number": 1, "start_s": 0.0, "end_s": 10.0, "action": "Remove keyboard"}
    ]
    report = score_guide(guide, gt)
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(1, 1)]
    assert report.matches[0].passed is True
    assert report.matches[0].wording_same_action is False
    assert report.missing_gt == ()
    assert report.extra_guide == ()
    assert report.coverage == pytest.approx(1.0)
    assert report.strict_passes == 1
    assert report.passed is True  # 1/1 meets the 7-of-9 ratio


def test_scorer_pry_paraphrase_is_a_match() -> None:
    """keeps verb paraphrases: pry off/open the cover match."""
    guide = [
        {"number": 1, "start_s": 60.0, "end_s": 87.0, "action": "Pry off cover"}
    ]
    gt = [
        {
            "number": 5,
            "start_s": 60.0,
            "end_s": 87.0,
            "action": "Pry open the cover",
        }
    ]
    report = score_guide(guide, gt)
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(1, 5)]
    assert report.matches[0].passed is True
    assert report.missing_gt == ()
    assert report.extra_guide == ()
    assert report.coverage == pytest.approx(1.0)


def test_scorer_current_guide_time_aligned_score() -> None:
    """Time alignment: the run-4 guide pairs by
    overlap, so its one real timing hit (external-battery removal lands
    within ±3 s of GT 2) scores -- wording stays reported, never the gate."""
    guide = _replay("exact").steps
    assert len(guide) == 6
    report = score_guide(guide, _gt_proposed(), excluded_spans=_gt_excluded())
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [
        (1, 2),
        (2, 3),
        (3, 5),
        (4, 7),
        (5, 8),
        (6, 9),
    ]
    assert [m.passed for m in report.matches] == [
        True, False, False, False, False, False,
    ]
    # The strict pass pairs across a wording the old gate rejected.
    assert report.matches[0].wording_same_action is False
    assert report.missing_gt == (1, 4, 6)
    assert report.extra_guide == ()
    assert report.coverage == pytest.approx(6 / 9)
    assert report.boundary_accuracy == pytest.approx(1 / 6)
    assert report.strict_passes == 1
    assert report.passed is False  # 1/9 still fails the 7/9 line


def test_scorer_identical_guide_scores_full_marks() -> None:
    # Time alignment: GT against itself pairs
    # every row by overlap, including the generic-only rows the old
    # wording gate dropped (1 Intro, 6/7 internal-battery remove/install).
    gt = _gt_proposed()
    report = score_guide(gt, gt)
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [
        (n, n) for n in range(1, 10)
    ]
    assert report.missing_gt == ()
    assert report.extra_guide == ()
    assert report.coverage == pytest.approx(1.0)
    assert report.boundary_accuracy == pytest.approx(1.0)
    assert report.strict_passes == 9
    assert report.passed is True


def test_scorer_reports_missing_and_extra_separately() -> None:
    # verb-less placeholders ("Alpha phase") never match, so
    # the separation check uses matchable repair wordings instead.
    gt = [
        {"number": 1, "start_s": 0.0, "end_s": 10.0, "action": "Remove bottom cover"},
        {"number": 2, "start_s": 10.0, "end_s": 20.0, "action": "Inspect keyboard"},
    ]
    guide = [
        {"number": 1, "start_s": 0.0, "end_s": 10.0, "action": "Remove bottom cover"},
        {"number": 2, "start_s": 50.0, "end_s": 60.0, "action": "Pry off bottom cover"},
    ]
    report = score_guide(guide, gt)
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [(1, 1)]
    assert report.matches[0].passed is True
    assert report.missing_gt == (2,)
    assert report.extra_guide == (2,)
    assert report.coverage == pytest.approx(0.5)
    assert report.strict_passes == 1
    assert report.passed is False


def test_scorer_new_consensus_replay_score() -> None:
    guide = _replay("align").steps
    report = score_guide(guide, _gt_proposed(), excluded_spans=_gt_excluded())
    # Time alignment: every GT row pairs with
    # its best-overlap guide step, so coverage is 9/9 and the two real
    # timing hits (GT 1/2) score; the rest miss on boundaries, not wording.
    # Tail exclusion (2026-10-29): GT 9 now pairs guide 23 (182-193)
    # instead of guide 24 (193-204.5, zero overlap with 170.4-189.2),
    # and guide 24 vanishes into the excluded span instead of extra.
    assert [(m.guide_number, m.gt_number) for m in report.matches] == [
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 4),
        (9, 5),
        (11, 6),
        (16, 7),
        (18, 8),
        (23, 9),
    ]
    assert [m.passed for m in report.matches] == [
        True, True, False, False, False, False, False, False, False,
    ]
    assert report.missing_gt == ()
    assert report.extra_guide == (
        4, 6, 7, 8, 10, 12, 13, 14, 15, 17, 19, 20, 21, 22, 25,
    )
    assert report.coverage == pytest.approx(1.0)
    assert report.boundary_accuracy == pytest.approx(2 / 9)
    assert report.strict_passes == 2
    assert report.passed is False  # 2/9 still fails the 7/9 line
