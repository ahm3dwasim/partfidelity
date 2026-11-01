"""Step alignment: identical-wording match, transition verbs, phases fallback.

The live spark3 check failed chunk 0 twice: transition phases ("boot
to BIOS", "shut down and flip", "power on") carried no known verb, so
consensus dropped them to singles and contiguity failed closed. These
tests pin the three fixes plus the subsumption the extended vocabulary
needs (an extra transition verb never splits an action; an extra core
verb still does). Fixtures use short paraphrased wordings, never raw
model output from a live run.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.chunks import ChunkWindow, stitch_bundles
from app.contracts import PartRequest, Step, StepBundle
from app.providers import Providers, select_providers
from app.providers.meta_spark import MetaSparkClient
from app.settings import (
    STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR,
    STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR,
    load_provider_settings,
    load_settings,
)
from app.spark import (
    SparkStageConfig,
    StepsDisagreement,
    chunk_bundle_digest,
    consensus_bundle,
    consensus_digest,
    validate_spark_payload,
)
from app.step_alignment import (
    DEFAULT_VERB_LESS_OVERLAP_FRACTION,
    TRANSITION_VERBS,
    Window,
    action_verbs,
    cluster_windows,
    same_action,
)

from test_spark_schema import make_payload, make_step

CONTIGUOUS = SparkStageConfig()  # default span mode is contiguous phases
LEGACY = SparkStageConfig(span_mode="legacy")


def _consensus(step_sets: list[tuple], config: SparkStageConfig = CONTIGUOUS):
    runs = [
        validate_spark_payload(make_payload(steps=steps)) for steps in step_sets
    ]
    return consensus_bundle(runs, config)


def _bundle(steps: list[Step]) -> StepBundle:
    return StepBundle(
        device_guess="gadget",
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


# --- fix 1: identical normalised wordings ------------------------------------


def test_identical_wordings_match_case_and_space_insensitive() -> None:
    assert same_action("Seal  the CASE", "seal the case", phases_mode=True) is True
    assert same_action("seal the case", "Seal  the CASE", phases_mode=True) is True


def test_identical_wordings_need_shared_part_evidence() -> None:
    assert same_action("Battery", "Battery", phases_mode=True) is False
    assert same_action("Remove battery", "Remove battery", phases_mode=True) is (
        False
    )


def test_identical_negated_wordings_kept_and_flagged() -> None:
    assert (
        same_action(
            "Do not remove battery", "Do not remove battery", phases_mode=True
        )
        is False
    )
    step = make_step(
        start=50.0, end=60.0, action="Do not remove battery", parts=(),
    )
    # Time consensus never drops time on wording: the negated triple
    # survives as one flagged segment (the predicate above still
    # refuses to call it a match, so the flag is set).
    (kept,) = _consensus([(dict(step),)] * 3).steps
    assert (kept.start_s, kept.end_s) == (50.0, 60.0)
    assert kept.action == "Do not remove battery"
    assert kept.time_needs_check is True


def test_identical_touching_windows_link_on_tolerance_alone() -> None:
    """Branch (a-text): a boundary dispute over identical verb-less text merges."""
    windows = [
        Window(0.0, 10.0, "Seal the case"),
        Window(10.0, 20.0, "Seal the case"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0, phases_mode=True) == [[0, 1]]


def test_identical_disjoint_parts_still_split_when_touching() -> None:
    windows = [
        Window(0.0, 10.0, "Inspect device", ("cooling fan",)),
        Window(10.0, 20.0, "Inspect device", ("display panel",)),
    ]
    assert cluster_windows(windows, tolerance_s=5.0, phases_mode=True) == [
        [0],
        [1],
    ]


def test_identical_wordings_with_differing_qualifiers_still_split() -> None:
    """the fast path must not skip the qualifier veto."""
    left = frozenset({"left", "fan"})
    right = frozenset({"right", "fan"})
    bare = frozenset({"fan"})
    for phases in (False, True):
        assert (
            same_action(
                "Inspect device", "Inspect device", left, right,
                phases_mode=phases,
            )
            is False
        )
        assert (
            same_action(
                "Inspect device", "Inspect device", left, bare,
                phases_mode=phases,
            )
            is False  # one-sided qualifier still vetoes
        )
        windows = [
            Window(0.0, 10.0, "Inspect device", ("left fan",)),
            Window(10.0, 20.0, "Inspect device", ("right fan",)),
        ]
        assert cluster_windows(
            windows, tolerance_s=5.0, phases_mode=phases
        ) == [[0], [1]]
    assert (
        same_action(
            "Inspect device", "Inspect device", left, left, phases_mode=True
        )
        is True  # positive control: equal qualifiers still merge
    )


# --- fix 2: transition vocabulary, anywhere verbs, subsumption ---------------


@pytest.mark.parametrize(
    "action,expected",
    [
        ("Boot to setup to disable the pack", {"boot", "disable"}),
        ("Power on and disable the pack", {"power on", "disable"}),
        ("Power off the gadget", {"power off"}),
        ("Shut down and flip to base", {"shut", "shut down", "flip"}),
        ("Flip the unit and loosen lid screws", {"flip", "loosen"}),
        ("Turn over the gadget", {"turn", "turn over"}),
        ("Close the lid and test charging", {"close", "test"}),
        ("Locate the lid screws", {"locate"}),
        ("Verify the charge light", {"verify"}),
        ("Shut down and reposition the unit", {"shut", "shut down", "reposition"}),
    ],
)
def test_action_verbs_count_anywhere_in_a_clause(
    action: str, expected: set[str]
) -> None:
    assert action_verbs(action, phases_mode=True) == frozenset(expected)


def test_action_verbs_match_word_boundaries_only() -> None:
    assert action_verbs("Opened the lid", phases_mode=True) == frozenset()
    assert action_verbs("Check the shutter", phases_mode=True) == frozenset(
        {"check"}
    )


def test_subsumption_ignores_transition_only_extras() -> None:
    assert (
        same_action(
            "Flip the unit and loosen lid screws", "Loosen lid screws",
            phases_mode=True,
        )
        is True
    )
    assert (
        same_action(
            "Boot to setup to disable the pack", "Power on to disable the pack",
            phases_mode=True,
        )
        is False  # different transition verbs never merge
    )
    assert (
        same_action("Flip the latch", "Turn the latch", phases_mode=True)
        is False
    )


def test_subsumption_needs_a_nonempty_core() -> None:
    assert same_action("Alpha phase", "Flip the unit", phases_mode=True) is False
    assert same_action("Flip the unit", "Flip the unit", phases_mode=True) is True


def test_subsumption_is_configurable() -> None:
    pair = ("Flip over and loosen lid screws", "Loosen lid screws")
    assert same_action(*pair, phases_mode=True) is True
    assert same_action(*pair, transition_verbs=frozenset(), phases_mode=True) is (
        False
    )


def test_transition_verbs_setting() -> None:
    assert load_provider_settings({}).step_alignment_transition_verbs == (
        TRANSITION_VERBS
    )
    tuned = load_provider_settings(
        {STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR: "Flip, turn over"}
    )
    assert tuned.step_alignment_transition_verbs == frozenset({"flip", "turn over"})
    for bad in ("flip,,turn", "flip;", "re-move,flip", "flip,"):
        with pytest.raises(RuntimeError):
            load_provider_settings({STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR: bad})
    blank = load_provider_settings({STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR: "  "})
    assert blank.step_alignment_transition_verbs == TRANSITION_VERBS


def test_consensus_digest_keys_transition_verbs() -> None:
    tuned = SparkStageConfig(transition_verbs=frozenset({"flip"}))
    assert consensus_digest("j", "v", CONTIGUOUS) != consensus_digest("j", "v", tuned)
    assert consensus_digest("j", "v", tuned) == consensus_digest("j", "v", tuned)
    with pytest.raises(ValueError):
        SparkStageConfig(transition_verbs=frozenset({"not a verb!"}))
    with pytest.raises(ValueError):
        SparkStageConfig(transition_verbs=["flip"])  # type: ignore[arg-type]


# --- legacy/sparse mode keeps the old predicate -----------


def test_legacy_verbs_count_leading_verb_per_clause_only() -> None:
    assert action_verbs("Boot to setup to disable the pack") == frozenset(
        {"boot"}
    )
    assert action_verbs("Pry open the cover") == frozenset({"pry"})
    assert action_verbs("Remove and install the pack") == frozenset(
        {"remove", "install"}
    )


def test_legacy_identical_verbless_wordings_split() -> None:
    assert same_action("Alpha phase", "Alpha phase") is False
    assert same_action("Seal the case", "seal the case") is False
    windows = [
        Window(0.0, 10.0, "Seal the case"),
        Window(10.0, 20.0, "Seal the case"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0) == [[0], [1]]


def test_legacy_needs_strictly_equal_verb_sets() -> None:
    assert (
        same_action("Flip over and loosen lid screws", "Loosen lid screws")
        is False
    )


def test_legacy_consensus_rejects_identical_steps_without_known_verbs() -> None:
    """Sparse/legacy consensus still fails closed on verb-less agreement."""
    step = make_step(
        start=50.0, end=60.0, action="Seal the case", parts=(),
    )
    with pytest.raises(StepsDisagreement):
        _consensus([(dict(step),)] * 3, LEGACY)


# --- fix 3: phases-mode verb-less overlap fallback ----------------------------


def _phases_runs() -> list[tuple]:
    """Three runs tiling [0, 30): verb-less middles, anchored neighbours."""
    middles = ("Alpha phase", "Beta phase", "Gamma phase")
    runs = []
    for middle in middles:
        runs.append(
            (
                make_step(
                    start=0.0, end=10.0, action="Introduce gadget on desk",
                    parts=("gadget lid",),
                ),
                make_step(
                    start=10.0, end=20.0, action=middle, parts=("gadget lid",),
                ),
                make_step(
                    start=20.0, end=30.0, action="Remove base screws",
                    parts=("base screws",),
                ),
            )
        )
    return runs


def test_phases_fallback_keeps_verbless_consensus_step() -> None:
    bundle = _consensus(_phases_runs(), CONTIGUOUS)
    assert [step.action for step in bundle.steps] == [
        "Introduce gadget on desk",
        "Alpha phase",
        "Remove base screws",
    ]


def test_sparse_mode_drops_verbless_singles() -> None:
    bundle = _consensus(_phases_runs(), LEGACY)
    assert [step.action for step in bundle.steps] == [
        "Introduce gadget on desk",
        "Remove base screws",
    ]


def test_fallback_needs_the_configured_fraction() -> None:
    windows = [
        Window(0.0, 10.0, "Alpha phase"),
        Window(6.0, 16.0, "Beta phase"),  # 0.4 of the shorter span
    ]
    assert cluster_windows(windows, tolerance_s=5.0, phases_mode=True) == [
        [0],
        [1],
    ]
    assert cluster_windows(
        windows,
        tolerance_s=5.0,
        phases_mode=True,
        verb_less_overlap_fraction=0.25,
    ) == [[0, 1]]


def test_fallback_needs_mutual_best_overlap() -> None:
    windows = [
        Window(0.0, 10.0, "Alpha phase"),  # best is run 1 (0.5)...
        Window(5.0, 15.0, "Beta phase"),  # ...but run 1's best is run 2
        Window(5.0, 15.0, "Gamma phase"),
    ]
    assert cluster_windows(
        windows, tolerance_s=5.0, phases_mode=True, run_of=[0, 1, 2]
    ) == [[0], [1, 2]]


def test_fallback_vetoes_negation() -> None:
    windows = [
        Window(0.0, 10.0, "Never alpha phase"),
        Window(2.0, 12.0, "Never beta phase"),
    ]
    assert cluster_windows(windows, tolerance_s=5.0, phases_mode=True) == [
        [0],
        [1],
    ]


def test_fallback_vetoes_disjoint_parts() -> None:
    windows = [
        Window(0.0, 10.0, "Alpha phase", ("cooling fan",)),
        Window(2.0, 12.0, "Alpha phase", ("display panel",)),
    ]
    assert cluster_windows(windows, tolerance_s=5.0, phases_mode=True) == [
        [0],
        [1],
    ]


def test_stitch_fallback_follows_span_mode() -> None:
    bundles = [
        (ChunkWindow(0.0, 60.0),
         _bundle([_step(1, 54.0, 60.0, "Alpha phase", "gadget lid")])),
        (ChunkWindow(58.0, 118.0),
         _bundle([_step(1, 0.0, 4.0, "Beta phase", "gadget lid")])),
    ]
    (merged,) = stitch_bundles(bundles, overlap_s=2.0).steps
    assert (merged.start_s, merged.end_s) == (54.0, 62.0)
    legacy = stitch_bundles(bundles, overlap_s=2.0, span_mode="legacy")
    assert [(s.start_s, s.end_s) for s in legacy.steps] == [
        (54.0, 60.0),
        (58.0, 62.0),
    ]


def test_verb_less_overlap_fraction_setting() -> None:
    assert load_provider_settings({}).step_alignment_verb_less_overlap_fraction == (
        DEFAULT_VERB_LESS_OVERLAP_FRACTION
    )
    tuned = load_provider_settings(
        {STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR: "0.25"}
    )
    assert tuned.step_alignment_verb_less_overlap_fraction == 0.25
    for bad in ("0", "0.0", "1.5", "-0.5", "bogus", "nan", "inf"):
        with pytest.raises(RuntimeError):
            load_provider_settings(
                {STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR: bad}
            )
    blank = load_provider_settings(
        {STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR: "  "}
    )
    assert blank.step_alignment_verb_less_overlap_fraction == (
        DEFAULT_VERB_LESS_OVERLAP_FRACTION
    )


def test_consensus_digest_keys_span_mode_and_fraction() -> None:
    assert consensus_digest("j", "v", CONTIGUOUS) != consensus_digest(
        "j", "v", LEGACY
    )
    tuned = SparkStageConfig(verb_less_overlap_fraction=0.25)
    assert consensus_digest("j", "v", CONTIGUOUS) != consensus_digest("j", "v", tuned)
    assert consensus_digest("j", "v", tuned) == consensus_digest("j", "v", tuned)
    for bad in (0, 0.0, 1.5, True, "0.5"):
        with pytest.raises(ValueError):
            SparkStageConfig(verb_less_overlap_fraction=bad)  # type: ignore[arg-type]


def test_chunk_bundle_digest_keys_alignment_knobs() -> None:
    base = {
        "parent_job_id": "parent-1",
        "source_hash": "aaa",
        "index": 0,
        "window_start_s": 0.0,
        "window_end_s": 60.0,
        "cut_start_s": 0.0,
        "cut_duration_s": 60.0,
    }
    first = chunk_bundle_digest(**base)
    assert chunk_bundle_digest(
        **base, config=SparkStageConfig(alignment_mode="exact")
    ) != first
    assert chunk_bundle_digest(
        **base, config=SparkStageConfig(transition_verbs=frozenset({"flip"}))
    ) != first
    assert chunk_bundle_digest(
        **base, config=SparkStageConfig(verb_less_overlap_fraction=0.25)
    ) != first


# --- threading: settings reach the stage config ------------------------------


def test_factory_carries_new_alignment_knobs() -> None:
    settings = load_settings(
        {
            STEP_ALIGNMENT_TRANSITION_VERBS_ENV_VAR: "flip",
            STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION_ENV_VAR: "0.25",
        }
    )
    trio = select_providers(
        settings, replay=Providers(spark=1, sam=2, readback=3)
    )
    config = trio.spark.spark_stage_config
    assert config.transition_verbs == frozenset({"flip"})
    assert config.verb_less_overlap_fraction == 0.25


def test_live_client_carries_new_alignment_knobs() -> None:
    client = MetaSparkClient(
        api_key="k-test",
        base_url="https://api.test/v1",
        model_id="muse-spark-1.3",
        input_price=Decimal("0.00000125"),
        output_price=Decimal("0.00000425"),
        timeout_s=30,
        ctx=None,
        transition_verbs=frozenset({"flip"}),
        verb_less_overlap_fraction=0.25,
    )
    assert client.spark_stage_config.transition_verbs == frozenset({"flip"})
    assert client.spark_stage_config.verb_less_overlap_fraction == 0.25
    defaulted = MetaSparkClient(
        api_key="k-test",
        base_url="https://api.test/v1",
        model_id="muse-spark-1.3",
        input_price=Decimal("0.00000125"),
        output_price=Decimal("0.00000425"),
        timeout_s=30,
        ctx=None,
    )
    assert defaulted.spark_stage_config.transition_verbs == TRANSITION_VERBS
    assert defaulted.spark_stage_config.verb_less_overlap_fraction == (
        DEFAULT_VERB_LESS_OVERLAP_FRACTION
    )
