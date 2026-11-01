"""Deterministic monotonic phase alignment for step samples. fix 2.

Shared by Spark consensus (:mod:`app.spark`) and chunk stitching
(:mod:`app.chunks`) so the two cannot drift: the same link predicate,
phasing, and rival rule. Exact text+parts identity dropped obvious
three-run paraphrases (cover pry-off, internal-battery removal); the link
below aligns monotonically by temporal overlap/order first and treats
the strict action signature as the match rule.

Two samples link only when their windows touch or overlap within
``tolerance_s`` (callers reuse their existing configured line: the
consensus spread, the stitch overlap) AND one wording branch holds:

a. fully equal, matchable signatures (same verbs, qualifiers, and
   parts, at least one verb, at least one shared non-generic part,
   neither negated): tolerance alone suffices, so a boundary dispute
   over identical wordings still merges with its spread flagged;
a-text. contiguous/phases mode only: identical normalised wordings that
   :func:`same_action` accepts (shared part evidence, equal
   qualifiers, vetoes clear): tolerance alone suffices, even with
   no known verb -- a boundary dispute over identical verb-less
   wordings still merges;
b. strict proportional overlap (at least a quarter of the shorter
   window) plus :func:`same_action`: matching normalised verb sets,
   equal qualifier sets, no negation, and a shared non-generic part;
c. phases mode only: both sides verb-less, neither negated, no
   disjoint-parts veto, spans overlapping by at least the configured
   fraction of the shorter one, and each side's best overlap
   (mutual-best, ties allowed, other runs only).

Branch (a) is signature identity, not a text shortcut, and (a-text) is
its verb-less identical-text counterpart: paraphrases still need
their proportional overlap, and negated wordings, differing
qualifiers, or disjoint parts never link even when identical.

:func:`signature` reduces a step to ``(verbs, negated, parts,
qualifiers)``. In contiguous/phases mode verbs are every known verb
anywhere in each conjunction clause (``and``/``or``/``then``/
punctuation split clauses, so ``Remove and install X`` carries two
verbs and ``Boot to BIOS to disable X`` carries ``boot`` plus
``disable``), normalised through the configured synonym groups;
sparse/legacy mode keeps the old rule of each clause's leading
known verb only (``Pry open the cover`` carries ``pry``). Verb sets
match when equal, or -- in phases mode only -- when one side's
extra verbs are all transition verbs (``TRANSITION_VERBS``):
incidental repositioning or power wordings (``flip``, ``turn
over``, ``shut down``, ``boot``, ``power on``) never split an
otherwise identical action, while an extra core verb (``Remove and
install X`` vs ``Install X``) still splits. A wording with no known
verb is noun-only: in phases mode it matches only identical
wordings with shared part evidence (branch (a-text)/(b)), or a
mutual-best overlap partner (branch (c)); in sparse/legacy mode it
never matches, not even identical text. Negation words mark a step
negated and it never matches. Qualifiers (internal/external,
left/right, ...) must be equal as sets: ``Remove internal battery``
and ``Remove external battery`` are different actions, and identical
wordings with differing qualifiers never match either. Parts are
the remaining content tokens from the action and the explicit part
phrases, minus generic nouns, normalised through the configured part
synonym groups (``cable``/``connector``); a pair matches only with a
shared part -- an empty part set on either side never matches, even
for identical wordings.

Phasing is monotonic in time, not connected components: each run's
steps map to phases in non-decreasing phase-time order, a step joins a
phase only when it links every other-run member of that phase, and a
phase needs an other-run anchor (same-run steps never seed a phase
together). The run with the fewest steps pilots the phase skeleton;
more-segmented runs align many-to-one onto it. No bridge merges two
endpoints that do not link directly, and a crossed paraphrase cannot
match backwards in time. A cluster survives consensus only with samples
from a majority of distinct runs (:func:`majority_of`). Stitching keeps
every cluster and merges each into its union. Two surviving clusters
whose windows overlap by more than half of the shorter one are kept and
flagged (:func:`windows_disagree`): honest disagreement, never silent
consensus.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

ALIGNMENT_MODES = ("align", "exact")
DEFAULT_ALIGNMENT_MODE = "align"

# Articles, prepositions, particles, conjunctions, and presentation verbs
# that carry no action meaning. "show" earns its place: overview steps
# share it with unrelated close-ups ("Show laptop overview" vs "Show
# screwdriver tool"), and it must not link them as a noun. "then" joins
# "and"/"or" as a clause separator, never part evidence.
STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "then", "to", "of", "in", "on", "off",
        "out", "up", "down", "over", "for", "with", "from", "at", "by",
        "show", "how",
    }
)

# Generic subject nouns that recur across phases ("Disconnect battery
# cable" vs "Remove internal battery", "Inspect device" fan vs display):
# shared on their own they are not action evidence. Plurals included:
# content_tokens does not stem. "cover" is deliberately NOT generic: it
# is the discriminative subject of the pry/screw/refit phases, told
# apart by their verbs and qualifiers instead. Configured via
# PARTFIDELITY_STEP_ALIGNMENT_GENERIC_TOKENS; this is the default.
GENERIC_TOKENS = frozenset(
    {"battery", "batteries", "device", "devices"}
)

# Negation markers: a step carrying one never matches anything, not
# even identical text ("Do not remove battery" vs "Remove battery").
# Canonical "dont" is "don't" with the apostrophe stripped, as the
# detector strips apostrophes before matching. Configured via
# PARTFIDELITY_STEP_ALIGNMENT_NEGATION_WORDS; this is the default.
NEGATION_WORDS = frozenset(
    {"not", "dont", "do not", "never", "avoid", "without"}
)

# Discriminative qualifiers: two steps match only with equal qualifier
# sets, so "Remove internal battery" never finds "Remove external
# battery". A shared qualifier alone is not part evidence: matching
# also needs a shared non-generic part. Configured via
# PARTFIDELITY_STEP_ALIGNMENT_QUALIFIERS; this is the default.
QUALIFIERS = frozenset(
    {
        "internal", "external",
        "left", "right",
        "top", "bottom", "upper", "lower",
        "front", "back",
        "first", "second", "third",
        "new", "old",
    }
)

_GENERIC_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VERB_RE = re.compile(r"[a-z0-9]+(?: [a-z0-9]+)*")
_APOSTROPHE_RE = re.compile(r"['\u2019]")
_CLAUSE_SPLIT_RE = re.compile(r"\s*(?:,|;|&|\band\b|\bor\b|\bthen\b)\s*")

# Link branch (b): the strict overlap fraction of the shorter window a
# verb-evidence link demands. Keeps adjacent same-topic phases apart
# (touching windows never link on verbs alone). Algorithm constant.
LINK_OVERLAP_DIVISOR = 4  # a quarter of the shorter window

# Verb synonym groups for :func:`signature`. Two steps are the same
# action only with equal normalised verb SETS: groups may overlap
# (``lift`` removes a battery and pries a cover), and an overlapping
# verb normalises to the union of its groups, so it matches only
# itself -- never transitively. Multi-word verbs (``take out``) match
# at the start of a clause. Configured via
# PARTFIDELITY_STEP_ALIGNMENT_VERB_SYNONYMS; this is the default.
VERB_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"remove", "detach", "pull", "take out", "unscrew", "loosen", "lift"}),
    frozenset({"pry", "open", "lift"}),
    frozenset({"disconnect", "unplug"}),
    frozenset(
        {
            "install", "fit", "position", "seat", "place",
            "reinstall", "refit", "reattach",
        }
    ),
    frozenset({"connect", "reconnect", "attach"}),
    frozenset({"inspect", "check"}),
    frozenset({"introduce", "present", "display", "opening", "intro"}),
    frozenset({"show"}),
    frozenset({"disable"}),
    # Transition verbs: the contiguous prompt names
    # phase boundaries the old lexicon could not see ("boot to BIOS",
    # "shut down and flip", "close up", "power on"). ``power on`` and
    # ``power off`` stay apart: opposite actions never merge.
    # ``flip`` and ``turn`` stay apart too: a latch flip is not a knob
    # turn, and only shared part evidence could claim otherwise.
    frozenset({"boot"}),
    frozenset({"power on"}),
    frozenset({"power off"}),
    frozenset({"shut", "shut down"}),
    frozenset({"flip"}),
    frozenset({"turn", "turn over"}),
    frozenset({"close"}),
    frozenset({"locate"}),
    frozenset({"test", "verify"}),
    frozenset({"reposition"}),
)
# Note: ``show`` is a singleton group. It is a verb (identical ``show``
# wordings still link with part evidence), but it is not an
# introduction synonym: a screwdriver close-up that lists the laptop lid
# as a part would otherwise merge into the intro overview. ``disable``
# likewise matches only itself (the BIOS step names no synonym).

# Transition verbs for verb-set subsumption: repositioning and power
# wordings incidental to the core action. When two verb sets differ
# ONLY by transition verbs on one side (``Flip over and loosen lid
# screws`` vs ``Loosen lid screws``), the extra transition verbs are
# subsumed and the sets match; an extra core verb (``Remove and
# install X`` vs ``Install X``) still splits. ``open``, ``close``,
# ``locate``, ``test``, and ``verify`` are core manipulations, not
# transitions. Configured via
# PARTFIDELITY_STEP_ALIGNMENT_TRANSITION_VERBS; this is the default.
# A transition word outside the active verb lexicon is inert: it never
# matches an extra because the lexicon never yields it.
TRANSITION_VERBS = frozenset(
    {
        "boot",
        "power on", "power off",
        "shut", "shut down",
        "flip",
        "turn", "turn over",
        "reposition",
    }
)

# Branch (c): the default overlap fraction of the shorter window two
# verb-less phases need in phases mode (each other's best overlap).
# Configured via
# PARTFIDELITY_STEP_ALIGNMENT_VERBLESS_OVERLAP_FRACTION; this is the
# default.
DEFAULT_VERB_LESS_OVERLAP_FRACTION = 0.5

# Part synonym groups for :func:`signature`. Parts normalise the same
# way verbs do: ``cable`` and ``connector`` are the same subject, so
# "Disconnect battery cable" finds "Unplug battery connector".
# Configured via PARTFIDELITY_STEP_ALIGNMENT_PART_SYNONYMS; this is the
# default.
PART_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"cable", "connector"}),
)


@dataclass(frozen=True)
class Window:
    """One alignable step window: times, action, and part phrases."""

    start_s: float
    end_s: float
    action: str
    part_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionSignature:
    """The strict match identity of one step.

    ``verbs`` and ``parts`` hold canonical forms (the union of every
    synonym group containing the word, or the word alone): equality is
    exact set equality, never intersection. ``qualifiers`` are raw
    qualifier words, likewise compared as equal sets.
    """

    verbs: frozenset[frozenset[str]]
    negated: bool
    parts: frozenset[frozenset[str]]
    qualifiers: frozenset[str]


def window_of(step: object) -> Window:
    """Adapt a validated run step or consensus step to a :class:`Window`."""
    parts = getattr(step, "parts")
    return Window(
        start_s=getattr(step, "start_s"),
        end_s=getattr(step, "end_s"),
        action=getattr(step, "action"),
        part_phrases=tuple(getattr(part, "phrase") for part in parts),
    )


def normalize_action(action: str) -> str:
    """Lowercase whitespace-normalized action text."""
    return " ".join(action.strip().lower().split())


def content_tokens(text: str) -> frozenset[str]:
    """Lowercase alphanumeric tokens minus stopwords."""
    token: list[str] = []
    found: set[str] = set()
    for char in text.lower():
        if char.isalnum():
            token.append(char)
        elif token:
            found.add("".join(token))
            token = []
    if token:
        found.add("".join(token))
    return frozenset(word for word in found if word not in STOPWORDS)


def part_tokens(phrases: Iterable[str]) -> frozenset[str]:
    """Union of content tokens across every part phrase."""
    merged: set[str] = set()
    for phrase in phrases:
        merged |= content_tokens(phrase)
    return frozenset(merged)


def normalize_generic_tokens(values: Iterable[str]) -> frozenset[str]:
    """Validate generic tokens: lowercase alphanumeric, returned as a set."""
    items: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"generic token must be a string, got {value!r}")
        text = value.strip().lower()
        if _GENERIC_TOKEN_RE.fullmatch(text) is None:
            raise ValueError(f"generic token must match [a-z0-9]+, got {value!r}")
        items.append(text)
    return frozenset(items)


def normalize_negation_words(values: Iterable[str]) -> frozenset[str]:
    """Validate negation words: apostrophes stripped, one phrase each."""
    items: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"negation word must be a string, got {value!r}")
        text = _APOSTROPHE_RE.sub("", value.strip().lower())
        text = " ".join(text.split())
        if _VERB_RE.fullmatch(text) is None:
            raise ValueError(
                f"negation word must match [a-z0-9]+( [a-z0-9]+)*, got {value!r}"
            )
        items.append(text)
    if not items:
        raise ValueError("negation words must hold at least one entry")
    return frozenset(items)


def normalize_qualifiers(values: Iterable[str]) -> frozenset[str]:
    """Validate qualifiers: lowercase alphanumeric, returned as a set."""
    items: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"qualifier must be a string, got {value!r}")
        text = value.strip().lower()
        if _GENERIC_TOKEN_RE.fullmatch(text) is None:
            raise ValueError(f"qualifier must match [a-z0-9]+, got {value!r}")
        items.append(text)
    if not items:
        raise ValueError("qualifiers must hold at least one entry")
    return frozenset(items)


def _normalize_word_groups(
    groups: tuple[frozenset[str], ...] | list[frozenset[str]],
    *,
    kind: str,
    allow_phrases: bool,
) -> tuple[frozenset[str], ...]:
    """Validate synonym groups: tuple/list of non-empty word sets.

    Anything misshapen fails loudly so a misspelled override can never
    silently match nothing. Groups may overlap; matching normalises
    each word to the union of its groups, never transitively.
    """
    if isinstance(groups, (str, bytes)) or not isinstance(groups, (tuple, list)):
        raise ValueError(
            f"{kind} synonyms must be a tuple/list of sets,"
            f" got {type(groups).__name__}"
        )
    normalized: list[frozenset[str]] = []
    for group in groups:
        if isinstance(group, (str, bytes)) or not isinstance(group, (set, frozenset)):
            raise ValueError(
                f"{kind} synonym groups must be sets,"
                f" got {type(group).__name__}"
            )
        words: set[str] = set()
        for word in group:
            if not isinstance(word, str):
                raise ValueError(f"{kind} must be a string, got {word!r}")
            text = " ".join(word.strip().lower().split())
            pattern = _VERB_RE if allow_phrases else _GENERIC_TOKEN_RE
            if pattern.fullmatch(text) is None:
                expected = (
                    "[a-z0-9]+( [a-z0-9]+)*"
                    if allow_phrases
                    else "[a-z0-9]+"
                )
                raise ValueError(f"{kind} must match {expected}, got {word!r}")
            words.add(text)
        if not words:
            raise ValueError(f"{kind} synonym group must not be empty")
        normalized.append(frozenset(words))
    if not normalized:
        raise ValueError(f"{kind} synonyms must hold at least one group")
    return tuple(normalized)


def normalize_verb_synonyms(
    groups: tuple[frozenset[str], ...] | list[frozenset[str]],
) -> tuple[frozenset[str], ...]:
    """Validate verb synonym groups (multi-word verbs allowed)."""
    return _normalize_word_groups(groups, kind="verb", allow_phrases=True)


def normalize_part_synonyms(
    groups: tuple[frozenset[str], ...] | list[frozenset[str]],
) -> tuple[frozenset[str], ...]:
    """Validate part synonym groups (single words only)."""
    return _normalize_word_groups(groups, kind="part", allow_phrases=False)


def normalize_transition_verbs(values: Iterable[str]) -> frozenset[str]:
    """Validate transition verbs: verb-shaped words, returned as a set.

    Empty is allowed (subsumption off); anything misshapen fails
    loudly so a misspelled override can never silently match nothing.
    """
    items: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"transition verb must be a string, got {value!r}")
        text = " ".join(value.strip().lower().split())
        if _VERB_RE.fullmatch(text) is None:
            raise ValueError(
                "transition verb must match [a-z0-9]+( [a-z0-9]+)*,"
                f" got {value!r}"
            )
        items.append(text)
    return frozenset(items)


def normalize_verb_less_overlap_fraction(value: float) -> float:
    """Validate the phases-mode verb-less overlap fraction: 0 < f <= 1.

    Zero would link mutual-best phases that never overlap; above one
    nothing could ever link. Anything outside fails loudly.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 1
    ):
        raise ValueError(
            "verb-less overlap fraction must be a finite number"
            f" with 0 < f <= 1, got {value!r}"
        )
    return float(value)


def _clauses(action: str) -> tuple[str, ...]:
    """Split an action into conjunction clauses (``and``/``or``/``then``/...)."""
    return tuple(
        piece
        for piece in (
            clause.strip() for clause in _CLAUSE_SPLIT_RE.split(action)
        )
        if piece
    )


def action_verbs(
    action: str,
    verb_synonyms: tuple[frozenset[str], ...] = VERB_SYNONYM_GROUPS,
    *,
    phases_mode: bool = False,
) -> frozenset[str]:
    """The verbs an action carries.

    In contiguous/phases mode each conjunction clause contributes
    every synonym-group verb its words hold contiguously
    (``connect`` in ``Screw down and connect battery``, ``disable``
    in ``Boot to BIOS to disable X``). Matching is word-boundary
    based, so ``open`` never fires on ``opened``; overlapping verbs
    both count (``turn`` and ``turn over``), which is harmless
    because matching compares canonical groups, not raw verbs.
    Sparse/legacy mode keeps the old rule: each clause contributes
    its leading known verb only (longer verbs win over prefixes), so
    a mid-clause group verb is a particle, not a verb (``open`` in
    ``Pry open the cover``). Unknown words are nouns, never verbs --
    so noun-only steps carry none.
    """
    if not isinstance(phases_mode, bool):
        raise ValueError(
            f"phases_mode must be a bool, got {type(phases_mode).__name__}"
        )
    groups = normalize_verb_synonyms(verb_synonyms)
    if not phases_mode:
        verbs = sorted(
            {verb for group in groups for verb in group},
            key=len,
            reverse=True,
        )
        found: set[str] = set()
        for clause in _clauses(action):
            lowered = " ".join(clause.strip().lower().split())
            for verb in verbs:
                if lowered == verb or lowered.startswith(verb + " "):
                    found.add(verb)
                    break
        return frozenset(found)
    verbs = sorted({verb for group in groups for verb in group})
    found = set()
    for clause in _clauses(action):
        words = _GENERIC_TOKEN_RE.findall(clause.lower())
        for verb in verbs:
            parts = verb.split(" ")
            width = len(parts)
            if any(
                words[start:start + width] == parts
                for start in range(len(words) - width + 1)
            ):
                found.add(verb)
    return frozenset(found)


def is_negated(
    action: str,
    negation_words: frozenset[str] = NEGATION_WORDS,
) -> bool:
    """True when the action carries a negation word or phrase."""
    words = normalize_negation_words(negation_words)
    stripped = _APOSTROPHE_RE.sub("", action.lower())
    tokens = set(_GENERIC_TOKEN_RE.findall(stripped))
    flat = " ".join(_GENERIC_TOKEN_RE.findall(stripped))
    for entry in words:
        if " " in entry:
            if re.search(r"\b%s\b" % re.escape(entry), flat):
                return True
        elif entry in tokens:
            return True
    return False


def _canonical_member(
    word: str, groups: tuple[frozenset[str], ...]
) -> frozenset[str]:
    """The word's canonical form: the union of its groups, or itself."""
    pooled = frozenset().union(*(group for group in groups if word in group))
    return pooled if pooled else frozenset({word})


def signature(
    action: str,
    parts: frozenset[str] = frozenset(),
    *,
    generic_tokens: frozenset[str] = GENERIC_TOKENS,
    verb_synonyms: tuple[frozenset[str], ...] = VERB_SYNONYM_GROUPS,
    negation_words: frozenset[str] = NEGATION_WORDS,
    qualifiers: frozenset[str] = QUALIFIERS,
    part_synonyms: tuple[frozenset[str], ...] = PART_SYNONYM_GROUPS,
    phases_mode: bool = False,
) -> ActionSignature:
    """Reduce a step to its strict match identity.

    ``parts`` is the step's explicit part tokens (see :func:`part_tokens`),
    unioned with the action's own content tokens; verbs, qualifiers,
    generic nouns, and negation words are not parts. ``phases_mode``
    selects the contiguous verb detection (every known verb anywhere
    in each clause); sparse/legacy keeps each clause's leading verb.
    """
    generic = normalize_generic_tokens(generic_tokens)
    groups = normalize_verb_synonyms(verb_synonyms)
    negs = normalize_negation_words(negation_words)
    quals = normalize_qualifiers(qualifiers)
    psyns = normalize_part_synonyms(part_synonyms)
    found_verbs = action_verbs(action, groups, phases_mode=phases_mode)
    verb_words = {word for verb in found_verbs for word in verb.split()}
    neg_words = {word for entry in negs for word in entry.split()}
    tokens = content_tokens(action) | set(parts)
    qualifier_set = frozenset(token for token in tokens if token in quals)
    part_words = frozenset(
        token
        for token in tokens
        if token not in verb_words
        and token not in quals
        and token not in generic
        and token not in neg_words
    )
    return ActionSignature(
        verbs=frozenset(_canonical_member(verb, groups) for verb in found_verbs),
        negated=is_negated(action, negs),
        parts=frozenset(_canonical_member(word, psyns) for word in part_words),
        qualifiers=qualifier_set,
    )


def _verbs_match(
    first_verbs: frozenset[frozenset[str]],
    second_verbs: frozenset[frozenset[str]],
    transition_groups: frozenset[frozenset[str]],
) -> bool:
    """True when two canonical verb sets are the same action's verbs.

    Equal non-empty sets match; otherwise the smaller set must be a
    non-empty strict subset and every extra group transition-only, so
    ``Flip over and loosen X`` finds ``Loosen X`` while ``Remove and
    install X`` never finds ``Install X`` (an extra core verb splits).
    """
    if first_verbs == second_verbs:
        return bool(first_verbs)
    if len(first_verbs) < len(second_verbs):
        smaller, larger = first_verbs, second_verbs
    else:
        smaller, larger = second_verbs, first_verbs
    return (
        bool(smaller)
        and smaller < larger
        and larger - smaller <= transition_groups
    )


def same_action(
    first_action: str,
    second_action: str,
    first_parts: frozenset[str] = frozenset(),
    second_parts: frozenset[str] = frozenset(),
    *,
    generic_tokens: frozenset[str] = GENERIC_TOKENS,
    verb_synonyms: tuple[frozenset[str], ...] = VERB_SYNONYM_GROUPS,
    negation_words: frozenset[str] = NEGATION_WORDS,
    qualifiers: frozenset[str] = QUALIFIERS,
    part_synonyms: tuple[frozenset[str], ...] = PART_SYNONYM_GROUPS,
    transition_verbs: frozenset[str] = TRANSITION_VERBS,
    phases_mode: bool = False,
) -> bool:
    """True when two steps are the same action under the strict signature.

    The one shared predicate used by the aligner (:func:`cluster_windows`
    via :func:`_linked`) and the timing scorer. Disjoint explicit parts
    veto (same words, different subject); neither side negated. In
    contiguous/phases mode, identical normalised wordings with a
    shared part and equal qualifier sets match without needing a
    verb; otherwise both sides carry at least one verb and the
    normalised verb SETS match (equal, or differing only by
    transition verbs -- a compound ``Remove and install X`` still
    never matches ``Install X``). In sparse/legacy mode both sides
    carry at least one verb (noun-only steps never match, not even
    identical text) and the verb SETS are strictly equal. In both
    modes the qualifier sets are equal (``internal`` never matches
    ``external``, even for identical wordings); and the sides share
    at least one non-generic part -- an empty part set on either side
    never matches, even for identical wordings (``Remove battery``
    never finds ``Remove device``). Conservative throughout: when
    unsure, no match.
    """
    if not isinstance(phases_mode, bool):
        raise ValueError(
            f"phases_mode must be a bool, got {type(phases_mode).__name__}"
        )
    if first_parts and second_parts and not (first_parts & second_parts):
        return False  # disjoint parts veto: same words, different subject
    first = signature(
        first_action, first_parts,
        generic_tokens=generic_tokens,
        verb_synonyms=verb_synonyms,
        negation_words=negation_words,
        qualifiers=qualifiers,
        part_synonyms=part_synonyms,
        phases_mode=phases_mode,
    )
    second = signature(
        second_action, second_parts,
        generic_tokens=generic_tokens,
        verb_synonyms=verb_synonyms,
        negation_words=negation_words,
        qualifiers=qualifiers,
        part_synonyms=part_synonyms,
        phases_mode=phases_mode,
    )
    if first.negated or second.negated:
        return False
    if not phases_mode:
        if not first.verbs or not second.verbs:
            return False  # legacy: noun-only steps never match
        if first.verbs != second.verbs:
            return False  # legacy: strictly equal verb sets
    else:
        if (
            normalize_action(first_action) == normalize_action(second_action)
            and first.parts & second.parts
            and first.qualifiers == second.qualifiers
        ):
            return True  # identical wordings: shared part, equal qualifiers
        groups = normalize_verb_synonyms(verb_synonyms)
        transition_groups = frozenset(
            _canonical_member(word, groups)
            for word in normalize_transition_verbs(transition_verbs)
        )
        if not _verbs_match(first.verbs, second.verbs, transition_groups):
            return False
    if first.qualifiers != second.qualifiers:
        return False
    return bool(first.parts & second.parts)


def majority_of(runs: int) -> int:
    """Distinct-run majority for ``runs`` runs (derived, never configured)."""
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError(f"runs must be an int >= 1, got {runs!r}")
    return runs // 2 + 1


def windows_disagree(
    first_start: float, first_end: float, second_start: float, second_end: float
) -> bool:
    """True when the windows overlap by more than half of the shorter one."""
    overlap = min(first_end, second_end) - max(first_start, second_start)
    shorter = min(first_end - first_start, second_end - second_start)
    return overlap > shorter / 2


def _signature_equal(
    first_action: str,
    second_action: str,
    first_parts: frozenset[str],
    second_parts: frozenset[str],
    *,
    generic_tokens: frozenset[str],
    verb_synonyms: tuple[frozenset[str], ...],
    negation_words: frozenset[str],
    qualifiers: frozenset[str],
    part_synonyms: tuple[frozenset[str], ...],
    phases_mode: bool,
) -> bool:
    """True for fully equal, matchable signatures (branch (a) identity).

    Matchable needs a real part: equal signatures share their part
    set, so a nonempty one is the shared non-generic part evidence;
    generic-only wordings never link, even when identical.
    """
    first = signature(
        first_action, first_parts,
        generic_tokens=generic_tokens,
        verb_synonyms=verb_synonyms,
        negation_words=negation_words,
        qualifiers=qualifiers,
        part_synonyms=part_synonyms,
        phases_mode=phases_mode,
    )
    if first.negated or not first.verbs or not first.parts:
        return False
    return first == signature(
        second_action, second_parts,
        generic_tokens=generic_tokens,
        verb_synonyms=verb_synonyms,
        negation_words=negation_words,
        qualifiers=qualifiers,
        part_synonyms=part_synonyms,
        phases_mode=phases_mode,
    )


def _linked(
    first: Window,
    second: Window,
    first_parts: frozenset[str],
    second_parts: frozenset[str],
    tolerance_s: float,
    generic_tokens: frozenset[str],
    verb_synonyms: tuple[frozenset[str], ...],
    negation_words: frozenset[str],
    qualifiers: frozenset[str],
    part_synonyms: tuple[frozenset[str], ...],
    transition_verbs: frozenset[str],
    phases_mode: bool,
) -> bool:
    gap = max(first.start_s, second.start_s) - min(first.end_s, second.end_s)
    if gap > tolerance_s:
        return False
    if _signature_equal(
        first.action, second.action, first_parts, second_parts,
        generic_tokens=generic_tokens,
        verb_synonyms=verb_synonyms,
        negation_words=negation_words,
        qualifiers=qualifiers,
        part_synonyms=part_synonyms,
        phases_mode=phases_mode,
    ):
        return True  # matchable identical signatures link on tolerance alone
    if phases_mode and normalize_action(first.action) == normalize_action(
        second.action
    ):
        # Identical wordings link on tolerance alone once the shared
        # predicate accepts them (shared parts, equal qualifiers,
        # vetoes clear) -- even with no known verb. Paraphrases
        # still need branch (b); sparse mode has no (a-text) branch.
        return same_action(
            first.action, second.action, first_parts, second_parts,
            generic_tokens=generic_tokens,
            verb_synonyms=verb_synonyms,
            negation_words=negation_words,
            qualifiers=qualifiers,
            part_synonyms=part_synonyms,
            transition_verbs=transition_verbs,
            phases_mode=phases_mode,
        )
    if gap >= 0:
        return False  # paraphrases must strictly overlap, not merely touch
    shorter = min(first.end_s - first.start_s, second.end_s - second.start_s)
    if -gap * LINK_OVERLAP_DIVISOR < shorter:
        return False  # paraphrases still need a proportional overlap
    return same_action(
        first.action, second.action, first_parts, second_parts,
        generic_tokens=generic_tokens,
        verb_synonyms=verb_synonyms,
        negation_words=negation_words,
        qualifiers=qualifiers,
        part_synonyms=part_synonyms,
        transition_verbs=transition_verbs,
        phases_mode=phases_mode,
    )


def _overlap_fraction(first: Window, second: Window) -> float:
    """The windows' overlap as a fraction of the shorter one (<= 0 apart)."""
    overlap = min(first.end_s, second.end_s) - max(first.start_s, second.start_s)
    shorter = min(first.end_s - first.start_s, second.end_s - second.start_s)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _mutual_best_overlaps(
    windows: Sequence[Window],
    owner: Sequence[int],
    verb_less: Sequence[bool],
    fraction: float,
) -> set[tuple[int, int]]:
    """Mutual-best overlap pairs among verb-less other-run windows.

    Each verb-less window's best partners are the other-run verb-less
    windows with the top overlap fraction (ties all count); a pair is
    mutual-best when each side holds the other, and only when the top
    fraction reaches ``fraction``. Deterministic index-order pairs.
    """
    best: list[set[int]] = [set() for _ in windows]
    for index, window in enumerate(windows):
        if not verb_less[index]:
            continue
        top = fraction
        for other in range(len(windows)):
            if (
                owner[other] == owner[index]
                or not verb_less[other]
            ):
                continue
            score = _overlap_fraction(window, windows[other])
            if score > top:
                top = score
                best[index] = {other}
            elif score == top and score >= fraction:
                best[index].add(other)
    mutual: set[tuple[int, int]] = set()
    for index in range(len(windows)):
        for other in best[index]:
            if index in best[other]:
                mutual.add((min(index, other), max(index, other)))
    return mutual


def cluster_windows(
    windows: Sequence[Window],
    *,
    tolerance_s: float,
    generic_tokens: frozenset[str] = GENERIC_TOKENS,
    verb_synonyms: tuple[frozenset[str], ...] = VERB_SYNONYM_GROUPS,
    negation_words: frozenset[str] = NEGATION_WORDS,
    qualifiers: frozenset[str] = QUALIFIERS,
    part_synonyms: tuple[frozenset[str], ...] = PART_SYNONYM_GROUPS,
    transition_verbs: frozenset[str] = TRANSITION_VERBS,
    phases_mode: bool = False,
    verb_less_overlap_fraction: float = DEFAULT_VERB_LESS_OVERLAP_FRACTION,
    run_of: Sequence[int] | None = None,
) -> list[list[int]]:
    """Group window indices into monotonic phases.

    Each run's steps (``run_of[i]``, in index order) map to phases in
    non-decreasing phase-time order; a step joins the earliest phase
    whose other-run members start at or after its run's latest joined
    phase time and which it links every other-run member of. Same-run
    steps never seed a phase together (many-to-one segmentation splits
    still share a phase once an other-run window anchors it). A step
    that joins no phase leaves its run's floor where it was, so an
    extra step a finer run inserts never strands that run's later steps
    behind it. With no ``run_of`` every window is its own run.
    In phases mode, verb-less windows also link their mutual-best
    overlap partner (branch (c)); sparse mode never does, and sparse
    mode matches with the legacy predicate throughout (each clause's
    leading verb, strictly equal verb sets, noun-only never matches).
    Deterministic: phases list in creation order, members in index order.
    """
    generic = normalize_generic_tokens(generic_tokens)
    if not isinstance(generic_tokens, frozenset):
        raise ValueError(
            f"generic_tokens must be a frozenset, got {type(generic_tokens).__name__}"
        )
    synonyms = normalize_verb_synonyms(verb_synonyms)
    if not isinstance(verb_synonyms, tuple) or any(
        not isinstance(group, frozenset) for group in verb_synonyms
    ):
        raise ValueError(
            "verb_synonyms must be a tuple of frozensets,"
            f" got {type(verb_synonyms).__name__}"
        )
    negs = normalize_negation_words(negation_words)
    if not isinstance(negation_words, frozenset):
        raise ValueError(
            "negation_words must be a frozenset,"
            f" got {type(negation_words).__name__}"
        )
    quals = normalize_qualifiers(qualifiers)
    if not isinstance(qualifiers, frozenset):
        raise ValueError(
            f"qualifiers must be a frozenset, got {type(qualifiers).__name__}"
        )
    psyns = normalize_part_synonyms(part_synonyms)
    if not isinstance(part_synonyms, tuple) or any(
        not isinstance(group, frozenset) for group in part_synonyms
    ):
        raise ValueError(
            "part_synonyms must be a tuple of frozensets,"
            f" got {type(part_synonyms).__name__}"
        )
    trans = normalize_transition_verbs(transition_verbs)
    if not isinstance(transition_verbs, frozenset):
        raise ValueError(
            "transition_verbs must be a frozenset,"
            f" got {type(transition_verbs).__name__}"
        )
    if not isinstance(phases_mode, bool):
        raise ValueError(
            f"phases_mode must be a bool, got {type(phases_mode).__name__}"
        )
    fraction = normalize_verb_less_overlap_fraction(verb_less_overlap_fraction)
    if run_of is None:
        owner = list(range(len(windows)))
    else:
        owner = list(run_of)
        if len(owner) != len(windows):
            raise ValueError(
                f"run_of has {len(owner)} entries for {len(windows)} windows"
            )
        for entry in owner:
            if isinstance(entry, bool) or not isinstance(entry, int) or entry < 0:
                raise ValueError(f"run_of entries must be ints >= 0, got {entry!r}")
    by_run: dict[int, list[int]] = {}
    for index, run in enumerate(owner):
        by_run.setdefault(run, []).append(index)
    # The least-segmented run pilots the phase skeleton; ties go to the
    # lowest run id. Deterministic, and order-independent for the common
    # one-coarse/many-fine shape.
    run_ids = sorted(by_run, key=lambda run: (len(by_run[run]), run))
    parts = [part_tokens(window.part_phrases) for window in windows]
    verb_less = [
        not action_verbs(window.action, synonyms, phases_mode=phases_mode)
        for window in windows
    ]
    negated = [is_negated(window.action, negs) for window in windows]
    mutual = (
        _mutual_best_overlaps(windows, owner, verb_less, fraction)
        if phases_mode
        else set()
    )

    def linked(i: int, j: int) -> bool:
        if _linked(
            windows[i], windows[j],
            parts[i], parts[j],
            tolerance_s=tolerance_s,
            generic_tokens=generic,
            verb_synonyms=synonyms,
            negation_words=negs,
            qualifiers=quals,
            part_synonyms=psyns,
            transition_verbs=trans,
            phases_mode=phases_mode,
        ):
            return True
        if not phases_mode:
            return False
        # Branch (c): verb-less mutual-best overlap. Negation and
        # disjoint explicit parts still veto, as everywhere.
        pair = (min(i, j), max(i, j))
        if pair not in mutual:
            return False
        if negated[i] or negated[j]:
            return False
        return not (
            parts[i] and parts[j] and not (parts[i] & parts[j])
        )

    phases: list[list[int]] = []
    starts = [window.start_s for window in windows]
    for run in run_ids:
        floor = float("-inf")
        for index in by_run[run]:
            placed = False
            for phase in range(len(phases)):
                others = [m for m in phases[phase] if owner[m] != run]
                if not others:
                    continue  # same-run steps never seed a phase together
                earliest = min(starts[m] for m in others)
                if earliest < floor:
                    continue  # monotonic: never match backwards in time
                if all(linked(index, m) for m in others):
                    phases[phase].append(index)
                    floor = earliest
                    placed = True
                    break
            if not placed:
                phases.append([index])
    return [sorted(members) for members in phases]


__all__ = (
    "ALIGNMENT_MODES",
    "DEFAULT_ALIGNMENT_MODE",
    "DEFAULT_VERB_LESS_OVERLAP_FRACTION",
    "GENERIC_TOKENS",
    "LINK_OVERLAP_DIVISOR",
    "NEGATION_WORDS",
    "PART_SYNONYM_GROUPS",
    "QUALIFIERS",
    "STOPWORDS",
    "TRANSITION_VERBS",
    "VERB_SYNONYM_GROUPS",
    "ActionSignature",
    "Window",
    "action_verbs",
    "cluster_windows",
    "content_tokens",
    "is_negated",
    "majority_of",
    "normalize_action",
    "normalize_generic_tokens",
    "normalize_negation_words",
    "normalize_part_synonyms",
    "normalize_qualifiers",
    "normalize_transition_verbs",
    "normalize_verb_less_overlap_fraction",
    "normalize_verb_synonyms",
    "part_tokens",
    "same_action",
    "signature",
    "window_of",
    "windows_disagree",
)
