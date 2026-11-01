"""`cms-synthetic`: diagnosis phrases generated from the graph of a release, with exact gold codes.

Four generators read the classification itself:

- `index`: an Alphabetic Index path read as a phrase (`Pneumonia > aspiration > due to > food` ->
  "aspiration pneumonia due to food"), with the code the term gives and its manifestation code;
- `inclusion`: a Tabular inclusion term of a billable code;
- `neoplasm`: a Neoplasm Table row and column ("malignant neoplasm of lung, upper lobe");
- `drug`: a Table of Drugs and Chemicals row and column ("adverse effect of digoxin").

A code that needs a 7th character gets one at random, and its meaning is added to the phrase
("..., initial encounter"), so the label stays exact. Each phrase then appears as generated
(`base`) and in one perturbed form chosen at random among those that apply: `nonessential` (a
nonessential modifier added, which must not change the code), `word-order` (the Index order or a
reordered inclusion term), `abbreviation` (a common clinical abbreviation) or `typo` (one edit in
one word). Phrases that two generators label differently are dropped as ambiguous.

The phrases follow the classification's own wording, so the set measures regressions and gives a
public benchmark; it says little about real clinical text (docs/evaluation.md).
"""

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import pairwise

from text2icd.eval.datasets import Example, GoldCode, Role, Split
from text2icd.graph.model import Code, Graph, Pointer, Term
from text2icd.ontology.codes import category

SOURCE = "cms-synthetic"
GENERATORS = ("index", "inclusion", "neoplasm", "drug")
PERTURBATIONS = ("base", "nonessential", "word-order", "abbreviation", "typo")
SAMPLE_SIZE = 1000
MIN_PER_STRATUM = 10

# Common clinical abbreviations, used to perturb phrases (the resolver's own lexicon is separate).
ABBREVIATIONS = {
    "acute kidney failure": "AKI",
    "acute myocardial infarction": "AMI",
    "acute respiratory distress syndrome": "ARDS",
    "atrial fibrillation": "AFib",
    "benign prostatic hyperplasia": "BPH",
    "cerebrovascular accident": "CVA",
    "chronic kidney disease": "CKD",
    "chronic obstructive pulmonary disease": "COPD",
    "congestive heart failure": "CHF",
    "coronary artery disease": "CAD",
    "deep vein thrombosis": "DVT",
    "diabetes mellitus": "DM",
    "diabetic ketoacidosis": "DKA",
    "end stage renal disease": "ESRD",
    "gastroesophageal reflux disease": "GERD",
    "gastrointestinal": "GI",
    "heart failure": "HF",
    "human immunodeficiency virus": "HIV",
    "hypertension": "HTN",
    "hypertensive": "HTN",
    "malignant neoplasm": "cancer",
    "multiple sclerosis": "MS",
    "myocardial infarction": "MI",
    "obstructive sleep apnea": "OSA",
    "osteoarthritis": "OA",
    "pulmonary embolism": "PE",
    "rheumatoid arthritis": "RA",
    "systemic lupus erythematosus": "SLE",
    "transient ischemic attack": "TIA",
    "type 1 diabetes mellitus": "T1DM",
    "type 2 diabetes mellitus": "T2DM",
    "upper respiratory infection": "URI",
    "urinary tract infection": "UTI",
    "fracture": "fx",
    "history of": "h/o",
    "secondary to": "2/2",
    "without": "w/o",
    "with": "w/",
    "bilateral": "b/l",
    "left": "L",
    "right": "R",
}
_ABBREVIATION = re.compile(
    r"\b(" + "|".join(sorted(map(re.escape, ABBREVIATIONS), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# Subterms that start with one of these continue the phrase ("due to food", "with coma").
_CONNECTOR_LIST = (
    "with|without|due to|in|of|from|following|complicating|complicated by|affecting|during|"
    "after|and|on|by|involving|resulting from|associated with|secondary to|at|for|to|as|not|"
    "except|causing|or|within|into|between"
)
CONNECTORS = tuple(_CONNECTOR_LIST.split("|"))
_ADJECTIVE_LIST = """
    acute chronic benign malignant congenital acquired primary secondary left right bilateral
    unilateral upper lower male female severe mild moderate old new recurrent simple complete
    incomplete partial total mixed multiple single deep superficial open closed displaced
    nondisplaced intermittent persistent transient specified unspecified other late early
    juvenile senile adult gross minor major small large healed latent active inactive mature
    immature stable unstable newborn first second third fourth degree nontraumatic sleep
    inadequate absent insufficient deficient premature low high short long poor
    """
_ADJECTIVES = frozenset(_ADJECTIVE_LIST.split())
_ADJECTIVE_SUFFIX = re.compile(
    r"(?:al|ic|ous|ary|ive|oid|ile|less|ory|able|ible|ac|form|ed|ar)$", re.IGNORECASE
)
_NOT_ADJECTIVE_LIST = (
    "ovary colic animal canal hospital signal bed history memory category laboratory"
)
_NOT_ADJECTIVES = frozenset(_NOT_ADJECTIVE_LIST.split())
_UNSPECIFIC = re.compile(r"\b(?:NEC|NOS)\b")
_BRACKETS = re.compile(r"\[[^\]]*\]")
_PARENS = re.compile(r"\(([^()]*)\)")
_SPACES = re.compile(r"\s+")

_BEHAVIOUR = {
    "malignant primary": "malignant neoplasm of",
    "malignant secondary": "secondary malignant neoplasm of",
    "ca in situ": "carcinoma in situ of",
    "benign": "benign neoplasm of",
    "uncertain behavior": "neoplasm of uncertain behavior of",
    "unspecified behavior": "neoplasm of unspecified behavior of",
}
_INTENT = {
    "poisoning accidental (unintentional)": "accidental poisoning by {}",
    "poisoning intentional self-harm": "intentional self-poisoning with {}",
    "poisoning assault": "assault by poisoning with {}",
    "poisoning undetermined": "poisoning by {}, undetermined intent",
    "adverse effect": "adverse effect of {}",
    "underdosing": "underdosing of {}",
}


@dataclass(frozen=True, slots=True)
class Seed:
    """A generated phrase before perturbation, with what the perturbations need."""

    key: str
    generator: str
    phrase: str
    gold: tuple[tuple[str, Role], ...]
    source: str
    reordered: str | None = None  # the phrase in another word order
    with_modifier: str | None = None  # the phrase with one nonessential modifier
    seventh: str | None = None


# -- Phrases ---------------------------------------------------------------------------------


def clean(text: str) -> str:
    """Drop NEC and NOS, bracketed synonyms and stray punctuation; one space between words."""
    text = _BRACKETS.sub(" ", text)
    text = _UNSPECIFIC.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    text = re.sub(r"\s+,", ",", text)
    return text.strip(" ,;")


def drop_parentheses(text: str) -> str:
    while _PARENS.search(text):
        text = _PARENS.sub(" ", text)
    return clean(text)


def unwrap_one(text: str, rng: random.Random) -> str | None:
    """Keep the words of one parenthesised group, in place; drop the others."""
    groups = [m for m in _PARENS.finditer(text) if m.group(1).strip() not in ("", "s")]
    if not groups:
        return None
    chosen = rng.choice(groups)
    kept = text[: chosen.start()] + " " + chosen.group(1) + " " + text[chosen.end() :]
    return drop_parentheses(kept)


def lower_first(text: str) -> str:
    """Lowercase a leading capital, but not an acronym (`HIV`), eponym (`Dawson's`) or a word
    with another capital (`McArdle`)."""
    first = text.split(" ", 1)[0]
    if len(first) > 1 and (any(c.isupper() for c in first[1:]) or "'" in first):
        return text
    return text[:1].lower() + text[1:]


def is_connector(text: str) -> bool:
    lowered = text.casefold()
    return any(lowered == c or lowered.startswith(c + " ") for c in CONNECTORS)


def _bare_connector(text: str) -> bool:
    return text.casefold() in CONNECTORS


def is_adjective(text: str) -> bool:
    """A rough test: every word is a known adjective or ends like one (`-al`, `-ic`, `-ous`)."""
    words = text.replace("-", " ").split()
    return bool(words) and all(_adjective_word(word) for word in words)


def _adjective_word(word: str) -> bool:
    lowered = word.casefold().strip("(),")
    if lowered in _NOT_ADJECTIVES:
        return False
    if lowered in _ADJECTIVES or lowered.isdigit():
        return True
    return len(lowered) >= 5 and bool(_ADJECTIVE_SUFFIX.search(lowered))


def _variant(text: str, of: str) -> bool:
    """`diseased` of `Disease`, `aortic` of `aorta`: the same word in another form."""
    n = min(4, len(text), len(of))
    return text.casefold()[:n] == of.casefold()[:n]


def _pieces(text: str) -> list[str]:
    """The comma-separated parts of a title, without the variants of its first part."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return parts[:1] + [part for part in parts[1:] if not _variant(part, parts[0])]


def _join_clause(items: Sequence[str]) -> str:
    """Join a connector clause: `with` > `remission` -> "with remission", and
    `due to` > `drugs` > `rib` -> "due to drugs, rib"."""
    out = items[0]
    for previous, item in pairwise(items):
        out += (" " if _bare_connector(previous) else ", ") + item
    return out


def index_phrase(titles: Sequence[str]) -> str:
    """Read an Index path as a phrase ("aspiration pneumonia due to food"):

    - adjectives go before the main term, the deeper ones first ("benign mesothelioma");
    - eponyms, and short nouns directly under the main term, go next to it ("Brill-Zinsser
      disease", "bile duct calculus");
    - other subterms follow with "of" ("cleft of mitral valve leaflet");
    - subterms led by a connector (`with`, `due to`) and everything under them come last, in
      order ("with cholecystitis, acute").

    A main term that is itself an adjective or an eponym (`Abnormal`, `Hutchinson's`) takes its
    nouns after it, without "of"."""
    main, *subterms = titles
    parts = _pieces(clean(main))
    core = lower_first(parts[0])
    adjectives: list[str] = []
    nouns: list[str] = []
    after: list[str] = []
    for qualifier in parts[1:]:
        if is_connector(qualifier):
            continue
        _place(lower_first(qualifier), True, adjectives, nouns, after)
    clause: list[str] = []
    for depth, title in enumerate(subterms):
        text = clean(title)
        if not text:
            continue
        if clause or is_connector(text):
            clause.append(text)
            continue
        for piece in _pieces(text):
            _place(piece, depth == 0, adjectives, nouns, after)
    describing = is_adjective(core) or core.endswith("'s") or core.isupper()
    head = [*adjectives, core, *nouns] if describing else [*adjectives, *nouns, core]
    phrase = " ".join(head)
    if after:
        phrase += (" " if describing else " of ") + ", ".join(after)
    if clause:
        phrase += " " + _join_clause(clause)
    return phrase


def _place(
    piece: str, top: bool, adjectives: list[str], nouns: list[str], after: list[str]
) -> None:
    if is_adjective(piece):
        adjectives.insert(0, piece)
    elif _eponym(piece) or (top and len(piece.split()) <= 2):
        nouns.append(piece)
    else:
        after.append(piece)


def _eponym(text: str) -> bool:
    """`Brill-Zinsser`, `Eichstedt's`: Index subterms are lowercase unless they name someone."""
    return text[:1].isupper() or "'s" in text


def chart_phrase(titles: Sequence[str]) -> str:
    """The Index order, as problem lists often read: "pneumonia, aspiration, due to food"."""
    main, *subterms = titles
    parts = [", ".join(_pieces(clean(t))) for t in subterms if clean(t)]
    head = lower_first(_pieces(clean(main))[0])
    return head if not parts else f"{head}, {_join_clause(parts)}"


def reorder(phrase: str) -> str | None:
    """Another word order for a free phrase: "X of Y" -> "Y X", "A, B" -> "B A"."""
    if phrase.count(" of ") == 1 and "," not in phrase:
        left, right = phrase.split(" of ")
        return f"{right} {left}"
    if phrase.count(", ") == 1:
        left, right = phrase.split(", ")
        return f"{right} {left}"
    return None


def abbreviate(phrase: str) -> str | None:
    """Replace the longest known long form with its abbreviation, once."""
    matches = list(_ABBREVIATION.finditer(phrase))
    if not matches:
        return None
    match = max(matches, key=lambda m: len(m.group(0)))
    short = ABBREVIATIONS[match.group(0).casefold()]
    return phrase[: match.start()] + short + phrase[match.end() :]


_KEYBOARD = "qwertyuiopasdfghjklzxcvbnm"


def add_typo(phrase: str, rng: random.Random) -> str | None:
    """One edit (swap, deletion, doubling or a neighbouring key) in one word of 5+ letters."""
    words = phrase.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) >= 5 and w.isalpha()]
    if not candidates:
        return None
    i = rng.choice(candidates)
    word = words[i]
    at = rng.randrange(1, len(word) - 1)
    edit = rng.choice(("swap", "delete", "double", "key"))
    if edit == "swap":
        word = word[:at] + word[at + 1] + word[at] + word[at + 2 :]
    elif edit == "delete":
        word = word[:at] + word[at + 1 :]
    elif edit == "double":
        word = word[:at] + word[at] + word[at:]
    else:
        position = _KEYBOARD.find(word[at].lower())
        neighbour = _KEYBOARD[(position + 1) % len(_KEYBOARD)] if position >= 0 else "e"
        word = word[:at] + neighbour + word[at + 1 :]
    if word == words[i]:
        return None
    words[i] = word
    return " ".join(words)


# -- Seeds from the graph --------------------------------------------------------------------


def seeds(graph: Graph, rng_seed: int = 0) -> Iterator[Seed]:
    yield from _index_seeds(graph, rng_seed)
    yield from _inclusion_seeds(graph, rng_seed)
    yield from _table_seeds(graph, "neoplasm", rng_seed)
    yield from _table_seeds(graph, "drug", rng_seed)


def _rng(rng_seed: int, key: str) -> random.Random:
    """A generator per example, so one example's draws do not depend on the others."""
    return random.Random(f"{rng_seed}:{key}")


def _path(graph: Graph, term: Term) -> list[Term]:
    chain = [term]
    while (parent := graph.term_parent(chain[-1].id)) is not None:
        chain.append(parent)
    return chain[::-1]


def _usable(graph: Graph, pointer: Pointer) -> bool:
    return (
        pointer.target is not None
        and not pointer.partial
        and pointer.status in ("ok", "corrected")
        and pointer.target in graph
    )


def _complete(graph: Graph, target: str, rng: random.Random) -> tuple[Code, str | None] | None:
    """The billable code for `target`: itself, or one of its 7th-character codes at random."""
    code = graph.code(target)
    if code.billable:
        return code, None
    generated = [c for c in graph.children(target) if c.kind == "generated" and c.billable]
    if not generated:
        return None
    chosen = rng.choice(generated)
    meanings = dict(graph.seven_char_def(target))
    return chosen, meanings.get(chosen.key[-1])


def _with_modifier(
    chain: Sequence[Term], render: Callable[[Sequence[str]], str], rng: random.Random
) -> str | None:
    """The phrase with one of the term's own nonessential modifiers: an adjective before the
    title (`acute pneumonia`), other words in place (`iliac region`). A main term's modifiers are
    used only for the main term itself, since they may not fit every subterm."""
    term = chain[-1]
    modifiers = [m for m in term.nonessential if m != "s"]
    if not modifiers:
        return None
    modifier = rng.choice(modifiers)
    titles = [t.title for t in chain]
    if is_adjective(modifier):
        titles[-1] = f"{modifier} {lower_first(term.title)}"
    else:
        titles[-1] = drop_parentheses(term.printed_title.replace(f"({modifier})", modifier, 1))
    return render(titles)


def _index_seeds(graph: Graph, rng_seed: int) -> Iterator[Seed]:
    for term in graph.terms("index"):
        pointers = graph.pointers(term.id)
        codes = [p for p in pointers if p.field == "code"]
        manifestations = [p for p in pointers if p.field == "manifestation"]
        if len(codes) != 1 or len(manifestations) > 1 or not _usable(graph, codes[0]):
            continue
        rng = _rng(rng_seed, f"index-{term.id}")
        completed = _complete(graph, str(codes[0].target), rng)
        if completed is None:
            continue
        code, meaning = completed
        gold: list[tuple[str, Role]] = [(code.key, "primary")]
        if manifestations:
            manifestation = manifestations[0]
            if not _usable(graph, manifestation):
                continue
            companion = graph.code(str(manifestation.target))
            if not companion.billable or companion.key == code.key:
                continue
            gold.append((companion.key, "companion"))
        chain = _path(graph, term)
        titles = [t.title for t in chain]
        phrase = index_phrase(titles)
        suffix = f", {meaning}" if meaning else ""
        modified = _with_modifier(chain, index_phrase, rng)
        yield Seed(
            key=f"index-{term.id}",
            generator="index",
            phrase=phrase + suffix,
            gold=tuple(gold),
            source=term.path,
            reordered=chart_phrase(titles) + suffix,
            with_modifier=modified + suffix if modified else None,
            seventh=code.key[-1] if meaning else None,
        )


def _inclusion_seeds(graph: Graph, rng_seed: int) -> Iterator[Seed]:
    for text in graph.tabular_texts():
        if text.kind != "inclusion" or not graph.code(text.node).billable:
            continue
        phrase = lower_first(drop_parentheses(text.text))
        if len(phrase) < 3:
            continue
        rng = _rng(rng_seed, f"inclusion-{text.id}")
        modified = unwrap_one(text.text, rng)
        yield Seed(
            key=f"inclusion-{text.id}",
            generator="inclusion",
            phrase=phrase,
            gold=((graph.code(text.node).key, "primary"),),
            source=text.text,
            reordered=reorder(phrase),
            with_modifier=lower_first(modified) if modified else None,
        )


def _table_seeds(graph: Graph, book: str, rng_seed: int) -> Iterator[Seed]:
    """One column per row, at random, so the large tables do not crowd out the Index."""
    templates = _BEHAVIOUR if book == "neoplasm" else _INTENT
    for term in graph.terms(book):
        chain = _path(graph, term)
        titles = [t.title for t in (chain[1:] if book == "neoplasm" else chain)]
        parts = [", ".join(_pieces(clean(t))) for t in titles if clean(t)]
        if not parts:
            continue
        subject = _join_clause([lower_first(parts[0]), *parts[1:]])
        cells = [
            (pointer, template)
            for pointer in graph.pointers(term.id)
            if pointer.field == "cell"
            and pointer.column is not None
            and _usable(graph, pointer)
            and (template := templates.get(_heading(graph, book, pointer.column))) is not None
        ]
        if not cells:
            continue
        key = f"{book}-{term.id}"
        rng = _rng(rng_seed, key)
        pointer, template = rng.choice(cells)
        completed = _complete(graph, str(pointer.target), rng)
        if completed is None:
            continue
        code, meaning = completed
        phrase = template.format(subject) if "{}" in template else f"{template} {subject}"
        yield Seed(
            key=key,
            generator=book,
            phrase=phrase + (f", {meaning}" if meaning else ""),
            gold=((code.key, "primary"),),
            source=f"{term.path} [{graph.column_heading(book, pointer.column or 0)}]",
            seventh=code.key[-1] if meaning else None,
        )


def _heading(graph: Graph, book: str, column: int) -> str:
    return (graph.column_heading(book, column) or "").casefold()


# -- Examples --------------------------------------------------------------------------------


def _variants(seed: Seed, rng: random.Random) -> dict[str, str]:
    out = {"base": seed.phrase}
    options = {
        "nonessential": seed.with_modifier,
        "word-order": seed.reordered,
        "abbreviation": abbreviate(seed.phrase),
        "typo": add_typo(seed.phrase, rng),
    }
    usable = {name: text for name, text in options.items() if text and text != seed.phrase}
    if usable:
        name = rng.choice(sorted(usable))
        out[name] = usable[name]
    return out


def split_of(code: str, rng_seed: int) -> Split:
    """Half of the categories go to `test`, by a hash, so a category's phrases stay together."""
    digest = hashlib.sha256(f"{rng_seed}:{category(code)}".encode()).digest()
    return "test" if digest[0] < 128 else "dev"


def generate(graph: Graph, rng_seed: int = 0) -> list[Example]:
    """Every example the release gives, in generator order."""
    examples: list[Example] = []
    labels: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for seed in seeds(graph, rng_seed):
        rng = _rng(rng_seed, f"{seed.key}-variants")
        codes = tuple(code for code, _ in seed.gold)
        for perturbation, phrase in _variants(seed, rng).items():
            labels[_SPACES.sub(" ", phrase.casefold())].add(codes)
            meta: dict[str, str] = {
                "generator": seed.generator,
                "perturbation": perturbation,
                "from": seed.source,
            }
            if seed.seventh:
                meta["seventh"] = seed.seventh
            examples.append(
                Example(
                    id=f"{seed.key}-{perturbation}",
                    phrase=phrase,
                    gold=tuple(GoldCode(code=code, role=role) for code, role in seed.gold),
                    release=graph.release,
                    source=SOURCE,
                    split=split_of(seed.gold[0][0], rng_seed),
                    meta=dict(meta),
                )
            )
    seen: set[tuple[str, tuple[str, ...]]] = set()
    kept = []
    for example in examples:
        phrase = _SPACES.sub(" ", example.phrase.casefold())
        codes = tuple(gold.code for gold in example.gold)
        if len(labels[phrase]) > 1 or (phrase, codes) in seen:
            continue
        seen.add((phrase, codes))
        kept.append(example)
    return kept


def sample(
    examples: Sequence[Example], size: int = SAMPLE_SIZE, rng_seed: int = 0
) -> list[Example]:
    """A sample stratified by generator, perturbation and whether there is a companion code, each
    stratum in proportion to its size with at least `MIN_PER_STRATUM`; returned in id order."""
    strata: dict[tuple[str, str, bool], list[Example]] = defaultdict(list)
    for example in examples:
        key = (
            str(example.meta["generator"]),
            str(example.meta["perturbation"]),
            bool(example.companions),
        )
        strata[key].append(example)
    total = len(examples)
    rng = random.Random(f"{rng_seed}:sample")
    chosen: list[Example] = []
    for key in sorted(strata):
        members = strata[key]
        quota = max(MIN_PER_STRATUM, round(size * len(members) / total)) if total else 0
        chosen += rng.sample(members, min(quota, len(members)))
    return sorted(chosen, key=lambda e: e.id)


def describe(examples: Sequence[Example]) -> list[str]:
    """Counts by generator and perturbation."""
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for example in examples:
        counts[str(example.meta["generator"])][str(example.meta["perturbation"])] += 1
    lines = []
    for generator in GENERATORS:
        if generator in counts:
            parts = ", ".join(
                f"{p} {counts[generator][p]:,}" for p in PERTURBATIONS if counts[generator][p]
            )
            lines.append(f"  {generator}: {sum(counts[generator].values()):,} ({parts})")
    return lines
