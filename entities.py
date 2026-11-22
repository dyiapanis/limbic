"""
Limbic entity extraction — NLP named entity recognition.

Extracts named entities from fact content using spaCy NER + custom
EntityRuler for deployment-specific entities. Entities are stored on facts
and in a separate entities table for graph traversal.

Language handling:
  - Pipelines are loaded for the languages configured in limbic.yaml
    (nlp.languages, e.g. ["en"] — language-neutral plugin default).
  - Language detection routes per-content via Unicode script ranges
    (no word lists, no hardcoded language pairs).
  - Unconfigured-language content falls back to the first loaded pipeline.

Vendored pipeline: the `en` model ships inside this plugin at
`models/en_core_web_sm/` (spaCy's own MIT-licensed wheel, unpacked — see
`models/en_core_web_sm/LICENSE`). Nothing to install: entity extraction works
on a fresh install, offline, with no package manager involvement.

Other languages (fr, de, el …) are still opt-in and are NOT bundled:
`hermes limbic languages --set <langs>` saves the config and downloads each
model into the dependency environment. A language whose model is missing
degrades silently to no extraction for that language — reflected in provider
status (`entity_extraction`) and repairable post-hoc via `entities-backfill`.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── NER junk filter ────────────────────────────────────────────────────
#
# Junk classes never leave extraction: pure numbers/units (NE labels and
# no-letter digit tokens), number-words and function words, single-char
# tokens. A live store was collecting exactly these (fixtures: "One",
# "10", "then") — they are not graph coordinates, only noise edges that
# orphan into the entities table. `entities --gc` sweeps what older writes
# already stored; this filter stops the inflow.
JUNK_NER_LABELS = frozenset({"CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"})

DEFAULT_JUNK_WORDS = frozenset("""
    then now today yesterday tomorrow
    one two three four five six seven eight nine ten
    eleven twelve thirteen fourteen fifteen sixteen seventeen
    eighteen nineteen twenty thirty forty fifty sixty seventy
    eighty ninety hundred thousand million billion
    first second third fourth fifth sixth seventh eighth ninth tenth
    single several many much few half quarter twice
""".split())

_HAS_DIGIT = re.compile(r"\d")


def is_junk_entity(name: str, ent_type: str = "", extra_words=None) -> bool:
    """True when an extracted entity name falls in a junk class.

    Classes: single-char; token with a digit but no letters ("10", "$40",
    "50%", "16:00"); NER number/quantity labels (CARDINAL, ORDINAL, MONEY,
    PERCENT, QUANTITY); number-word/function-word list ("One", "then").
    Config supersedes nothing: `entity_junk_words` (limbic.yaml) only
    EXTENDS the word list — the default classes always hold.
    """
    token = (name or "").strip()
    if len(token) < 2:  # single-char
        return True
    if _HAS_DIGIT.search(token) and not any(ch.isalpha() for ch in token):
        return True  # pure number/price/percent/time token
    if (ent_type or "").strip().upper() in JUNK_NER_LABELS:
        return True
    if extra_words:
        token_l = token.lower()
        for word in extra_words:
            w = str(word).strip().lower()
            if w and token_l == w:
                return True
    return token.lower() in DEFAULT_JUNK_WORDS


def junk_matcher(extra_words=None):
    """Matcher for the gc sweep — same classes as extraction, plus config words."""
    extras = [str(w) for w in (extra_words or []) if str(w).strip()]

    def match(name: str, ent_type: str = "") -> bool:
        return is_junk_entity(name, ent_type, extras)

    return match

# Known entity patterns — CONFIG-DRIVEN, empty by default.
# Deployments add personal/project names via limbic.yaml:
#   known_entities: ["PERSON:Grace", "ORG:Acme Corp"]
# The plugin ships empty: no private names in a publishable tree.
KNOWN_PATTERNS = []

# Script ranges (start, inclusive-end) used for language routing.
# Matches spaCy's language coverage; scripts without a configured
# pipeline fall back to the first loaded pipeline.
_SCRIPT_RANGES: list[tuple[str, int, int]] = [
    # Cyrillic (ru, bg, uk …)
    ("ru", 0x0400, 0x04FF),
    # Greek (el)
    ("el", 0x0370, 0x03FF),
    # Hebrew (he)
    ("he", 0x0590, 0x05FF),
    # Arabic (ar)
    ("ar", 0x0600, 0x06FF),
    # Devanagari (hi, mr …)
    ("hi", 0x0900, 0x097F),
    # Thai
    ("th", 0x0E00, 0x0E7F),
    # Korean Hangul
    ("ko", 0xAC00, 0xD7AF),
    # Hiragana / Katakana / CJK Unified Ideographs (ja, zh)
    ("ja", 0x3040, 0x30FF),
    ("ja", 0x4E00, 0x9FFF),
]


def _model_path(model: str) -> str:
    """What to hand ``spacy.load`` for *model*.

    The plugin vendors the English pipeline as data (``models/<model>/``), so a
    fresh install needs no download at all; anything else resolves by package
    name out of the dependency environment.
    """
    vendored = Path(__file__).resolve().parent / "models" / model
    return str(vendored) if (vendored / "meta.json").is_file() else model


class EntityExtractor:
    """NLP entity extraction with spaCy + custom EntityRuler."""

    # Model name mapping per language
    MODEL_MAP = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "es": "es_core_news_sm",
        "pt": "pt_core_news_sm",
        "it": "it_core_news_sm",
        "nl": "nl_core_news_sm",
        "el": "el_core_news_sm",
        "ru": "ru_core_news_sm",
        "pl": "pl_core_news_sm",
        "ja": "ja_core_news_sm",
        "zh": "zh_core_web_sm",
        "ko": "ko_core_news_sm",
        "multilingual": "xx_ent_wiki_sm",
    }

    def __init__(self, languages: list[str] | None = None, known_patterns: list[dict] | None = None,
                 junk_words: list[str] | None = None):
        self._pipelines: dict[str, object] = {}
        self._available = False
        # Config-driven patterns (limbic.yaml known_entities) override the
        # empty module default — deployments own their names, not the plugin.
        self._known_patterns = known_patterns or []
        # Config junk words (limbic.yaml entity_junk_words) EXTEND the default
        # classes; defaults are never removable.
        self._junk_extra = [str(w).strip() for w in (junk_words or []) if str(w).strip()]

        languages = languages or ["en"]
        self._languages = list(languages)

        for lang in languages:
            model = self.MODEL_MAP.get(lang)
            if not model:
                logger.warning("limbic: unknown language '%s', skipping", lang)
                continue
            try:
                nlp = self._load_model(model)
                if nlp:
                    self._pipelines[lang] = nlp
                    logger.info("limbic: spaCy model loaded: %s", model)
            except Exception as e:
                # en ships vendored; any other language's model must be installed. Never
                # download from here — log the fix instead (see `hermes limbic languages`).
                logger.warning(
                    "limbic: spaCy model %s not available — entity extraction for '%s' disabled. "
                    "Install with: hermes limbic languages --set %s", model, lang, lang)

        if self._pipelines:
            self._available = True
        else:
            logger.warning("limbic: no spaCy models loaded — entity extraction disabled")

    def _load_model(self, model: str):
        """Load a spaCy model and add the custom EntityRuler.

        The bundled model is loaded from its vendored directory; any other
        language falls back to the installed package name.
        """
        import spacy
        from spacy.pipeline import EntityRuler

        nlp = spacy.load(_model_path(model))
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = KNOWN_PATTERNS if not self._known_patterns else self._known_patterns
        if patterns:
            ruler.add_patterns(patterns)
        return nlp

    @property
    def available(self) -> bool:
        return self._available

    def _detect_language(self, text: str) -> str:
        """Script-range language routing. Falls back to the first loaded pipeline.

        Latin-script languages (en, fr, de, es …) share the Unicode ranges, so
        script detection alone can't separate them. spaCy pipelines are
        language-specific models, not tokenizers — routing a French sentence
        through en_core_web_sm still extracts names; precision differs, not
        correctness. Deployments needing Latin-language separation configure
        the pipelines they use; the loaded-pipeline fallback covers the rest.
        # ponytail: script ranges route non-Latin; Latin languages resolved
        # by fallback + config, langdetect only if precision measurably matters.
        """
        if not text:
            return "en"

        # Strip punctuation/spaces — count only script-bearing characters
        letters = [c for c in text if unicodedata.category(c).startswith("L")]
        if not letters:
            return "en"

        script_counts: dict[str, int] = {}
        for c in letters:
            cp = ord(c)
            for lang, start, end in _SCRIPT_RANGES:
                if start <= cp <= end:
                    script_counts[lang] = script_counts.get(lang, 0) + 1
                    break

        if script_counts:
            # Detected script language — caller falls back to first loaded
            # pipeline when this language isn't configured.
            return max(script_counts, key=lambda k: script_counts[k])

        # Latin / unhandled script → first loaded pipeline handles it
        return "en"

    def extract(self, content: str) -> list[dict]:
        """Extract named entities from text.

        Returns list of {name, type} dicts.
        Returns empty list if spaCy is not available.
        """
        if not self._available or not content or not content.strip():
            return []

        lang = self._detect_language(content)
        nlp = self._pipelines.get(lang) or next(iter(self._pipelines.values()), None)

        if not nlp:
            return []

        try:
            doc = nlp(content)
            entities = []
            seen = set()
            for ent in doc.ents:
                name = ent.text.strip()
                if not name or len(name) < 2:
                    continue
                if is_junk_entity(name, ent.label_, self._junk_extra):
                    continue  # junk class — never stored, gc sweeps legacy rows
                # Deduplicate by (name, type) — spaCy can produce overlapping spans
                key = (name.lower(), ent.label_)
                if key in seen:
                    continue
                seen.add(key)
                entities.append({"name": name, "type": ent.label_})
            return entities
        except Exception as e:
            logger.debug("limbic: entity extraction failed: %s", e)
            return []