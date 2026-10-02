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

The plugin ships with no spaCy model bundled; models auto-download on
first use (~14.5MB each for _sm models).
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# Known entity patterns — CONFIG-DRIVEN, empty by default.
# Deployments add personal/project names via limbic.yaml:
#   known_entities: ["PERSON:Alice", "ORG:Acme Corp"]
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

    def __init__(self, languages: list[str] | None = None, known_patterns: list[dict] | None = None):
        self._pipelines: dict[str, object] = {}
        self._available = False
        # Config-driven patterns (limbic.yaml known_entities) override the
        # empty module default — deployments own their names, not the plugin.
        self._known_patterns = known_patterns or []

        languages = languages or ["en"]

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
                # Model not installed — try auto-download
                logger.info("limbic: spaCy model %s not found, auto-downloading...", model)
                try:
                    self._download_model(model)
                    nlp = self._load_model(model)
                    if nlp:
                        self._pipelines[lang] = nlp
                        logger.info("limbic: spaCy model downloaded and loaded: %s", model)
                except Exception as dl_err:
                    logger.warning("limbic: auto-download of %s failed: %s", model, dl_err)

        if self._pipelines:
            self._available = True
        else:
            logger.warning("limbic: no spaCy models loaded — entity extraction disabled")

    @staticmethod
    def _download_model(model: str):
        """Auto-download a spaCy model via subprocess."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            raise RuntimeError(f"spacy download failed: {result.stderr[:200]}")

    def _load_model(self, model: str):
        """Load a spaCy model and add the custom EntityRuler."""
        import spacy
        from spacy.pipeline import EntityRuler

        nlp = spacy.load(model)
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