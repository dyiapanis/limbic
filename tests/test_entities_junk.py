"""Entity junk filter — junk classes never leave extraction.

Fixtures of record are the junk types a live store was collecting: "One"
(number-word), "10" (pure number), "then" (function word), plus single-char
tokens and the number/quantity NE labels. Extraction-side the filter runs in
EntityExtractor.extract; config junk words only EXTEND the defaults. The
`entities --gc` sweep mirrors these classes over already-stored rows
(see test_entities_gc.py).
"""
import pytest

from limbic.entities import DEFAULT_JUNK_WORDS, EntityExtractor, is_junk_entity, junk_matcher


def test_fixture_one_is_junk():
    assert is_junk_entity("One", "CARDINAL")
    # the word-list class holds regardless of what the NE labeler said
    assert is_junk_entity("One", "PERSON")


def test_fixture_10_is_junk():
    assert is_junk_entity("10", "CARDINAL")
    # digit-no-letter class is label-agnostic (taggers split on "10")
    assert is_junk_entity("10", "DATE")


def test_fixture_then_is_junk():
    assert is_junk_entity("then", "UNKNOWN")
    assert is_junk_entity("then", "PERSON")


def test_single_char_is_junk():
    assert is_junk_entity("a", "PERSON")
    assert is_junk_entity("I", "PERSON")


def test_pure_number_price_percent_tokens_are_junk():
    for token in ("3.5", "1,000", "$40", "50%", "16:00", "22/11/2026", "100-200", "+61 7"):
        assert is_junk_entity(token, "PERSON"), token


def test_number_and_quantity_labels_are_junk():
    for label in ("CARDINAL", "ORDINAL", "MONEY", "PERCENT", "QUANTITY"):
        assert is_junk_entity("three hours", label), label


def test_durable_names_and_dates_survive():
    assert not is_junk_entity("Grace", "PERSON")
    assert not is_junk_entity("Harry Potter", "WORK_OF_ART")
    assert not is_junk_entity("Brisbane", "GPE")
    assert not is_junk_entity("November 2026", "DATE")  # durable marker escapes
    assert not is_junk_entity("wombats", "PERSON")


def test_config_words_extend_never_replace_defaults():
    matcher = junk_matcher(["Wombat", "  ", None])
    assert matcher("Wombat", "PERSON")
    assert matcher("wombat", "")
    # a config word list never lifts the default classes
    assert matcher("10", "PERSON") and matcher("then", "")


def test_function_word_and_number_word_fixtures_live_in_default_list():
    # "then" has no NE label of its own and "One" may be tagged anything —
    # the default word list is what makes both classes unconditional.
    assert "then" in DEFAULT_JUNK_WORDS
    assert "one" in DEFAULT_JUNK_WORDS


def test_extraction_never_returns_junk_even_when_labeled():
    # The EntityRuler forces the junk tokens through as PERSON — no dependence
    # on how the statistical tagger happens to treat number-words. The filter
    # must stop them anyway.
    pytest.importorskip("spacy")
    extractor = EntityExtractor(
        languages=["en"],
        known_patterns=[
            {"label": "PERSON", "pattern": [{"LOWER": "one"}]},
            {"label": "PERSON", "pattern": [{"TEXT": "10"}]},
            {"label": "PERSON", "pattern": [{"LOWER": "then"}]},
        ],
    )
    assert extractor.available
    junk = {"One", "10", "then"}
    ents = extractor.extract("One waited 10 minutes then called Grace.")
    names = {e["name"] for e in ents}
    assert not (junk & names), names  # fixture junk never leaves extraction
    assert "Grace" in names  # real entities still come through
    assert not junk & {e["name"] for e in extractor.extract("We met 10 of them, then one more.")}
