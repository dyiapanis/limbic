"""Vendored English pipeline — the model ships inside the plugin as data.

A fresh install must extract entities with no download, no pip step and no
package named `en_core_web_sm` in site-packages. Run:

    cd <plugins parent> && python -m pytest limbic/tests/test_vendored_model.py -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

spacy = pytest.importorskip("spacy")

from limbic.entities import _model_path  # noqa: E402


def test_model_path_prefers_vendored_dir():
    path = _model_path("en_core_web_sm")
    assert os.path.isdir(path), "vendored model missing from the package"
    assert os.path.isfile(os.path.join(path, "meta.json"))


def test_vendored_model_loads_and_extracts():
    """Loads from the vendored directory — proves no package install is needed."""
    nlp = spacy.load(_model_path("en_core_web_sm"))
    ents = {e.text for e in nlp("Demetri met Adira in Brisbane last Tuesday.").ents}
    assert ents & {"Demetri", "Adira", "Brisbane"}, ents


def test_other_languages_fall_back_to_package_name():
    assert _model_path("fr_core_news_sm") == "fr_core_news_sm"
