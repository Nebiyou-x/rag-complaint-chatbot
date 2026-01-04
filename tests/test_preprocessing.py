import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"i am writing to file a complaint", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def test_clean_text_lowercase():
    text = "THIS IS A COMPLAINT"
    cleaned = clean_text(text)
    assert cleaned == "this is a complaint"


def test_clean_text_removes_special_chars():
    text = "Bad service!!! $$$"
    cleaned = clean_text(text)
    assert "!" not in cleaned
    assert "$" not in cleaned


def test_clean_text_removes_boilerplate():
    text = "I am writing to file a complaint about billing"
    cleaned = clean_text(text)
    assert "i am writing to file a complaint" not in cleaned
