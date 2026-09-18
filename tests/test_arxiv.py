from app.arxiv import normalize_id

def test_normalize_id():
    assert normalize_id("2401.12345") == "2401.12345"
    assert normalize_id("arXiv:2401.12345") == "2401.12345"
    assert normalize_id("https://arxiv.org/abs/2401.12345v2") == "2401.12345v2"

def test_invalid_id():
    assert normalize_id("not-an-arxiv-id") is None
