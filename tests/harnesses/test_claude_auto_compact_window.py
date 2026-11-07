from free_claude_code.harnesses.claude import auto_compact_window


def test_default_window() -> None:
    assert auto_compact_window({}) == "190000"


def test_window_override() -> None:
    assert auto_compact_window({"FCC_AUTO_COMPACT_WINDOW": " 262144 "}) == "262144"


def test_invalid_override_falls_back() -> None:
    assert auto_compact_window({"FCC_AUTO_COMPACT_WINDOW": "big"}) == "190000"
    assert auto_compact_window({"FCC_AUTO_COMPACT_WINDOW": "10"}) == "190000"
