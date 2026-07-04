from data.context_utils import build_context_text


def test_python_repr_context_applies_window_and_excludes_future_turns():
    context = "['<a>old turn', '<b>recent turn', '<a>target text', '<b>future turn']"

    result = build_context_text(
        context,
        "target text",
        context_window=1,
        use_speaker_tag=False,
    )

    assert result == "<b>recent turn"


def test_python_repr_context_with_speaker_tags_keeps_recent_first_order():
    context = "['<a>old turn', '<b>recent turn', '<a>target text']"

    result = build_context_text(
        context,
        "target text",
        context_window=2,
        use_speaker_tag=True,
    )

    assert result == 'Speaker_1: "recent turn" Speaker_0: "old turn"'


def test_json_context_still_parses_normally():
    context = '["<a>old turn", "<b>recent turn", "<a>target text"]'

    result = build_context_text(
        context,
        "target text",
        context_window=2,
        use_speaker_tag=False,
    )

    assert result == "<b>recent turn <a>old turn"
