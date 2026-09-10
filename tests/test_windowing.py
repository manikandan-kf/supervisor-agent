"""Conversation window budgets — the token-aware trim in nodes._window.

The checkpointer keeps the whole conversation; what must not grow with it is
the slice replayed into every governance model call and worker dispatch. These
pin the three behaviours that matter: short conversations pass untouched, long
ones are bounded, and a degenerate turn still yields something to send.
"""

from langchain_core.messages import AIMessage, HumanMessage

from supervisor.nodes import _window


def _turns(n: int, size: int = 40) -> list:
    messages = []
    for i in range(n):
        messages.append(HumanMessage(content=f"question {i} " + "x" * size))
        messages.append(AIMessage(content=f"answer {i} " + "y" * size))
    return messages


def test_a_short_conversation_is_not_trimmed():
    messages = _turns(3)
    assert _window(messages, max_tokens=4000) == messages


def test_a_long_conversation_is_bounded_and_keeps_the_recent_end():
    messages = _turns(200)
    window = _window(messages, max_tokens=500)

    assert 0 < len(window) < len(messages)
    # strategy="last": the newest message survives, the oldest goes first.
    assert window[-1] is messages[-1]
    assert messages[0] not in window


def test_the_window_opens_on_a_human_turn():
    # An assistant reply whose question fell outside the window would be an
    # answer to nothing — start_on="human" drops it.
    window = _window(_turns(200), max_tokens=500)
    assert window[0].type == "human"


def test_an_oversized_single_turn_degrades_to_that_turn_not_to_silence():
    messages = [HumanMessage(content="z" * 100_000)]
    window = _window(messages, max_tokens=100)
    assert window == messages[-1:]


def test_empty_input_stays_empty():
    assert _window([], max_tokens=100) == []
    assert _window(None, max_tokens=100) == []
