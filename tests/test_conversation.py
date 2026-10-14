from twotor.conversation import IdlePolicy, SentenceBuffer


def test_idle_checks_back_off_and_stop_after_three_unanswered():
    policy = IdlePolicy(90, 0)
    flags = dict(active=True, busy=False, focus=False, speaking=False)
    assert not policy.due(89, **flags)
    assert policy.due(90, **flags)
    policy.prompted(90)
    assert not policy.due(269, **flags)
    assert policy.due(270, **flags)
    policy.prompted(270)
    assert policy.due(630, **flags)
    policy.prompted(630)
    assert not policy.due(10000, **flags)
    policy.activity(10000)
    assert policy.due(10090, **flags)


def test_no_proactive_when_paused_busy_focus_or_speaking():
    policy = IdlePolicy(90, 0)
    flags = dict(active=True, busy=False, focus=False, speaking=False)
    for key, value in [("active", False), ("busy", True), ("focus", True), ("speaking", True)]:
        assert not policy.due(1000, **(flags | {key: value}))


def test_streaming_sentences_preserve_text_and_bound_long_inputs():
    buffer = SentenceBuffer()
    assert buffer.feed("How is your") == []
    assert buffer.feed(" day? I hope ") == ["How is your day?"]
    assert buffer.feed("it is going well.") == ["I hope it is going well."]
    text = "practice " * 100
    chunks = buffer.feed(text) + buffer.feed("", final=True)
    assert " ".join(chunks).split() == text.split()
    assert max(map(len, chunks)) <= 220
