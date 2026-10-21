"""Opt-in live Chat Completions logprob checks through the official SDK."""

import math


def check_logprobs(client, model, checks):
    base = dict(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: Hello world!"}],
        temperature=0,
        max_tokens=24,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    def validate(text, scores, count):
        assert scores and text, (text, scores)
        raw = bytearray()
        for score in scores:
            assert math.isfinite(score.logprob) and score.logprob <= 0, score
            assert score.bytes is not None, score
            raw.extend(score.bytes)
            assert len(score.top_logprobs) <= count, score
            previous = 0.0
            for alternative in score.top_logprobs:
                assert math.isfinite(alternative.logprob), alternative
                assert alternative.logprob <= previous, score
                assert alternative.bytes is not None, alternative
                previous = alternative.logprob
            if count:
                assert score.top_logprobs, score
                assert sum(math.exp(item.logprob) for item in score.top_logprobs) <= 1.000001
        assert raw.decode("utf-8", errors="replace") == text, (text, raw)

    reference = None
    for count in (0, 5, 20):
        request = dict(base, logprobs=True, top_logprobs=count)
        complete = client.chat.completions.create(**request)
        choice = complete.choices[0]
        assert choice.logprobs is not None and choice.logprobs.refusal is None
        validate(choice.message.content, choice.logprobs.content, count)
        assert complete.usage.to_dict()["gufo"]["draft_rounds"] == 0
        current = (choice.message.content, [item.logprob for item in choice.logprobs.content])
        if reference is None:
            reference = current
        else:
            assert current == reference, (current, reference)
        checks[f"logprobs_buffered_{count}"] = complete.to_dict()

        text, scores, usage = "", [], None
        with client.chat.completions.create(
            **request, stream=True, stream_options={"include_usage": True}
        ) as stream:
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                for item in chunk.choices:
                    text += item.delta.content or ""
                    if item.logprobs:
                        scores.extend(item.logprobs.content or [])
        validate(text, scores, count)
        assert text == choice.message.content
        assert [item.to_dict() for item in scores] == [
            item.to_dict() for item in choice.logprobs.content
        ]
        assert usage and usage.to_dict()["gufo"]["draft_rounds"] == 0
        checks[f"logprobs_stream_{count}"] = {
            "text": text, "logprobs": [item.to_dict() for item in scores],
            "usage": usage.to_dict(),
        }

    disabled = client.chat.completions.create(**base, logprobs=False)
    assert disabled.choices[0].logprobs is None
    assert disabled.choices[0].message.content == reference[0]
    checks["logprobs_disabled"] = disabled.to_dict()
