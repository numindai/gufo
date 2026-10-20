#!/usr/bin/env python3
"""Manual GGUF Jinja HTTP checks using independent HF prompt/token fixtures.

Each fixture contains name, context, expected_token_ids and expected_output.
Optional stream, expect_status and error_contains exercise protocol failures.
Run against an already started Gufo server; this script never downloads models.
"""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads(args.fixtures.read_text())
    report = {"url": args.url, "model": args.model, "requests": []}
    for fixture in fixtures:
        name = fixture["name"]
        context = fixture["context"]
        body = {
            "model": args.model,
            "messages": context["messages"],
            "chat_template_kwargs": {
                key: value for key, value in context.items()
                if key not in ("messages", "add_generation_prompt")
            },
            "temperature": 0,
            "max_tokens": fixture.get("max_tokens", 384),
            "stream": fixture.get("stream", False),
            "cache_prompt": fixture.get("cache_prompt", True),
        }
        if "tools" in fixture:
            body["tools"] = fixture["tools"]
        if "kwargs_override" in fixture:
            body["chat_template_kwargs"] = fixture["kwargs_override"]
        if body["stream"]:
            body["stream_options"] = {"include_usage": True}
        request_path = args.output / f"{name}.request.json"
        request_path.write_text(json.dumps(body, ensure_ascii=False))
        start = time.monotonic()
        completed = subprocess.run(
            ["curl", "--silent", "--show-error", "--max-time", "180",
             "--write-out", "\n%{http_code}",
             "--header", "Content-Type: application/json",
             "--data-binary", f"@{request_path}",
             args.url.rstrip("/") + "/v1/chat/completions"],
            capture_output=True, text=True, check=True,
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        raw, status = completed.stdout.rsplit("\n", 1)
        (args.output / f"{name}.response.txt").write_text(raw)
        result = {"name": name, "status": int(status), "elapsed_ms": elapsed_ms}
        report["requests"].append(result)
        try:
            assert int(status) == fixture.get("expect_status", 200), raw
            if int(status) != 200:
                error = json.loads(raw)["error"]
                assert fixture["error_contains"] in error["message"], raw
                if "error_code" in fixture:
                    assert error["code"] == fixture["error_code"], raw
            else:
                if body["stream"]:
                    assert "data: [DONE]" in raw, raw
                    chunks = [
                        json.loads(line[6:]) for line in raw.splitlines()
                        if line.startswith("data: ") and line != "data: [DONE]"
                    ]
                    assert all("error" not in chunk for chunk in chunks), raw
                    assert any(choice.get("finish_reason") == "stop" for chunk in chunks for choice in chunk.get("choices", [])), raw
                    text = "".join(
                        choice.get("delta", {}).get("content") or ""
                        for chunk in chunks for choice in chunk.get("choices", [])
                    )
                    response = next(chunk for chunk in reversed(chunks) if chunk.get("usage"))
                else:
                    response = json.loads(raw)
                    assert response["choices"][0]["finish_reason"] == "stop", raw
                    text = response["choices"][0]["message"]["content"]
                    result["reasoning"] = response["choices"][0]["message"].get("reasoning_content")
                result["text"] = text
                result["usage"] = response["usage"]
                result["timings"] = response["usage"]["gufo"]
                assert json.loads(text) == fixture["expected_output"], text
                assert response["usage"]["prompt_tokens"] == len(fixture["expected_token_ids"]), response
                cached = response["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                if fixture.get("expect_cached"):
                    assert cached > 0, response
                if fixture.get("cache_prompt") is False:
                    assert cached == 0, response
                if fixture.get("expect_reasoning"):
                    assert result.get("reasoning"), response
                assert response["usage"]["gufo"]["prefill_tokens"] == response["usage"]["prompt_tokens"] - cached, response
                drafts = response["usage"]["draft_tokens"]
                assert drafts == 0, response
            result["passed"] = True
            print(f"{name}: PASS ({elapsed_ms:.1f} ms)", flush=True)
        finally:
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
