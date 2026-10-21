#!/usr/bin/env python3
"""Regenerate exact-template references with Python Jinja2, independently of Minja.

Requires Jinja2 3.1.6. The generation extension only marks assistant output;
its body renders unchanged, as in Hugging Face's assistant-token tracking.
Normal C++ tests consume goldens.json and do not require Python Jinja2.
"""

import hashlib
import json
from pathlib import Path

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Generation(Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return nodes.CallBlock(self.call_method("render_body"), [], [], body).set_lineno(lineno)

    def render_body(self, caller):
        return caller()


def raise_exception(message):
    raise jinja2.TemplateError(message)


def text(value):
    return {"type": "text", "text": value}


def message(role, content, **kwargs):
    return {"role": role, "content": content, **kwargs}


def main():
    assert jinja2.__version__ == "3.1.6", "Use the recorded reference version"
    root = Path(__file__).resolve().parent
    source = (root / "chat_template.jinja").read_text()
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[Generation])
    environment.globals["raise_exception"] = raise_exception
    renderer = environment.from_string(source)
    cases = []
    user = message("user", "Alice is 30 years old.")
    image = {"type": "image", "image": ""}
    schema = '{"name":"string","age":"integer"}'

    def add(name, messages=None, **kwargs):
        context = {"messages": [user] if messages is None else messages,
                   "add_generation_prompt": True, **kwargs}
        case = {"name": name, "context": context}
        try:
            rendered = renderer.render(**context)
            case["rendered_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
        except jinja2.TemplateError as error:
            case["error"] = str(error)
        cases.append(case)

    add("content_default")
    add("content_thinking", enable_thinking=True)
    for mode in ["structured", "content", "markdown", "template-generation",
                 "bbox", "embeddings", "unknown"]:
        kwargs = {"mode": mode}
        if mode == "structured":
            kwargs["template"] = schema
        if mode == "bbox":
            kwargs["bbox_json"] = '{"name":"Alice"}'
        add("mode_" + mode, **kwargs)
    add("structured_instructions", template=schema, instructions="Use exact names.")
    add("structured_thinking", template=schema, enable_thinking=True)
    examples = [message("developer", [text("Bob is 24."), text('{"name":"Bob","age":24}')]),
                message("developer", [text("Eve"), text("is 45."), text('{"name":"Eve","age":45}')])]
    add("developer_text_examples", [*examples, user], template=schema)
    add("developer_image_examples", [
        message("system", [text("Keep all records.")]),
        message("developer", [image, text("Example OCR"), text('{"name":"Bob","age":24}')]),
        message("developer", [text("Second example"), image, image, text('{"name":"Eve","age":45}')]),
        message("user", [image, text("Target OCR"), image])],
        template=schema, add_vision_id=True)
    add("developer_no_input", [message("developer", [text("output only")]), user], template=schema)
    add("developer_empty", [message("developer", []), user], template=schema)
    add("developer_string_output", [message("developer", [text("Bob"), '{"name":"Bob"}']), user], template=schema)
    add("developer_missing_content", [{"role": "developer"}, user], template=schema)
    add("content_ignores_examples", [*examples, user])
    add("user_parts", [message("user", [text("  café 猫  "), image, text(" second ")])], add_vision_id=True)
    add("user_null", [message("user", None)])
    add("user_missing_content", [{"role": "user"}])
    add("multiple_users", [user, message("user", "ignored", name="example"), message("user", "Next page")])
    add("system_extra", [message("system", "  Custom rules.  "), user])
    system_prompt = renderer.render(messages=[user], add_generation_prompt=True).split("<|im_start|>system\n")[1].split("<|im_end|>")[0]
    add("system_deduplicated", [message("system", system_prompt), user])
    add("assistant_reasoning", [user, message("assistant", "Alice", reasoning_content="  Think briefly.  ")], add_generation_prompt=False, enable_thinking=True)
    add("assistant_inline_reasoning", [user, message("assistant", "<think>\nThink briefly.\n</think>\n\nAlice")], add_generation_prompt=False, enable_thinking=True)
    add("assistant_drop_reasoning", [user, message("assistant", "<think>\nThink briefly.\n</think>\n\nAlice")], add_generation_prompt=False)
    add("assistant_middle", [user, message("assistant", "ignored"), user])
    add("no_generation_prompt", add_generation_prompt=False)
    add("bbox_inferred", bbox_json='{"name":"Alice"}')
    add("empty_messages", [])
    add("conflicting_template_bbox", template=schema, bbox_json="{}")
    add("embeddings_template", mode="embeddings", template=schema)
    add("embeddings_bbox", mode="embeddings", bbox_json="{}")
    add("structured_missing_template", mode="structured")
    for value in [None, "", "  ", {}, 42]:
        add("bbox_invalid_" + repr(value), mode="bbox", bbox_json=value)
    for mode in ["bbox", "template-generation", "embeddings"]:
        add("thinking_invalid_" + mode, mode=mode, enable_thinking=True,
            **({"bbox_json": "{}"} if mode == "bbox" else {}))
    add("system_image", [message("system", [image]), user])
    add("system_late", [user, message("system", "late")])
    add("content_invalid_item", [message("user", [{"type": "audio"}])])
    add("content_invalid_type", [message("user", 42)])
    add("assistant_generation_conflict", [user, message("assistant", "Alice")])
    result = {"reference": "Jinja2 " + jinja2.__version__,
              "template_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "cases": cases}
    (root / "goldens.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"Generated {len(cases)} independent reference cases")


if __name__ == "__main__":
    main()
