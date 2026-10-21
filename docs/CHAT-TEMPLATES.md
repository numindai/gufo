# GGUF chat templates

Gufo uses compiled model-specific chat formatters by default. For models served
by the Qwen GPU runner, `--jinja` selects the Jinja template stored in the GGUF's
`tokenizer.chat_template` metadata. This supports fine-tuned models whose prompt
format differs from Gufo's recognized templates.

The option changes prompt rendering. Tokenization, inference, scheduling and
sampling still run in Gufo. Minja provides the C++ renderer; serving does not
require Python, Transformers or llama.cpp.

## Serve an embedded template

Start a text server with an explicit context limit:

```sh
gufo serve llm --jinja --model /models/NuExtract.gguf \
  --served-model-name nuextract --context 16384 --sessions 1 \
  --speculative off --temperature 0 --host 127.0.0.1 --port 8080
```

`--jinja` is off by default. Without it, the server keeps its compiled formatter
and recognized-template hash validation. With it, the server uses the embedded
template even when its hash is recognized. Missing, empty or invalid template
metadata fails loading; there is no fallback to compiled formatting.
The startup log confirms `event=chat_template renderer=gguf_jinja`.

The option applies to the shared Qwen GPU runner, including the Qwen3.8-27B and
Qwen3.6-35B-A3B paths. DeepSeek and Qwen3.8-Flash-Next reject it. It does not
enable new model architectures or change the local `chat` and `prompt`
commands' formatter selection.

For a NuExtract checkpoint with its extraction template embedded, send the
extraction schema as a JSON **string** in `chat_template_kwargs.template`:

```sh
curl --fail-with-body http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data-binary @- <<'JSON'
{
  "model": "nuextract",
  "messages": [
    {"role": "user", "content": "Alice is 30 years old."}
  ],
  "chat_template_kwargs": {
    "template": "{\"name\":\"string\",\"age\":\"integer\"}",
    "enable_thinking": false
  },
  "temperature": 0,
  "max_tokens": 128
}
JSON
```

The extracted JSON appears in `choices[0].message.content`, for example
`{"name":"Alice","age":30}`. `template` is a model-specific extraction schema,
not replacement Jinja source. The Jinja source always comes from the loaded
GGUF. Other checkpoints may use different variables.

## HTTP request contract

`/v1/chat/completions` passes the `chat_template_kwargs` object to the embedded
template. For NuExtract this includes `template`, `instructions`, `mode` and
`bbox_json`; their interpretation belongs to the model's template.

- Message order and `developer` roles are preserved. Text content arrays keep
  their boundaries, including the separate parts used by in-context examples.
  NuExtract uses developer messages as examples in structured mode: each
  `content` array contains input parts followed by one final text part holding
  the expected output. Inputs can include text and `image_url` parts. Put the
  examples before the target user document and provide the extraction schema
  in `chat_template_kwargs.template`. Developer messages are examples here;
  task instructions belong in `chat_template_kwargs.instructions`.
- Images are supported with the model's matching `--mmproj` BF16 sidecar.
  Send standard `image_url` content parts alongside text. The adapter converts
  them to Hugging Face `image` parts; the embedded template must emit one
  `<|image_pad|>` per image, in conversation order. Gufo expands each placeholder
  to the resized image's visual tokens and supplies its embeddings and position
  layout. Missing or extra placeholders are rejected instead of dropping images.
- Omitted template variables remain undefined so the template can choose its
  defaults. Explicit server reasoning defaults and request reasoning controls
  follow the existing [reasoning contract](SERVER.md#reasoning-controls).
  Conflicting request controls remain errors.
- The server owns `messages`, `tools` and `add_generation_prompt`. Kwargs cannot
  replace the conversation or disable the generation prompt.
- Buffered and streaming responses use the same rendered prompt. Set
  `"stream": true` for streaming and `"stream_options": {"include_usage": true}`
  for final usage.
- Tool declarations, tool calls and tool-result messages are unsupported
  by this HTTP adapter. The reusable renderer has a separate, broader contract
  below.
- Request strings and object keys must not contain model control-token
  spellings. Arbitrary Jinja transformations do not retain the source spans
  needed to encode those spellings as literal user text.

Template rendering failures and adapter input restrictions return HTTP 400
with `error.code: "invalid_chat_template"`. Normal HTTP validation still runs
first: malformed kwargs or conflicting reasoning controls return
`invalid_reasoning`, for example.

Exact token-prefix cache reuse remains available. The adapter does not infer
compiled-template turn boundaries from custom Jinja, so it does not request
the compiled formatter's additional stable-boundary checkpoint. Persistent
cache identities distinguish the two renderers. Changing template variables
can change the rendered prefix and reduce cache reuse.
Image cache identities include decoded pixels, image placement and the projector;
changing an image cannot reuse an incompatible image prefix.

This request contract is for Chat Completions. It does not extend the
Responses or Anthropic Messages APIs with model-specific template kwargs.

## Reusable C++ API

Link `gufo_core` and include
[`src/core/jinja_chat_template.hpp`](../src/core/jinja_chat_template.hpp).
Parse once with `CreateFromGguf`, then reuse the renderer for multiple contexts:

```cpp
#include <iostream>

#include "src/core/gguf_reader.hpp"
#include "src/core/jinja_chat_template.hpp"
#include "src/core/json.hpp"

int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(argv[1], &error);
  auto renderer =
      reader ? gufo::core::JinjaChatTemplate::CreateFromGguf(*reader, &error)
             : nullptr;
  if (!renderer) {
    std::cerr << error << '\n';
    return 1;
  }
  const auto context = gufo::json::parse(R"({
    "messages": [{"role": "user", "content": "Hello"}],
    "add_generation_prompt": true,
    "enable_thinking": false
  })");
  auto prompt = renderer->Render(context, &error);
  if (!prompt) {
    std::cerr << error << '\n';
    return 1;
  }
  std::cout << *prompt;
}
```

Supply the variables required by the artifact's template; the generic example
above does not supply NuExtract's extraction schema. `CreateFromGguf` returns
`nullptr` on failure. `Render` returns `std::nullopt` on failure. Both accept an
optional error string.

The renderer owns its parsed template and special-token strings; the GGUF
reader can be released after construction. Each render has an independent
context and preserves JSON object insertion order. Missing `bos_token` and
`eos_token` default to their GGUF vocabulary entries, or empty strings when
their metadata is absent. Missing `add_generation_prompt` defaults to `true`.
Explicit context values override these defaults.

Unlike the HTTP adapter, the API accepts arbitrary JSON variables, including
`tools`. It returns text only; callers own tokenization, image placement and
cache boundaries. An already rendered prompt can be passed to Gufo inference
with `gufo prompt --raw --file prompt.txt --model /path/to/model.gguf`.

Rendering supports Minja's chat-template subset of Jinja, not full Python
Jinja2 or filesystem imports. The default output bound is 1 MiB; pass a third
argument to `Render` to change it. The bound is checked after rendering and
does not impose an execution or memory limit on the template.

## Dependencies

[Minja](https://github.com/ochafik/minja) and its nlohmann/json dependency are
private headers of `gufo_core`; the public API uses Gufo's `json::Value`.
CMake and Nix use the same hash-pinned
[archive manifest](../cmake/jinja-dependencies.json). Licenses and versions are
recorded in [third-party notices](../THIRD_PARTY_NOTICES.md).

[CMake](../cmake/Jinja.cmake) adds Minja's missing `is undefined` inverse in a
build-local header copy, preserving its existing null/undefined semantics.
It also returns undefined for out-of-range list lookups, matching Jinja2 when
the NuExtract template skips an empty developer example.
The renderer also registers the `replace` filter used by NuExtract. These are
compatibility additions, not changes to the template stored in the GGUF.

Nix supplies the archives without network access during the build. For an
offline source build, set `GUFO_MINJA_URL` and `GUFO_NLOHMANN_JSON_URL` to local
copies of the pinned archives when configuring CMake; hashes are still checked.

## Validation

The hosted `jinja_chat_template_test` covers metadata errors, render isolation,
special-token defaults, ordered JSON, compatibility filters, Qwen reference
goldens and the HTTP adapter. `openai_chat_test` checks request parsing;
`serve_cli_test` checks flag parsing and help. Use the focused commands in
[testing](TESTING.md).

The unchanged supplied NuExtract template is retained in
[`tests/fixtures/nuextract`](../tests/fixtures/nuextract). Its 48 independent
Python Jinja2 reference cases cover developer examples, all extraction modes,
thinking, assistant output and validation errors. The HTTP prompt test also
checks example/target image pixels and placement through this exact template.
Regenerate the references with `python3 tests/fixtures/nuextract/generate.py`
using Jinja2 3.1.6; normal C++ tests do not require Python Jinja2.

Live model checks belong in [tests/functional](../tests/functional/README.md#gguf-jinja-templates).
Compare prompts against independent Hugging Face fixtures and run inference
through Gufo with `--jinja`. Check the same custom artifact without the flag
to confirm that default template validation remains active. Model-specific
results do not establish compatibility with every Jinja template or model.
