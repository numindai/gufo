#include "src/core/jinja_chat_template.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <future>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "src/cli/serve/jinja_prompt.hpp"
#include "tests/models/chat_template_golden_helpers.hpp"

namespace {
using gufo::core::GgufReader;
using gufo::core::GgufValueType;
using gufo::core::JinjaChatTemplate;
using gufo::json::parse;
using gufo::json::Value;
using gufo::testing::chat_goldens::Expect;

// Metadata-only GGUF; releasing both the reader and its backing storage here
// also verifies that the renderer owns its template and special-token strings.
std::unique_ptr<JinjaChatTemplate> Load(const Value& metadata,
                                        std::string* error = nullptr) {
  std::vector<std::uint8_t> bytes;
  const auto pod = [&]<typename T>(T value) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
    bytes.insert(bytes.end(), begin, begin + sizeof(value));
  };
  const auto string = [&](std::string_view value) {
    pod(static_cast<std::uint64_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  };
  pod(std::uint32_t{0x46554747});
  pod(std::uint32_t{3});
  pod(std::uint64_t{0});
  pod(static_cast<std::uint64_t>(metadata.members().size()));
  for (const auto& [key, value] : metadata.members()) {
    string(key);
    if (value.is_string()) {
      pod(static_cast<std::uint32_t>(GgufValueType::kString));
      string(value.str());
    } else if (value.is_array()) {
      pod(static_cast<std::uint32_t>(GgufValueType::kArray));
      pod(static_cast<std::uint32_t>(GgufValueType::kString));
      pod(static_cast<std::uint64_t>(value.items().size()));
      for (const auto& token : value.items())
        string(token.str());
    } else {
      pod(static_cast<std::uint32_t>(GgufValueType::kUint32));
      pod(static_cast<std::uint32_t>(value.as_size()));
    }
  }
  bytes.resize((bytes.size() + 31) / 32 * 32);
  auto reader = GgufReader::OpenMemory(bytes.data(), bytes.size(), error);
  Expect(reader != nullptr, "synthetic GGUF opens");
  return JinjaChatTemplate::CreateFromGguf(*reader, error);
}

std::unique_ptr<JinjaChatTemplate> Compile(std::string_view source) {
  auto metadata = Value::object();
  metadata["tokenizer.chat_template"] = std::string(source);
  std::string error;
  auto result = Load(metadata, &error);
  Expect(result != nullptr, "template compiles: " + error);
  return result;
}

std::string Read(std::string_view path) {
  std::ifstream input{std::string(path)};
  Expect(input.good(), "reference fixture opens");
  return {std::istreambuf_iterator<char>(input), {}};
}

void TestGgufAndContext() {
  auto metadata = parse(R"({
    "tokenizer.ggml.tokens": ["<bos>", "<eos>"],
    "tokenizer.ggml.bos_token_id": 0,
    "tokenizer.ggml.eos_token_id": 1
  })");
  metadata["tokenizer.chat_template"] =
      "{{ bos_token }}{% for m in messages %}"
      "{{ m.role }}:{{ m.content | trim }}{{ eos_token }}\n"
      "{% endfor %}{% if add_generation_prompt %}assistant:"
      "{% if enable_thinking %}<think>{% endif %}{% endif %}";
  std::string error;
  auto renderer = Load(metadata, &error);
  Expect(renderer != nullptr, "GGUF template loads: " + error);
  auto context = parse(R"({"messages":[{"role":"user","content":"  café  "}],
                          "enable_thinking":true})");
  const auto original = context.dump();
  Expect(renderer->Render(context, &error) ==
             "<bos>user:café<eos>\nassistant:<think>",
         "embedded template, GGUF special tokens and kwargs are evaluated");
  Expect(context.dump() == original, "caller context is not mutated");
  context["add_generation_prompt"] = false;
  context["bos_token"] = "override";
  Expect(renderer->Render(context) == "overrideuser:café<eos>\n",
         "explicit context overrides defaults");
  auto without_tokens = Compile("{{ bos_token }}{{ eos_token }}ok");
  Expect(without_tokens->Render(Value::object()) == "ok",
         "absent special tokens default to empty strings");

  auto tools = Compile(
      "{%- macro name(t) -%}{{ t.function.name }}{%- endmacro -%}"
      "{% for t in tools %}{{ name(t) }}={{ t.function.parameters | tojson }}"
      "{% endfor %}:{{ messages[0].tool_calls[0].function.arguments.n }}");
  Expect(tools->Render(parse(R"({
    "tools":[{"type":"function","function":{"name":"lookup",
      "parameters":{"z":true,"a":[1,null,"é"]}}}],
    "messages":[{"role":"assistant","tool_calls":[
      {"function":{"name":"lookup","arguments":{"n":2}}}]}]
  })")) == R"(lookup={"z": true, "a": [1, null, "é"]}:2)",
         "macros, tool schemas, ordered Unicode JSON and nested arguments");
}

void TestErrorsAndIsolation() {
  std::string error;
  for (const auto& metadata :
       {Value::object(), parse(R"({"tokenizer.chat_template":""})"),
        parse(R"({"tokenizer.chat_template":4})"),
        parse(R"({"tokenizer.chat_template":"{% if %}"})"),
        parse(R"({"tokenizer.chat_template":"ok",
                  "tokenizer.ggml.tokens":["x"],
                  "tokenizer.ggml.eos_token_id":2})")}) {
    Expect(Load(metadata, &error) == nullptr && !error.empty(),
           "invalid metadata and syntax are reported");
  }
  auto renderer = Compile(
      "{% if fail %}{{ raise_exception('bad conversation') }}{% endif %}"
      "{% set ns = namespace(n=0) %}"
      "{% for m in messages %}{% set ns.n = ns.n + 1 %}{% endfor %}"
      "{{ ns.n }}");
  Expect(!renderer->Render(parse(R"({"fail":true})"), &error) &&
             error.find("bad conversation") != std::string::npos,
         "template runtime errors reach the caller");
  Expect(!renderer->Render(Value::array(), &error) && !error.empty(),
         "non-object context is rejected");
  const auto context = parse(R"({"messages":[{},{}]})");
  Expect(!renderer->Render(context, &error, 0) &&
             error.find("max_output_bytes") != std::string::npos,
         "oversized output is rejected");
  Expect(renderer->Render(context, &error, 1) == "2" && error.empty(),
         "successful render clears stale errors and accepts exact bound");
  std::vector<std::future<std::optional<std::string>>> renders;
  for (int i = 0; i < 8; ++i)
    renders.push_back(std::async(std::launch::async,
                                 [&] { return renderer->Render(context); }));
  for (auto& render : renders)
    Expect(render.get() == "2", "render contexts are independent");
}

void TestReplaceFilter() {
  const auto renderer = Compile(
      "{{ mode | replace('-', ' ') }}|"
      "{{ 'aaa' | replace('a', 'bb', count=2) }}|"
      "{{ 'aaa' | replace('a', 'b', 0) }}|"
      "{{ 'é猫' | replace('', '-') }}|"
      "{{ '' | replace('', '-') }}");
  Expect(renderer->Render(parse(R"({"mode":"template-generation"})")) ==
             "template generation|bbbba|aaa|-é-猫-|-",
         "NuExtract replace filter supports counts and Unicode empty matches");
}

void TestMissingListItems() {
  const auto renderer = Compile(
      "{{ items[0] }}|{{ items[-1] }}|"
      "{{ items[2] is undefined }}|{{ items[-3] is undefined }}|"
      "{{ empty[-1] | default('missing') }}");
  Expect(renderer->Render(parse(R"({"items":["first","last"],"empty":[]})")) ==
             "first|last|True|True|missing",
         "Missing list items are undefined, including empty developer parts");
}

void TestQwenReferenceGoldens() {
  const auto renderer = Compile(Read(GUFO_QWEN38_CHAT_TEMPLATE_REFERENCE));
  const auto goldens = parse(Read(GUFO_CHAT_TEMPLATE_HF_GOLDENS));
  auto context =
      parse(R"({"messages":[{"role":"user","content":"Name one color."}]})");
  const auto check = [&](std::string_view name) {
    std::string error;
    const auto rendered = renderer->Render(context, &error);
    Expect(rendered.has_value(), "Qwen reference renders: " + error);
    const auto& golden =
        gufo::testing::chat_goldens::GoldenCase(goldens, "qwen", name);
    Expect(gufo::testing::chat_goldens::Sha256(*rendered) ==
               golden.member_str("rendered_sha256"),
           "Jinja bytes match independent Hugging Face golden: " +
               std::string(name));
  };
  check("thinking_xhigh");
  context["enable_thinking"] = false;
  check("chat");
  context["enable_thinking"] = true;
  context["reasoning_effort"] = "low";
  check("thinking_low");
  context["reasoning_effort"] = "medium";
  check("thinking_medium");
  context["messages"] = parse(R"([
    {"role":"user","content":"Name one color."},
    {"role":"assistant","content":"Red",
     "reasoning_content":"I should answer with one color."},
    {"role":"user","content":"Name another."}
  ])");
  context["preserve_thinking"] = false;
  check("history_drop");
  context["enable_thinking"] = false;
  context["preserve_thinking"] = true;
  check("history_preserve");

  context = parse(R"({
    "messages":[{"role":"system","content":"Be concise."},
                {"role":"user","content":"Name one color."}],
    "tools":[{"type":"function","function":{"name":"get_weather",
      "description":"Get weather",
      "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                    "required":["city"]}}}]
  })");
  check("system_tools");
  context = parse(R"({
    "messages":[
      {"role":"user","content":"Name one color."},
      {"role":"assistant","content":"","reasoning_content":"Use the tool.",
       "tool_calls":[{"type":"function","function":{
         "name":"get_weather","arguments":{"city":"Rome"}}}]},
      {"role":"user","content":"<tool_response>21 C</tool_response>"}],
    "reasoning_effort":"medium","preserve_thinking":false
  })");
  check("tool_loop_preserve");
}

void TestNuExtractReferenceGoldens() {
  const std::string directory = GUFO_NUEXTRACT_FIXTURES;
  const auto source = Read(directory + "/chat_template.jinja");
  const auto goldens = parse(Read(directory + "/goldens.json"));
  Expect(gufo::testing::chat_goldens::Sha256(source) ==
             goldens.member_str("template_sha256"),
         "NuExtract reference is the unchanged supplied template");
  const auto renderer = Compile(source);
  for (const auto& fixture :
       gufo::testing::chat_goldens::Required(goldens, "cases").items()) {
    std::string error;
    const auto rendered = renderer->Render(
        gufo::testing::chat_goldens::Required(fixture, "context"), &error);
    const auto name = fixture.member_str("name");
    if (fixture.find("error")) {
      Expect(!rendered &&
                 error.find(fixture.member_str("error")) != std::string::npos,
             "NuExtract rejects " + name + ": " + error);
    } else {
      Expect(rendered.has_value(), "NuExtract renders " + name + ": " + error);
      Expect(gufo::testing::chat_goldens::Sha256(*rendered) ==
                 fixture.member_str("rendered_sha256"),
             "NuExtract bytes match Python Jinja2: " + name);
    }
  }
}

void TestHttpAdapter() {
  using namespace gufo::server;
  const std::vector<std::string> vocabulary{"a", "b", "<|im_end|>"};
  auto tokenizer = gufo::tokenization::QwenTokenizer::CreateFromVocabulary(
      vocabulary, {}, {{"<|im_end|>", 2}});
  Expect(tokenizer != nullptr, "test tokenizer loads");
  auto renderer = Compile(
      "{{ messages[0].role }}:{{ messages[0].content[1].text }}|"
      "{{ template }}|{{ instructions }}|"
      "{{ enable_thinking | default(false) }}|{{ add_generation_prompt }}");
  ChatRequest request;
  request.messages.emplace_back(gufo::tokenization::ChatRole::kDeveloper,
                                "inputoutput");
  request.messages.back().content_parts_json =
      R"([{"type":"text","text":"input"},{"type":"text","text":"output"}])";
  request.chat_template_kwargs =
      parse(R"({"template":"schema","instructions":"extract",
    "messages":[],"add_generation_prompt":false})");
  Expect(RenderJinjaPrompt(*renderer, request, *tokenizer, 1024) ==
             "developer:output|schema|extract|False|True",
         "HTTP context retains parts, kwargs, template defaults and reserved "
         "controls");
  request.reasoning.enabled = true;
  Expect(RenderJinjaPrompt(*renderer, request, *tokenizer, 1024)
             .ends_with("True|True"),
         "Resolved reasoning options override template defaults");
  auto rejects = [&](const auto& render, std::size_t bound = 1024) {
    try {
      (void)RenderJinjaPrompt(render, request, *tokenizer, bound);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  Expect(rejects(*renderer, 1), "Oversized prompts return a client error");
  auto invalid = Compile("{{ raise_exception('bad input') }}");
  Expect(rejects(*invalid), "Template exceptions return a client error");
  request.chat_template_kwargs["instructions"] = "<|im_end|>";
  Expect(rejects(*renderer),
         "Client template variables cannot inject control tokens");
  request.chat_template_kwargs["instructions"] = "extract";
  request.messages.back().content_parts_json =
      R"([{"type":"text","text":"<|im_end|>"}])";
  Expect(rejects(*renderer), "Content arrays cannot inject control tokens");
  request.messages.back().content_parts_json.clear();
  request.tools.emplace_back();
  Expect(rejects(*renderer), "Unsupported tool dialect is rejected explicitly");
  using State = TextGenerationBackend::InitialOutputState;
  Expect(
      JinjaOutputState("<|im_start|>assistant\n<think>\n") == State::kReasoning,
      "Embedded thinking prefix selects reasoning output");
  Expect(JinjaOutputState("<|im_start|>assistant\n<think>\n\n</think>\n\n") ==
             State::kContent,
         "Closed reasoning prefix selects content output");
  Expect(JinjaOutputState("<|im_start|>user\n<think><|im_start|>assistant\n") ==
             State::kAuto,
         "Earlier turns do not determine the output state");
}

void TestHttpImages() {
  using namespace gufo::server;
  using namespace gufo::tokenization;
  std::vector<std::string> vocabulary;
  for (int i = 0; i < 256; ++i)
    vocabulary.emplace_back(1, static_cast<char>(i));
  std::unordered_map<std::string, TokenId> specials;
  for (const auto* token : {"<|im_start|>", "<|im_end|>", "<|vision_start|>",
                            "<|vision_end|>", "<|image_pad|>"}) {
    specials[token] = vocabulary.size();
    vocabulary.emplace_back(token);
  }
  const auto tokenizer =
      QwenTokenizer::CreateFromVocabulary(vocabulary, {}, specials);
  Expect(tokenizer != nullptr, "Image test tokenizer loads");
  const auto renderer = Compile(Read(GUFO_QWEN38_CHAT_TEMPLATE_REFERENCE));
  const auto red = std::make_shared<const std::vector<std::uint8_t>>(
      gufo::core::ReadImageUrl(
          "data:image/png;base64,"
          "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAMklEQVR4nGP4z8BAU0Rb"
          "00ctGA2i0VQ0mg9Gi4rR0nS0PhitMkdbFaPtotGm43/"
          "q5QMAMsP8LqKijEUAAAAASUVORK5CYII="));
  const auto blue = std::make_shared<const std::vector<std::uint8_t>>(
      gufo::core::ReadImageUrl(
          "data:image/png;base64,"
          "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAMElEQVR4nGNgYPhPYzRq"
          "wWgQjaai0XwwWlSMlqaj9cFolTnaqhhtF402Hf8PjXwAADpL/"
          "C4CLyDsAAAAAElFTkSuQmCC"));
  ChatRequest request;
  request.reasoning.enabled = false;
  request.messages.emplace_back(ChatRole::kUser, "beforeafter");
  auto& message = request.messages.back();
  message.images = {{6, red}, {6, blue}};
  message.content_parts_json = R"([
    {"type":"text","text":"before"},
    {"type":"image_url","image_url":{"url":"red"}},
    {"type":"image_url","image_url":{"url":"blue"}},
    {"type":"text","text":"after"}])";
  const std::string expected =
      "<|im_start|>user\nbefore<|vision_start|><|image_pad|><|vision_end|>"
      "<|vision_start|><|image_pad|><|vision_end|>after<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  Expect(RenderJinjaPrompt(*renderer, request, *tokenizer, 4096) == expected,
         "Image parts retain interleaved text and adjacent image order");
  const auto prepare = [&](const auto& source,
                           std::string_view encoder = "test",
                           std::uint32_t limit = 4096) {
    return PrepareJinjaPrompt(source, request, *tokenizer, encoder, limit,
                              4096);
  };
  const auto prompt = prepare(*renderer);
  auto expected_tokens =
      tokenizer->Encode("<|im_start|>user\nbefore<|vision_start|>");
  const auto first_offset = expected_tokens.size();
  expected_tokens.insert(expected_tokens.end(), 64, 248056);
  auto middle = tokenizer->Encode("<|vision_end|><|vision_start|>");
  expected_tokens.insert(expected_tokens.end(), middle.begin(), middle.end());
  const auto second_offset = expected_tokens.size();
  expected_tokens.insert(expected_tokens.end(), 64, 248056);
  auto suffix = tokenizer->Encode(
      "<|vision_end|>after<|im_end|>\n"
      "<|im_start|>assistant\n<think>\n\n</think>\n\n");
  expected_tokens.insert(expected_tokens.end(), suffix.begin(), suffix.end());
  Expect(prompt.tokens == expected_tokens,
         "Each resized 256x256 image expands to 64 language image tokens");
  Expect(prompt.images.size() == 2 &&
             prompt.images[0].grid.offset == first_offset &&
             prompt.images[1].grid.offset == second_offset &&
             prompt.images[0].grid.height == 8 &&
             prompt.images[0].grid.width == 8 &&
             prompt.images[0].pixels.pixels[0] == 255 &&
             prompt.images[1].pixels.pixels[2] == 255,
         "Image pixels and grid offsets follow placeholder order");
  Expect(prompt.stable_prefix_tokens == 0 &&
             prompt.cache_identity.size() == 32 &&
             prompt.IdentityForPrefix(first_offset).empty() &&
             !prompt.IdentityForPrefix(first_offset + 1).empty(),
         "Jinja keeps image-aware cache identities without guessing a stable "
         "boundary");
  ChatTemplateOptions options;
  options.enable_thinking = false;
  const auto compiled = gufo::models::qwen::vision::Prepare(
      *tokenizer, request.messages, {}, options, "test", 4096);
  Expect(prompt.tokens == compiled.tokens && prompt.rope == compiled.rope &&
             prompt.cache_identity == compiled.cache_identity,
         "Reference Jinja and compiled prompts agree on expanded tokens, "
         "positions and pixels");
  Expect(prepare(*renderer).cache_identity == prompt.cache_identity &&
             prepare(*renderer, "other-projector").cache_identity !=
                 prompt.cache_identity,
         "Image cache identity is repeatable and projector-specific");
  std::swap(message.images[0].bytes, message.images[1].bytes);
  const auto swapped = prepare(*renderer);
  Expect(
      swapped.tokens == prompt.tokens &&
          swapped.cache_identity != prompt.cache_identity,
      "Different image pixels cannot reuse an identical token-only cache key");
  std::swap(message.images[0].bytes, message.images[1].bytes);
  message.content_parts_json.clear();
  Expect(prepare(*renderer).tokens == prompt.tokens,
         "Internal image offsets reconstruct the same ordered content parts");
  request.add_vision_id = true;
  Expect(RenderJinjaPrompt(*renderer, request, *tokenizer, 4096)
                 .find("Picture 2: <|vision_start|>") != std::string::npos,
         "Vision numbering reaches the embedded template");
  request.add_vision_id = false;
  const auto rejects = [&](const auto& source,
                           std::string_view encoder = "test",
                           std::uint32_t limit = 4096) {
    try {
      (void)prepare(source, encoder, limit);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };
  Expect(rejects(*renderer, ""), "Images require a matching projector");
  Expect(rejects(*renderer, "test", 64),
         "Expanded images obey the context limit");
  Expect(rejects(*Compile("no images")),
         "Templates cannot silently discard images");
  Expect(rejects(*Compile("<|image_pad|><|image_pad|><|image_pad|>")),
         "Templates cannot invent or duplicate image placeholders");
  message.content = "<|image_pad|>";
  message.images = {{0, red}};
  Expect(rejects(*renderer),
         "Image requests cannot inject image control tokens");
  message.content.clear();
  Expect(prepare(*renderer).images.size() == 1,
         "Image-only content is supported");
  request.messages.emplace_back(ChatRole::kUser, "second");
  request.messages.back().images = {{0, blue}};
  Expect(prepare(*renderer).images.size() == 2,
         "Images across message history are supported");
  request.messages.front().role = ChatRole::kDeveloper;
  const auto examples_renderer = Compile(
      "{% for message in messages %}{{ message.role }}:"
      "{% for part in message.content %}"
      "{% if part.type == 'image' %}"
      "<|vision_start|><|image_pad|><|vision_end|>"
      "{% else %}{{ part.text }}{% endif %}"
      "{% endfor %}{% endfor %}");
  Expect(RenderJinjaPrompt(*examples_renderer, request, *tokenizer, 4096) ==
             "developer:<|vision_start|><|image_pad|><|vision_end|>"
             "user:<|vision_start|><|image_pad|><|vision_end|>second",
         "Embedded templates preserve developer examples before user input");
  const auto examples_prompt = prepare(*examples_renderer);
  Expect(
      examples_prompt.images.size() == 2 &&
          examples_prompt.images[0].pixels.pixels[0] == 255 &&
          examples_prompt.images[1].pixels.pixels[2] == 255 &&
          examples_prompt.images[0].grid.offset <
              examples_prompt.images[1].grid.offset,
      "Developer example and user input keep their own pixels and positions");

  const auto nuextract = Compile(
      Read(std::string(GUFO_NUEXTRACT_FIXTURES) + "/chat_template.jinja"));
  request.chat_template_kwargs =
      parse(R"({"template":"{\"name\":\"string\"}"})");
  request.add_vision_id = true;
  request.messages.front().content = "{\"name\":\"Bob\"}";
  request.messages.front().content_parts_json = R"([
    {"type":"image_url","image_url":{"url":"red"}},
    {"type":"text","text":"{\"name\":\"Bob\"}"}])";
  const auto rendered =
      RenderJinjaPrompt(*nuextract, request, *tokenizer, 4096);
  Expect(
      rendered.find(
          "【examples_start】\n【example_input_start】Picture 1: "
          "<|vision_start|><|image_pad|><|vision_end|>【example_input_end】\n"
          "【example_output_start】{\"name\":\"Bob\"}【example_output_end】\n"
          "【examples_end】\n【document_start】\nPicture 2: "
          "<|vision_start|><|image_pad|><|vision_end|>\nsecond\n"
          "【document_end】") != std::string::npos,
      "Exact NuExtract template separates developer input/output from the "
      "target document and numbers images across both");
  const auto nuextract_prompt = prepare(*nuextract);
  Expect(nuextract_prompt.images.size() == 2 &&
             nuextract_prompt.images[0].pixels.pixels[0] == 255 &&
             nuextract_prompt.images[1].pixels.pixels[2] == 255 &&
             nuextract_prompt.images[0].grid.offset <
                 nuextract_prompt.images[1].grid.offset,
         "Exact NuExtract template preserves example and target pixels");
}

}  // namespace

int main() {
  TestHttpAdapter();
  TestHttpImages();
  TestGgufAndContext();
  TestErrorsAndIsolation();
  TestReplaceFilter();
  TestMissingListItems();
  TestQwenReferenceGoldens();
  TestNuExtractReferenceGoldens();
}
