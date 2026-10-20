#include "src/cli/serve/jinja_prompt.hpp"

#include <array>
#include <stdexcept>

#include "src/models/qwen/control_tokens.hpp"

namespace gufo::server {
namespace {

void CheckLiteralInput(const json::Value& value,
                       const tokenization::QwenTokenizer& tokenizer) {
  // Unlike the compiled formatter, arbitrary Jinja transformations provide no
  // source spans. Reject control spellings instead of turning client text into
  // model framing. Template-authored control tokens are still encoded normally.
  if (value.is_string()) {
    for (const auto& [spelling, id] : tokenizer.SpecialTokens()) {
      (void)id;
      if (value.str().find(spelling) != std::string::npos)
        throw JinjaPromptError(
            "GGUF Jinja inputs must not contain model control-token spellings");
    }
  } else if (value.is_array()) {
    for (const auto& item : value.items())
      CheckLiteralInput(item, tokenizer);
  } else if (value.is_object()) {
    for (const auto& [key, item] : value.members()) {
      CheckLiteralInput(json::Value(key), tokenizer);
      CheckLiteralInput(item, tokenizer);
    }
  }
}

}  // namespace

std::string RenderJinjaPrompt(const core::JinjaChatTemplate& renderer,
                              const ChatRequest& request,
                              const tokenization::QwenTokenizer& tokenizer,
                              std::size_t max_output_bytes) {
  if (!request.tools.empty())
    throw JinjaPromptError("GGUF Jinja HTTP tool calling is not supported");
  if (!request.chat_template_kwargs.is_object())
    throw JinjaPromptError("chat_template_kwargs must be an object");
  auto context = request.chat_template_kwargs;
  CheckLiteralInput(context, tokenizer);
  auto messages = json::Value::array();
  for (const auto& message : request.messages) {
    if (!message.images.empty() || !message.tool_calls.empty() ||
        message.role == tokenization::ChatRole::kTool)
      throw JinjaPromptError(
          "GGUF Jinja HTTP supports text messages without tools or images");
    auto item = json::Value::object();
    item["role"] = message.role == tokenization::ChatRole::kDeveloper
                       ? "developer"
                       : std::string(tokenization::ToString(message.role));
    auto content = message.content_parts_json.empty()
                       ? json::Value(message.content)
                       : json::parse(message.content_parts_json);
    CheckLiteralInput(content, tokenizer);
    if (!message.framing_suffix.empty()) {
      if (content.is_array()) {
        auto suffix = json::Value::object();
        suffix["type"] = "text";
        suffix["text"] = message.framing_suffix;
        content.push_back(std::move(suffix));
      } else {
        content = message.content + message.framing_suffix;
      }
    }
    item["content"] = std::move(content);
    if (!message.thought.empty()) {
      CheckLiteralInput(json::Value(message.thought), tokenizer);
      item["reasoning_content"] = message.thought;
    }
    if (!message.name.empty()) {
      CheckLiteralInput(json::Value(message.name), tokenizer);
      item["name"] = message.name;
    }
    messages.push_back(std::move(item));
  }
  // Conversation and generation controls are transport-owned, not kwargs.
  context["messages"] = std::move(messages);
  context["tools"] = json::Value::array();
  context["add_generation_prompt"] = true;
  if (request.reasoning.enabled)
    context["enable_thinking"] = *request.reasoning.enabled;
  if (request.reasoning.preserve_thinking)
    context["preserve_thinking"] = *request.reasoning.preserve_thinking;
  if (request.reasoning.effort) {
    constexpr std::array<const char*, 6> names{"minimal", "low",   "medium",
                                               "high",    "xhigh", "max"};
    context["reasoning_effort"] =
        names.at(static_cast<std::size_t>(*request.reasoning.effort));
  }
  std::string error;
  auto rendered = renderer.Render(context, &error, max_output_bytes);
  if (!rendered)
    throw JinjaPromptError(error);
  return std::move(*rendered);
}

TextGenerationBackend::InitialOutputState JinjaOutputState(
    std::string_view prompt) {
  using State = TextGenerationBackend::InitialOutputState;
  // Earlier messages may contain reasoning; only inspect the generation turn.
  if (const auto start = prompt.rfind(tokenization::kImStart);
      start != std::string_view::npos)
    prompt.remove_prefix(start);
  const auto open = prompt.rfind("<think>");
  const auto close = prompt.rfind("</think>");
  if (open != std::string_view::npos &&
      (close == std::string_view::npos || open > close))
    return State::kReasoning;
  return close != std::string_view::npos ? State::kContent : State::kAuto;
}

}  // namespace gufo::server
