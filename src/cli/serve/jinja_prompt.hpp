#ifndef GUFO_SERVER_JINJA_PROMPT_HPP_
#define GUFO_SERVER_JINJA_PROMPT_HPP_

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/cli/serve/text_generation_backend.hpp"
#include "src/core/jinja_chat_template.hpp"

namespace gufo::server {

class JinjaPromptError final : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

/// Text-only HTTP adapter. Throws JinjaPromptError for unsupported input or
/// template errors, which the transport reports as HTTP 400.
[[nodiscard]] std::string RenderJinjaPrompt(
    const core::JinjaChatTemplate& renderer, const ChatRequest& request,
    const tokenization::QwenTokenizer& tokenizer, std::size_t max_output_bytes);

/// Infer whether the embedded template has already opened a reasoning block.
[[nodiscard]] TextGenerationBackend::InitialOutputState JinjaOutputState(
    std::string_view prompt);

}  // namespace gufo::server
#endif  // GUFO_SERVER_JINJA_PROMPT_HPP_
