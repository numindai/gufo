#ifndef GUFO_CORE_JINJA_CHAT_TEMPLATE_HPP_
#define GUFO_CORE_JINJA_CHAT_TEMPLATE_HPP_

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "src/core/gguf_reader.hpp"
#include "src/core/json.hpp"

namespace gufo::core {

/// Reusable renderer for the Jinja subset supported by Minja, independent of
/// model-specific formatter selection and tokenization.
class JinjaChatTemplate {
public:
  /// Parses tokenizer.chat_template once. Missing, empty or invalid metadata
  /// is an error; no model-specific fallback is applied. Owns its data, so the
  /// reader and its backing bytes may be released after construction.
  [[nodiscard]] static std::unique_ptr<JinjaChatTemplate> CreateFromGguf(
      const GgufReader& reader, std::string* error_msg = nullptr);

  /// Context is a JSON object in Hugging Face form: messages, optional tools,
  /// add_generation_prompt and arbitrary model-specific template variables.
  /// Missing bos_token/eos_token default to GGUF vocabulary entries (or "");
  /// missing add_generation_prompt defaults to true. Other variables are left
  /// undefined so the embedded template can choose its own defaults.
  /// Each call has an independent context. Results exceeding max_output_bytes
  /// are rejected after rendering; this is not an execution or memory sandbox.
  [[nodiscard]] std::optional<std::string> Render(
      const json::Value& context, std::string* error_msg = nullptr,
      std::size_t max_output_bytes = 1024ULL * 1024ULL) const;

private:
  struct Impl;
  explicit JinjaChatTemplate(std::shared_ptr<const Impl> impl);
  std::shared_ptr<const Impl> impl_;
};

}  // namespace gufo::core

#endif  // GUFO_CORE_JINJA_CHAT_TEMPLATE_HPP_
