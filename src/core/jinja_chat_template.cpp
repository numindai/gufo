#include "src/core/jinja_chat_template.hpp"

#include <cstdint>
#include <exception>
#include <minja/minja.hpp>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace gufo::core {
namespace {

std::shared_ptr<minja::Context> TemplateBuiltins() {
  auto builtins = minja::Context::builtins();
  // Minja supports the string method, but not Jinja's replace filter (used
  // by NuExtract's task preamble). Register it without changing GGUF source.
  builtins->set(
      "replace",
      minja::simple_function(
          "replace", {"s", "old", "new", "count"},
          [](const std::shared_ptr<minja::Context>&, minja::Value& args) {
            auto text = args.at("s").to_str();
            const auto old = args.at("old").to_str();
            const auto replacement = args.at("new").to_str();
            auto count = args.contains("count") && !args.at("count").is_null()
                             ? args.at("count").get<std::int64_t>()
                             : -1;
            std::size_t position = 0;
            while (count != 0) {
              position = text.find(old, position);
              if (position == std::string::npos)
                break;
              text.replace(position, old.size(), replacement);
              position += replacement.size();
              if (count > 0)
                --count;
              if (old.empty()) {
                if (position == text.size())
                  break;
                // Empty matches lie between Unicode characters, not bytes.
                ++position;
                while (position < text.size() &&
                       (static_cast<unsigned char>(text.at(position)) &
                        0xc0U) == 0x80U) {
                  ++position;
                }
              }
            }
            return minja::Value(text);
          }));
  return builtins;
}

std::string SpecialToken(const GgufReader& reader, std::string_view key,
                         const std::vector<std::string_view>& tokens) {
  if (reader.FindMetadata(key) == nullptr)
    return {};
  const auto id = reader.GetMetadataUint32(key);
  if (!id || *id >= tokens.size())
    throw std::invalid_argument("Invalid GGUF token ID: " + std::string(key));
  return std::string(tokens.at(*id));
}

}  // namespace

struct JinjaChatTemplate::Impl {
  std::shared_ptr<minja::TemplateNode> root;
  std::string bos_token;
  std::string eos_token;
};

JinjaChatTemplate::JinjaChatTemplate(std::shared_ptr<const Impl> impl)
    : impl_(std::move(impl)) {}

std::unique_ptr<JinjaChatTemplate> JinjaChatTemplate::CreateFromGguf(
    const GgufReader& reader, std::string* error_msg) {
  if (error_msg)
    error_msg->clear();
  try {
    const auto source = reader.GetMetadataString("tokenizer.chat_template");
    if (!source || source->empty())
      throw std::invalid_argument(
          "GGUF requires a non-empty tokenizer.chat_template string");
    auto impl = std::make_shared<Impl>();
    impl->root = minja::Parser::parse(std::string(*source),
                                      {.trim_blocks = true,
                                       .lstrip_blocks = true,
                                       .keep_trailing_newline = false});
    const auto tokens = reader.GetMetadataStringArray("tokenizer.ggml.tokens");
    impl->bos_token =
        SpecialToken(reader, "tokenizer.ggml.bos_token_id", tokens);
    impl->eos_token =
        SpecialToken(reader, "tokenizer.ggml.eos_token_id", tokens);
    return std::unique_ptr<JinjaChatTemplate>(
        new JinjaChatTemplate(std::move(impl)));
  } catch (const std::exception& error) {
    if (error_msg)
      *error_msg = "Jinja chat template: " + std::string(error.what());
    return nullptr;
  }
}

std::optional<std::string> JinjaChatTemplate::Render(
    const json::Value& context, std::string* error_msg,
    std::size_t max_output_bytes) const {
  if (error_msg)
    error_msg->clear();
  try {
    if (!context.is_object())
      throw std::invalid_argument("context must be a JSON object");
    // Keep Gufo's public JSON API and insertion order; Minja's dependency
    // remains private to this translation unit.
    auto values = nlohmann::ordered_json::parse(context.dump());
    values.emplace("bos_token", impl_->bos_token);
    values.emplace("eos_token", impl_->eos_token);
    values.emplace("add_generation_prompt", true);
    auto output = impl_->root->render(
        minja::Context::make(minja::Value(values), TemplateBuiltins()));
    if (output.size() > max_output_bytes)
      throw std::length_error("rendered prompt exceeds max_output_bytes");
    return output;
  } catch (const std::exception& error) {
    if (error_msg)
      *error_msg = "Jinja chat template: " + std::string(error.what());
    return std::nullopt;
  }
}

}  // namespace gufo::core
