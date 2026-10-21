#ifndef GUFO_CORE_UTF8_HPP_
#define GUFO_CORE_UTF8_HPP_

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {

// Token pieces can split a Unicode scalar or contain malformed byte fallback
// output. JSON events must contain complete UTF-8; incomplete valid prefixes
// wait for the next piece, while malformed subsequences become U+FFFD.
class Utf8Decoder {
public:
  struct SourceSpan {
    std::size_t source_offset;
    std::size_t source_size;
    std::size_t text_offset;
    std::size_t text_size;
  };

  /// Optional source mapping for consumers retaining token byte provenance.
  /// Text offsets are relative to this call's output; source offsets are
  /// absolute.
  std::string Push(std::string_view bytes, bool final = false,
                   std::vector<SourceSpan>* spans = nullptr) {
    pending_.append(bytes);
    std::string output;
    output.reserve(pending_.size());
    std::size_t offset = 0;
    const auto record = [&](std::size_t begin, std::size_t text_begin) {
      if (spans)
        spans->push_back({source_offset_ + begin, offset - begin, text_begin,
                          output.size() - text_begin});
    };
    while (offset < pending_.size()) {
      const auto begin = offset;
      const auto text_begin = output.size();
      const auto lead = static_cast<unsigned char>(pending_[offset]);
      if (lead < 0x80) {
        output.push_back(pending_[offset++]);
        record(begin, text_begin);
        continue;
      }
      const unsigned length = lead >= 0xC2 && lead <= 0xDF   ? 2
                              : lead >= 0xE0 && lead <= 0xEF ? 3
                              : lead >= 0xF0 && lead <= 0xF4 ? 4
                                                             : 0;
      if (length == 0) {
        output.append("\xEF\xBF\xBD");
        ++offset;
        record(begin, text_begin);
        continue;
      }
      unsigned prefix = 1;
      while (prefix < length && offset + prefix < pending_.size()) {
        const auto byte = static_cast<unsigned char>(pending_[offset + prefix]);
        if (byte < 0x80 || byte > 0xBF ||
            (prefix == 1 &&
             ((lead == 0xE0 && byte < 0xA0) || (lead == 0xED && byte > 0x9F) ||
              (lead == 0xF0 && byte < 0x90) || (lead == 0xF4 && byte > 0x8F))))
          break;
        ++prefix;
      }
      if (prefix == length) {
        output.append(pending_, offset, length);
        offset += length;
      } else if (offset + prefix == pending_.size() && !final) {
        break;
      } else {
        output.append("\xEF\xBF\xBD");
        offset += prefix;
      }
      record(begin, text_begin);
    }
    pending_.erase(0, offset);
    source_offset_ += offset;
    return output;
  }

private:
  std::string pending_;
  std::size_t source_offset_{0};
};

}  // namespace gufo::core

#endif  // GUFO_CORE_UTF8_HPP_
