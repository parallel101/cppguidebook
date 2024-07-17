#include <fmt/format.h>
#include <boost/locale.hpp>

using boost::locale::conv::utf_to_utf;
using boost::locale::conv::from_utf;

int main() {
    std::u8string s = u8"你好";
    // UTF-8 转 UTF-32：
    std::u32string s32 = utf_to_utf<char32_t>(s);
    // UTF-32 转 UTF-16：
    std::u16string s16 = utf_to_utf<char16_t>(s);
    // UTF-16 转 UTF-8：
    s = utf_to_utf<char8_t>(s32);
    fmt::println("{}", from_utf(s, ""));
    return 0;
}
