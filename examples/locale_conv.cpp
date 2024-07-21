#include <boost/locale.hpp>
#include <filesystem>
#include <fstream>

// 外码类型：char
// 内码类型：wchar_t
// 状态类型：std::mbstate_t
using Codecvt = std::codecvt<char, wchar_t, std::mbstate_t>;

// 以 loc 规定的编码，把内码编码成外码
std::string narrow(std::locale const &loc, std::wstring const &wstr) {
    // use_facet 函数获得 locale 在字符转换 方面的 facet
    auto const &cvt = std::use_facet<Codecvt>(loc);
    std::string str(wstr.size() * 4, '\0');
    wchar_t const *from_next;
    char *to_next;
    std::mbstate_t state{};
    auto res = cvt.in(state, wstr.data(), wstr.data() + wstr.size(), from_next, str.data(), str.data() + str.size(), to_next);
    if (res == Codecvt::ok) {
        // 转换成功
        str.resize(to_next - str.data());
        return str;
    } else if (res == Codecvt::partial) {
        // 转换部分成功
        str.resize(to_next - str.data());
        return str;
    } else {
        // 转换失败
        return "";
    }
}

// 以 loc 规定的编码，把外码编码成内码
std::wstring widen(std::locale const &loc, std::string const &str) {
    // use_facet 函数获得 locale 在字符转换 方面的 facet
    auto const &cvt = std::use_facet<Codecvt>(loc);
    std::wstring wstr(str.size(), L'\0');
    char const *from_next;
    wchar_t *to_next;
    std::mbstate_t state{};
    auto res = cvt.out(state, str.data(), str.data() + str.size(), from_next, wstr.data(), wstr.data() + wstr.size(), to_next);
    if (res == Codecvt::ok) {
        // 转换成功
        wstr.resize(to_next - wstr.data());
        return wstr;
    } else if (res == Codecvt::partial) {
        // 转换部分成功
        wstr.resize(to_next - wstr.data());
        return wstr;
    } else {
        // 转换失败
        return L"";
    }
}

int main() {
    std::wstring s = L"日本語";
    std::locale loc = std::locale("");
    // 用 facet 来转换字符串
    myfacet.i(s[0]); // 转换宽字符到内码
    myfacet.narrow(s[0], '?'); // 转换内码到宽字符
}
