#include <boost/locale.hpp>
#include <filesystem>
#include <fstream>

int main() {
    std::wofstream fout(std::filesystem::path(L"你好.txt"));
    boost::locale::generator gen;
    std::locale loc = gen("zh_CN.GBK");
    fout.imbue(loc);
    fout << L"你好，世界"; // 按 GBK 写出文本文件
}
