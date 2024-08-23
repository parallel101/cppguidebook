#include <fmt/core.h>
#include <system_error>
#include "tl-expected.hpp"
#include <fcntl.h>
// 今日主题：现代 C++ 中的错误处理

// int : [INT_MIN, INT_MAX]
// optional<int> : [INT_MIN, INT_MAX] | {nullopt}
// variant<int, error_code> : [INT_MIN, INT_MAX] | {error_code}
// expected<int, error_code> : [INT_MIN, INT_MAX] | {error_code}

template <>
struct tl::bad_expected_access<std::error_code> : std::system_error {
  explicit bad_expected_access(std::error_code e) : std::system_error(std::move(e)) {}
};

namespace mybuss {

enum class login_errc {
    success = 0,
    not_valid_pass,
    not_login,
};

auto const &login_category() {
    static const struct : std::error_category {
        virtual std::string message(int val) const override {
            switch ((login_errc)val) {
                case login_errc::success:
                    return "登录成功！";
                case login_errc::not_valid_pass:
                    return "密码不正确！";
                case login_errc::not_login:
                    return "用户未登录！";
                default:
                    return "未知错误！";
            };
        }

        virtual const char *name() const noexcept override {
            return "login";
        }
    } instance;
    return instance;
}

std::error_code make_error_code(login_errc ec) {
    return std::error_code((int)ec, login_category());
}

}

tl::expected<int, std::error_code> sqrt(int x) {
    // 假装这是一个关于网站登录的业务函数
    if (x < 0) {
        return tl::unexpected{make_error_code(std::errc::argument_out_of_domain)};
    }
    if (x == 3) {
        return tl::unexpected{make_error_code(mybuss::login_errc::not_valid_pass)};
    }
    if (x == 4) {
        return tl::unexpected{make_error_code(mybuss::login_errc::not_login)};
    }
    for (int i = 0;; i++) {
        if (i * i >= x) {
            return i;
        }
    }
}

tl::expected<int, std::error_code> sqrfloor(int x) {
    if (x < 1) {
        return tl::unexpected{make_error_code(std::errc::invalid_argument)};
    }
    auto ret = sqrt(x * x);
    return ret.map([&] (int x) { fmt::println("x * 2"); return x * 2; });
    // 等价于：
    // if (!ret.has_value()) {
    //     return tl::unexpected{ret.error()};
    // }
    // x = ret.value();
    // x *= 2;
    // return x;

    // return ret.map_error([&] (std::error_code ec) {
    //     if (ec == make_error_code(mybuss::login_errc::not_login)) {
    //         ec = make_error_code(mybuss::login_errc::not_valid_pass);
    //     }
    //     return ec;
    // });
    // 等价于：
    // if (!ret.has_value()) {
    //     if (ret.error() == make_error_code(mybuss::login_errc::not_login)) {
    //         ret.error() = make_error_code(mybuss::login_errc::not_valid_pass);
    //     }
    // }
}

tl::expected<int, std::error_code> expectedStdError(int ret) {
    if (ret == -1) {
        return tl::unexpected{std::error_code(errno, std::generic_category())};
    }
    return ret;
}

int checkStdError(int ret) {
    if (ret == -1) {
        throw std::system_error(std::error_code(errno, std::system_category()));
    }
    return ret;
}

// tl::expected<void, std::error_code> expectedCudaError(int ret) {
//     if (ret != 0) {
//         return tl::unexpected{std::error_code(ret, cuda_category())};
//     }
//     return {};
// }

template <class T>
using expected = tl::expected<T, std::error_code>;

struct RAIIFile {
    int fd;

    tl::expected<size_t, std::error_code> write(std::span<const char> buf) {
        return expectedStdError(::write(fd, buf.data(), buf.size()));
    }
};

// tl::expected<void *, std::error_code> cppCudaMalloc(size_t size) {
//     expectedCudaError(cudaMalloc(p));
// }

int main() {
    RAIIFile file{expectedStdError(open("/tmp/test.log", O_WRONLY)).value()};
    std::string s = "asasasas";
    file.write(s).value();
    return 0;
}
