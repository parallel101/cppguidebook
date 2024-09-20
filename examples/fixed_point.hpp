#pragma once

#include <cstddef>
#include <climits>
#include <cmath>
#include <cassert>
//#include <string>

template <int kScale>
class FixedPoint {
    static_assert(kScale >= 1, "FixedPoint requires kScale >= 1");

    int digits;

public:
    FixedPoint() = default;

    FixedPoint(int base, int fract = 0)
        : digits(base * kScale + fract)
    {
        if (!(fract < kScale && fract >= 0))
            qWarning() << base << fract << kScale;
        assert(fract < kScale && fract >= 0);
        assert(base < INT_MAX / kScale);
    }

    FixedPoint(double value)
        : FixedPoint(static_cast<int>(std::floor(value)),
                  static_cast<int>((value - std::floor(value)) * kScale))
    {
    }

    int base() const {
        return digits >= 0 ? digits / kScale : digits / kScale - 1;
    }

    int fract() const {
        return digits >= 0 ? digits % kScale : kScale - digits % kScale;
    }

    int const &rawDigits() const {
        return digits;
    }

    int &rawDigits() {
        return digits;
    }

    operator double() const {
        // assert(base() * kScale + fract() == digits);
        return (double)digits / kScale;
    }

    FixedPoint &operator+=(FixedPoint const &that) {
        digits += that.digits;
        return *this;
    }

    FixedPoint &operator-=(FixedPoint const &that) {
        digits -= that.digits;
        return *this;
    }

    FixedPoint &operator*=(FixedPoint const &that) {
        digits *= that.digits;
        digits /= kScale;
        return *this;
    }

    FixedPoint &operator/=(FixedPoint const &that) {
        digits *= kScale;
        digits /= that.digits;
        return *this;
    }

    FixedPoint operator+(FixedPoint const &that) const {
        FixedPoint tmp = *this;
        tmp += that;
        return tmp;
    }

    FixedPoint operator-(FixedPoint const &that) const {
        FixedPoint tmp = *this;
        tmp -= that;
        return tmp;
    }

    FixedPoint operator*(FixedPoint const &that) const {
        FixedPoint tmp = *this;
        tmp *= that;
        return tmp;
    }

    FixedPoint operator/(FixedPoint const &that) const {
        FixedPoint tmp = *this;
        tmp /= that;
        return tmp;
    }

    FixedPoint operator+() const {
        return *this;
    }

    FixedPoint operator-() const {
        FixedPoint tmp;
        tmp.digits = -digits;
        return tmp;
    }

    /*static FixedPoint from_string(std::string str, bool *ok = nullptr, int strBase = 10) {
        if (str.isEmpty()) {
            if (ok) *ok = false;
            return FixedPoint();
        }
        int sign = 1;
        if (str.front() == '-' && str.size() >= 2) {
            str = str.substr(1);
            sign = -1;
        }
        if (ok) *ok = true;
        size_t dot = str.find('.');
        if (dot != -1) {
            int base = dot != 0 ?
                std::stoi(str.substr(0, dot), nullptr, strBase) : 0;
            unsigned int fract = dot + 1 != str.size() ?
                             std::stoul(str.substr(dot + 1)) : 0;
            if (kScale == 1) {
                fract = 0;
            } else {
                while (fract && fract < kScale / strBase) {
                    fract *= 10;
                }
                if (fract >= kScale)
                    fract /= strBase;
            }
            return FixedPoint(sign * base, fract);
        } else {
            int base = std::stoi(str.substr(0, dot), nullptr, strBase);
            return FixedPoint(sign * base);
        }
    }*/
};
