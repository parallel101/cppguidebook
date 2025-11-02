#include <bit>
#include <fmt/format.h>
#include <functional>
#include <map>
#include <memory>
#include <typeinfo>

struct Base
{
    int i = 1;

    virtual ~Base() = default;
};

struct Derived : Base
{
    int j = 2;

    Derived() {
    }
};

struct Derived2 : Base
{
    float d = 1.0;

    Derived2() {
    }
};

struct Derived3 : Derived2
{
    float e = 2.0;

    Derived3() {
    }
};


void example_dynamic_cast()
{
    Base *b = new Derived3;

    if (auto d = dynamic_cast<Derived *>(b)) {
        fmt::println("is Derived! can cast");

    } else if (auto d = dynamic_cast<Derived2 *>(b)) {
        fmt::println("is Derived2! can cast");

    } else {
        fmt::println("unknown type");
    }
}


void example_bit_cast()
{
    float f = 1.0;
    int i = std::bit_cast<int>(f);
    fmt::println("0x{:08x}", i);
    f = std::bit_cast<float>(i);
    fmt::println("{}", f);
}



struct Dsdsdsd {
};


void example_c_cast()
{
    int i;
    int *pi = &i;

    float *pf = (float *)pi; // reinterpret_cast

    const Base *b = new Derived;
    Derived *d = (Derived *)b; // static_cast + const_cast
    // Derived *d = static_cast<Derived *>(const_cast<Base *>(b));

    const int c = 1;
    const int *pc = &c;
    int *pc2 = (int *)pc; // const_cast
    float *pf2 = (float *)pc; // reinterpret_cast + const_cast
    // float *pf2 = reinterpret_cast<float *>(const_cast<int *>(pc)); // reinterpret_cast + const_cast
}




void func(std::shared_ptr<Derived>)
{
}

std::vector<std::shared_ptr<Base>> bs;

int main()
{
    bs.push_back(std::make_shared<Derived>());
    bs.push_back(std::make_shared<Derived2>());

    std::shared_ptr<Base> b = std::make_shared<Derived>();
    if (auto d = std::dynamic_pointer_cast<Derived>(b)) {
        func(d);
    }

    std::unique_ptr<Base> ub = std::make_unique<Derived>();
    auto d = std::unique_ptr<Derived>(static_cast<Derived *>(ub.release()));

    auto ubp = ub.release();
    if (auto d = std::unique_ptr<Derived>(dynamic_cast<Derived *>(ubp))) {
    } else {
        ub.reset(ubp);
    }

    dynamic_cast<Derived *>(ub.get());
}
