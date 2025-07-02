#include "value.h"

#include <cstdio>

int main() {
    auto a = Value(-4.0);
    auto b = Value(2.0);
    auto c = a + b;
    auto d = a * b + b.pow(3);
    c += c + 1;
    c += 1 + c + (-a);
    d += d * 2 + (b + a).relu();
    d += 3 * d + (b - a).relu();
    auto e = c - d;
    auto f = e.pow(2);
    auto g = f / 2.0;
    g += 10.0 / f;

    printf("%.4f\n", g.data());
}