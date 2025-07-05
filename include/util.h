#ifndef UTIL_H
#define UTIL_H

#include <random>

namespace {
#define SEED 40
}  // namespace

double rand(double l, double r) {
    std::mt19937 gen(SEED);
    std::uniform_real_distribution<> dis(l, r);
    return dis(gen);
}

#endif
