#include <cassert>
#include <cmath>

class Value {
public:
    Value(double x) : data_(x) { grad_ = 0; }

    double data() const { return data_; }

    Value operator+(const Value& other) const { return data_ + other.data(); }

    void operator+=(const Value& other) { data_ += other.data(); }

    Value operator-(const Value& other) const { return data_ - other.data(); }

    void operator-=(const Value& other) { data_ -= other.data(); }

    Value operator*(const Value& other) const { return data_ * other.data(); }

    void operator*=(const Value& other) { data_ *= other.data(); }

    Value operator/(const Value& other) const { return data_ / other.data(); }

    void operator/=(const Value& other) { data_ /= other.data(); }

    Value operator-() const { return -data(); }

    Value pow(double power) const { return ::pow(data_, power); }

    Value relu() const { return fmax(data(), 0); }

private:
    double data_;
    double grad_;
};

Value operator+(double a, const Value& b) {
    return a + b.data();
}

Value operator-(double a, const Value& b) {
    return a - b.data();
}

Value operator*(double a, const Value& b) {
    return a * b.data();
}

Value operator/(double a, const Value& b) {
    return a / b.data();
}