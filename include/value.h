#ifndef VALUE_H
#define VALUE_H

#include <cassert>
#include <cmath>
#include <functional>
#include <memory>
#include <stack>
#include <unordered_set>
#include <vector>

class Value {
private:
    inline static int cnt = 0;
    static constexpr double lr = 1e-3;

    struct Node : public std::enable_shared_from_this<Node> {
        double data_;
        double grad_;
        std::string label_;
        std::string op_;
        std::vector<std::shared_ptr<Node>> prev_;
        std::function<void()> backward_;

        Node(double x, std::string label = "")
            : data_(x), grad_(0), op_(""), label_(label + "_" + std::to_string(cnt)) {}
    };

    std::shared_ptr<Node> node_;

public:
    Value(double x = 0) : node_(std::make_shared<Node>(x)) {}

    double data() const { return node_->data_; }

    double grad() const { return node_->grad_; }

    void zero_grad() { node_->grad_ = 0; }

    void step() { node_->data_ += -lr * grad(); }

    std::string label() const { return node_->label_; }

    std::string name() const {
        return label() + "_data=" + std::to_string(data()) + "_grad=" + std::to_string(grad());
    }

    Value operator+(const Value& other) const {
        auto result = Value(node_->data_ + other.node_->data_);
        auto& node = result.node_;
        node->prev_ = {node_, other.node_};
        node->backward_ = [node_x = node_, node_y = other.node_, result_node = node]() {
            node_x->grad_ += result_node->grad_;
            node_y->grad_ += result_node->grad_;
        };
        return result;
    }

    Value& operator+=(const Value& other) {
        auto result = *this + other;
        node_ = result.node_;
        return *this;
    }

    Value operator-(const Value& other) const {
        auto result = Value(node_->data_ - other.node_->data_);
        auto& node = result.node_;
        node->prev_ = {node_, other.node_};
        node->backward_ = [node_x = node_, node_y = other.node_, result_node = node]() {
            node_x->grad_ += result_node->grad_;
            node_y->grad_ -= result_node->grad_;
        };
        return result;
    }

    Value& operator-=(const Value& other) {
        auto result = *this - other;
        node_ = result.node_;
        return *this;
    }

    Value operator*(const Value& other) const {
        auto result = Value(node_->data_ * other.node_->data_);
        auto& node = result.node_;
        node->prev_ = {node_, other.node_};
        node->backward_ = [node_x = node_, node_y = other.node_, result_node = node]() {
            node_x->grad_ += result_node->grad_ * node_y->data_;
            node_y->grad_ += result_node->grad_ * node_x->data_;
        };
        return result;
    }

    Value& operator*=(const Value& other) {
        auto result = *this * other;
        node_ = result.node_;
        return *this;
    }

    Value operator/(const Value& other) const {
        auto result = Value(node_->data_ / other.node_->data_);
        auto& node = result.node_;
        node->prev_ = {node_, other.node_};
        node->backward_ = [node_x = node_, node_y = other.node_, result_node = node]() {
            node_x->grad_ += result_node->grad_ / node_y->data_;
            node_y->grad_ -= result_node->grad_ * node_x->data_ / (node_y->data_ * node_y->data_);
        };
        return result;
    }

    Value& operator/=(const Value& other) {
        auto result = *this / other;
        node_ = result.node_;
        return *this;
    }

    Value operator-() const {
        auto result = Value(-node_->data_);
        result.node_->prev_ = {node_};
        result.node_->backward_ = [node_x = node_, result_node = result.node_]() {
            node_x->grad_ -= result_node->grad_;
        };
        return result;
    }

    Value pow(double power) const {
        auto result = Value(std::pow(node_->data_, power));
        result.node_->prev_ = {node_};
        result.node_->backward_ = [node_x = node_, result_node = result.node_, power]() {
            node_x->grad_ += power * std::pow(node_x->data_, power - 1) * result_node->grad_;
        };
        return result;
    }

    Value relu() const {
        auto result = Value(std::max(node_->data_, 0.0));
        result.node_->prev_ = {node_};
        result.node_->backward_ = [node_x = node_, result_node = result.node_]() {
            node_x->grad_ += (node_x->data_ > 0) ? result_node->grad_ : 0;
        };
        return result;
    }

    void backward() {
        std::unordered_set<std::shared_ptr<Node>> visited;
        std::stack<std::shared_ptr<Node>> stack;

        std::function<void(std::shared_ptr<Node>)> dfs = [&](std::shared_ptr<Node> node) {
            if (visited.count(node)) {
                return;
            }
            visited.insert(node);
            for (auto& prev : node->prev_) {
                dfs(prev);
            }
            stack.push(node);
        };

        dfs(node_);
        node_->grad_ = 1.0;

        while (!stack.empty()) {
            auto node = stack.top();
            stack.pop();
            if (node->backward_) {
                node->backward_();
            }
        }
    }
};

Value operator+(double a, const Value& b) {
    return Value(a) + b;
}

Value operator-(double a, const Value& b) {
    return Value(a) - b;
}

Value operator*(double a, const Value& b) {
    return Value(a) * b;
}

Value operator/(double a, const Value& b) {
    return Value(a) / b;
}

#endif
