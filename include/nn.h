#ifndef NN_H
#define NN_H

#include "util.h"
#include "value.h"

class Module {
public:
    virtual std::vector<Value> parameters() { return {}; }

    void zero_grad() {
        for (auto val : parameters()) {
            val.zero_grad();
        }
    }

    void step() {
        for (auto val : parameters()) {
            val.step();
        }
    }
};

class Neuron : public Module {
    // y = wx + b
private:
    std::vector<Value> weight_;
    Value bias_;
    uint64_t seed_;
    bool nonlin_;

public:
    Neuron(int nin, bool nonlin = true, uint64_t seed = 41) : seed_(seed), nonlin_(nonlin) {
        weight_.reserve(nin);
        for (int i = 0; i < nin; i++) {
            weight_.emplace_back(rand(-1, 1));
        }
        bias_ = Value(0);
    }

    Value operator()(const std::vector<Value>& x) {
        Value out = bias_;
        assert(x.size() == weight_.size());
        for (int i = 0; i < weight_.size(); ++i) {
            out += weight_[i] * x[i];
        }
        return nonlin_ ? out.relu() : out;
    }

    std::vector<Value> parameters() override {
        std::vector<Value> params;
        for (auto& w : weight_) {
            params.push_back(w);
        }
        params.push_back(bias_);
        return params;
    }
};

class Layer : public Module {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int nin, int nout, bool nonlin = true) {
        for (int i = 0; i < nout; ++i) {
            neurons.emplace_back(nin, nonlin);
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& x) {
        std::vector<Value> out;
        for (auto& neuron : neurons) {
            out.push_back(neuron(x));
        }
        return out;
    }

    std::vector<Value> parameters() override {
        std::vector<Value> params;
        for (auto& neuron : neurons) {
            auto neuron_params = neuron.parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }
};

class MLP : public Module {
private:
    std::vector<Layer> layers;

public:
    MLP(int nin, const std::vector<int>& nouts) {
        // nin: input dim
        // nouts: output dims
        std::vector<int> sizes;
        sizes.push_back(nin);
        for (auto n : nouts) {
            sizes.push_back(n);
        }

        for (int i = 0; i < nouts.size(); ++i) {
            bool last_layer = (i == nouts.size() - 1);
            layers.emplace_back(sizes[i], sizes[i + 1], !last_layer);
            // We need a size[i] * size[i + 1] matrix to transform size[i] tensor to size[i + 1] tensor
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& x) {
        std::vector<Value> out = x;
        for (auto& layer : layers) {
            out = layer(out);
        }
        return out;
    }

    std::vector<Value> parameters() override {
        std::vector<Value> params;
        for (auto& layer : layers) {
            auto layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

#endif
