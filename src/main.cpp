#include "nn.h"
#include "value.h"

int main() {
    auto x = std::vector<Value>{2.0, 3.0, -1};
    auto y = Value(3.0);
    // (1, 3) -> (1, 4) -> (1, 4) -> (1, 4) -> (1, 1)
    // (3,4) -> (4,4) -> (4,4) -> (4, 1)
    auto model = MLP(3, {4, 4, 1});
    for (int epoch = 0; epoch < 100; epoch++) {
        auto y0 = model(x)[0];
        if (epoch % 10 == 0) {
            printf("epoch = %d, y = %.4f\n", epoch / 10, y0.data());
        }
        auto loss = 0.5 * (y0 - y).pow(2);
        model.zero_grad();
        loss.backward();
        model.step();
    }
}