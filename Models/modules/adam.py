import torch
from typing import Dict

class CustomAdamOptimizer:
    def __init__(self, params:Dict, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Timestamp

        self.moment1 = {}
        self.moment2 = {}

        for key in params.keys():
            self.moment1[key] = torch.zeros_like(params[key])
            self.moment2[key] = torch.zeros_like(params[key])

    def step(self, params:Dict, closure=None):
        self.t += 1

        if closure is not None:
            closure()

        for i, key in enumerate(params.keys()):
            grad = params[key].grad
            if grad is None:
                continue

            self.moment1[key] = self.beta1 * self.moment1[key] + (1 - self.beta1) * grad
            self.moment2[key] = self.beta2 * self.moment2[key] + (1 - self.beta2) * grad**2

            # Bias correction
            moment1_hat = self.moment1[key] / (1 - self.beta1**self.t)
            moment2_hat = self.moment2[key] / (1 - self.beta2**self.t)

            params[key].data -= self.lr * moment1_hat / (torch.sqrt(moment2_hat) + self.epsilon)
        return params

    def zero_grad(self):
        for key in self.moment1.keys():
            self.moment1[key].zero_()
            self.moment2[key].zero_()
        # for moment1, moment2 in zip(self.moment1, self.moment2):
        #     moment1.zero_()
        #     moment2.zero_()