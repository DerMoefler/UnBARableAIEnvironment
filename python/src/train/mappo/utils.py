import math

import numpy as np
import torch
import torch.nn as nn


def check(input):
    return torch.from_numpy(input) if isinstance(input, np.ndarray) else input


def get_grad_norm(parameters):
    total_norm = 0
    for parameter in parameters:
        if parameter.grad is not None:
            total_norm += parameter.grad.norm() ** 2
    return math.sqrt(total_norm)


def huber_loss(error, delta):
    abs_error = torch.abs(error)
    quadratic = torch.minimum(abs_error, torch.as_tensor(delta, device=error.device, dtype=error.dtype))
    linear = abs_error - quadratic
    return 0.5 * quadratic ** 2 + delta * linear


def mse_loss(error):
    return error ** 2 / 2


class ValueNorm(nn.Module):
    def __init__(self, input_shape, device=torch.device("cpu"), epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.running_mean = torch.zeros(input_shape, device=device, dtype=torch.float32)
        self.running_var = torch.ones(input_shape, device=device, dtype=torch.float32)
        self.count = torch.tensor(epsilon, device=device, dtype=torch.float32)

    def update(self, values):
        values = check(values).to(device=self.running_mean.device, dtype=torch.float32)
        if values.ndim == 1:
            batch_mean = values.mean()
            batch_var = values.var(unbiased=False)
        else:
            batch_mean = values.mean(dim=0)
            batch_var = values.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(values.shape[0]), device=values.device)

        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count
        self.running_mean += delta * batch_count / total_count
        m_a = self.running_var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.running_var = m2 / total_count
        self.count = total_count

    def normalize(self, values):
        values = check(values).to(device=self.running_mean.device, dtype=torch.float32)
        return (values - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)

    def denormalize(self, values):
        values = check(values).to(device=self.running_mean.device, dtype=torch.float32)
        return values * torch.sqrt(self.running_var + self.epsilon) + self.running_mean