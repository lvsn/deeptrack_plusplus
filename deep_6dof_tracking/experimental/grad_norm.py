from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import torch
import torch.nn as nn


class GradNormFunction(Function):
    def forward(self, input):
        return input

    def backward(self, grad_output_input):
        print(grad_output_input)
        return grad_output_input


class GradNorm(torch.nn.Module):
    def __init__(self):
        super(GradNorm, self).__init__()
        self.init_losses = None

    def forward(self, input):
        if self.init_losses is None:
            self.init_losses = Variable(input.data.clone())
        input = GradNormFunction()(input)
        normalising_term = torch.sum(input / self.init_losses) / input.size(0)
        new_loss = (input / self.init_losses / normalising_term) ** 1.5
        return new_loss