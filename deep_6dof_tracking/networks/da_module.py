import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambdar):
        ctx.constant = lambdar
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.constant, None


class GradientRecordHook(torch.autograd.Function):
    '''
    Simple way to record gradient
    '''

    def __init__(self):
        self.gradients = []
        self.mag = None
        self.std = None

    def forward(self, x):
        '''
        Do Nothing
        '''
        return x.view_as(x)

    def backward(self, grad_output):
        # only record the magnitude, return the original gradient
        #grad = grad_output.clone()
        #self.mag = torch.sqrt(torch.sum(grad ** 2))
        #self.std = torch.std(grad_output)
        return grad_output


class DomainAdaptation(torch.nn.Module):

    def __init__(self, input_dims):
        super(DomainAdaptation, self).__init__()
        self.always_turn_off_module_train = False
        self.classifier = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        # gradient hooks
        self.grad_record = GradientRecordHook()

    def forward(self, latent_vector, lambdar):
        if self.always_turn_off_module_train:
            self.eval()  # only in some finetune situation
        self.activation_map = {}
        x = latent_vector
        x = self.grad_record(x)
        x = GradientScale.apply(x, lambdar)
        logit = self.classifier(x)
        soft = F.log_softmax(logit, dim=1)

        return soft, logit

    def turn_off_module_train(self):
        self.always_turn_off_module_train = True