import torch
from torch.autograd import Function
import numpy as np
#from torch.autograd.gradcheck import _as_tuple, zero_gradients, \
#    contiguous, make_jacobian, iter_tensors, iter_variables


class RodriguesFunction(Function):
    """
        rotation SE map
    """

    def forward(self, rotation):
        self.save_for_backward(rotation)
        rotation = rotation.clone()
        theta = torch.norm(rotation, p=2, dim=1).unsqueeze(-1)

        c = torch.cos(theta)
        s = torch.sin(theta)
        c1 = 1 - c
        itheta = self.compute_itheta(rotation, theta)
        rotation *= itheta

        rrt = self.compute_rrt(rotation)
        r_x = self.generate_skew_matrix(rotation)

        eye = torch.from_numpy(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])).type(rotation.type())
        eye = eye.repeat(rotation.size(0), 1, 1)
        c = c.expand(rotation.size(0), 9).contiguous().view(rotation.size(0), 3, 3)
        c1 = c1.expand(rotation.size(0), 9).contiguous().view(rotation.size(0), 3, 3)
        s = s.expand(rotation.size(0), 9).contiguous().view(rotation.size(0), 3, 3)

        R = c * eye + c1 * rrt + s * r_x
        return R.view(-1, 9)

    def compute_itheta(self, rotation, theta):
        itheta = torch.zeros((rotation.shape[0], 1)).type(rotation.type())
        mask = theta != 0
        itheta[mask] = 1. / theta[mask]
        return itheta

    def compute_rrt(self, rotation):
        rrt = torch.from_numpy(np.zeros((rotation.shape[0], 3, 3))).type(rotation.type())
        rrt[:, 0, 0] = rotation[:, 0] * rotation[:, 0]
        rrt[:, 0, 1] = rotation[:, 0] * rotation[:, 1]
        rrt[:, 0, 2] = rotation[:, 0] * rotation[:, 2]

        rrt[:, 1, 0] = rotation[:, 0] * rotation[:, 1]
        rrt[:, 1, 1] = rotation[:, 1] * rotation[:, 1]
        rrt[:, 1, 2] = rotation[:, 1] * rotation[:, 2]

        rrt[:, 2, 0] = rotation[:, 0] * rotation[:, 2]
        rrt[:, 2, 1] = rotation[:, 1] * rotation[:, 2]
        rrt[:, 2, 2] = rotation[:, 2] * rotation[:, 2]
        return rrt

    def compute_drrt(self, rotation):
        drrt = torch.from_numpy(np.zeros((rotation.shape[0], 3, 9))).type(rotation.type())

        drrt[:, 0, 0] = rotation[:, 0] + rotation[:, 0]
        drrt[:, 0, 1] = rotation[:, 1]
        drrt[:, 0, 2] = rotation[:, 2]
        drrt[:, 0, 3] = rotation[:, 1]
        drrt[:, 0, 6] = rotation[:, 2]

        drrt[:, 1, 1] = rotation[:, 0]
        drrt[:, 1, 3] = rotation[:, 0]
        drrt[:, 1, 4] = rotation[:, 1] + rotation[:, 1]
        drrt[:, 1, 5] = rotation[:, 2]
        drrt[:, 1, 7] = rotation[:, 2]

        drrt[:, 2, 2] = rotation[:, 0]
        drrt[:, 2, 5] = rotation[:, 1]
        drrt[:, 2, 6] = rotation[:, 0]
        drrt[:, 2, 7] = rotation[:, 1]
        drrt[:, 2, 8] = rotation[:, 2] + rotation[0, 2]

        return drrt

    def generate_skew_matrix(self, rotation):
        r_x = torch.from_numpy(np.zeros((rotation.shape[0], 3, 3))).type(rotation.type())
        r_x[:, 0, 1] = -rotation[:, 2]
        r_x[:, 0, 2] = rotation[:, 1]

        r_x[:, 1, 0] = rotation[:, 2]
        r_x[:, 1, 2] = -rotation[:, 0]

        r_x[:, 2, 0] = -rotation[:, 1]
        r_x[:, 2, 1] = rotation[:, 0]
        return r_x

    def backward(self, grad_output):
        f_rotation = self.saved_tensors[0].clone()
        data_type = f_rotation.type()
        theta = torch.norm(f_rotation, p=2, dim=1).unsqueeze(-1)
        c = torch.cos(theta)
        s = torch.sin(theta)
        c1 = 1 - c
        itheta = self.compute_itheta(f_rotation, theta)
        f_rotation *= itheta

        rrt = self.compute_rrt(f_rotation)
        r_x = self.generate_skew_matrix(f_rotation)

        J = torch.from_numpy(np.zeros((f_rotation.size(0), 3, 9))).type(data_type)
        generators = torch.from_numpy(np.zeros((f_rotation.size(0), 3, 9))).type(data_type)
        generators[:, 0, 5] = -1
        generators[:, 0, 7] = 1
        generators[:, 1, 2] = 1
        generators[:, 1, 6] = -1
        generators[:, 2, 1] = -1
        generators[:, 2, 3] = 1
        I = torch.from_numpy(np.eye(3)).type(data_type)
        drrt = self.compute_drrt(f_rotation)
        a2 = c1 * itheta
        a4 = s * itheta
        for i in range(3):
            ri = f_rotation[:, 0] if i == 0 else f_rotation[:, 1] if i == 1 else f_rotation[:, 2]
            ri = ri.unsqueeze(-1)
            a0 = -s * ri
            a1 = (s - 2 * c1 * itheta) * ri
            a3 = (c - s * itheta) * ri
            J[:, i, :] = a0 * I.view(9) + a1 * rrt.view(-1, 9) + a2 * drrt[:, i, :] + a3 * r_x.view(-1,
                                                                                                    9) + a4 * generators[
                                                                                                              :, i, :]

        # multiply the gradient with the output gradients
        return torch.bmm(J, grad_output.unsqueeze(-1)).squeeze(-1)


class RodriguesModule(torch.nn.Module):
    def forward(self, x):
        return RodriguesFunction()(x)


def get_analytical_jacobian(input, output):
    jacobian = make_jacobian(input, output.numel())
    jacobian_reentrant = make_jacobian(input, output.numel())
    grad_output = output.data.clone().zero_()
    flat_grad_output = grad_output.view(-1)
    reentrant = True
    correct_grad_sizes = True
    for i in range(flat_grad_output.numel()):
        flat_grad_output.zero_()
        flat_grad_output[i] = 1
        for jacobian_c in (jacobian, jacobian_reentrant):
            zero_gradients(input)
            output.backward(grad_output, create_graph=True)
            for jacobian_x, (d_x, x) in zip(jacobian_c, iter_variables(input)):
                if d_x is None:
                    jacobian_x[:, i].zero_()
                else:
                    if d_x.size() != x.size():
                        correct_grad_sizes = False
                    jacobian_x[:, i] = d_x
                    jacobian_x[:, i] = d_x.to_dense() if d_x.is_sparse else d_x

    for jacobian_x, jacobian_reentrant_x in zip(jacobian, jacobian_reentrant):
        if (jacobian_x - jacobian_reentrant_x).abs().max() != 0:
            reentrant = False
    return jacobian, reentrant, correct_grad_sizes

"""

def get_numerical_jacobian(fn, input, target, eps=1e-3):
    # To be able to use .view(-1) input must be contiguous
    input = contiguous(input)
    output_size = fn(input).numel()
    jacobian = make_jacobian(target, output_size)
    print("-----")
    print(target, output_size)

    # It's much easier to iterate over flattened lists of tensors.
    # These are reference to the same objects in jacobian, so any changes
    # will be reflected in it as well.
    x_tensors = [t for t in iter_tensors(target, True)]
    j_tensors = [t for t in iter_tensors(jacobian)]

    outa = torch.DoubleTensor(output_size)
    outb = torch.DoubleTensor(output_size)

    # TODO: compare structure
    for x_tensor, d_tensor in zip(x_tensors, j_tensors):
        flat_tensor = x_tensor.view(-1)
        for i in range(flat_tensor.nelement()):
            orig = flat_tensor[i]
            flat_tensor[i] = orig - eps
            outa.copy_(fn(input), broadcast=False)
            flat_tensor[i] = orig + eps
            outb.copy_(fn(input), broadcast=False)
            flat_tensor[i] = orig

            outb.add_(-1, outa).div_(2 * eps)
            d_tensor[i] = outb

    return jacobian

def gradcheck(func, inputs, eps=1e-6, atol=1e-5, rtol=1e-3):
    #Check gradients computed via small finite differences
     #  against analytical gradients

    #The check between numerical and analytical has the same behaviour as
    #numpy.allclose https://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html
    #meaning it check that
    #    absolute(a - n) <= (atol + rtol * absolute(n))
    #is true for all elements of analytical jacobian a and numerical jacobian n.

    #Args:
    #    func: Python function that takes Variable inputs and returns
    #        a tuple of Variables
    #    inputs: tuple of Variables
    #    eps: perturbation for finite differences
    #    atol: absolute tolerance
    #    rtol: relative tolerance

    #Returns:
    #    True if all differences satisfy allclose condition
    
    output = func(*inputs)
    output = _as_tuple(output)

    for i, o in enumerate(output):
        if not o.requires_grad:
            continue

        def fn(input):
            return _as_tuple(func(*input))[i].data

        numerical = get_numerical_jacobian(fn, inputs, inputs, eps)
        print("Numerical : {}".format(numerical))
        analytical, reentrant, correct_grad_sizes = get_analytical_jacobian(_as_tuple(inputs), o)
        print("Anatycal : {}".format(analytical))

        for a, n in zip(analytical, numerical):
            if not ((a - n).abs() <= (atol + rtol * n.abs())).all():
                return False

        if not reentrant:
            return False

        if not correct_grad_sizes:
            return False

    # check if the backward multiplies by grad_output
    zero_gradients(inputs)
    output = _as_tuple(func(*inputs))
    torch.autograd.backward(output, [o.data.new(o.size()).zero_() for o in output])
    var_inputs = list(filter(lambda i: isinstance(i, Variable), inputs))
    if not var_inputs:
        raise RuntimeError("no Variables found in input")
    for i in var_inputs:
        if i.grad is None:
            continue
        if not i.grad.data.eq(0).all():
            return False

    return True

if __name__ == '__main__':
    from torch.autograd import Variable
    import cv2
    import numpy as np
    import time

    angles = np.array([[0.123, 0.707, 0.435], [0.7, 0.76, 0.7]])

    torch_angles = torch.from_numpy(angles.copy())
    # torch_angles = torch_angles.repeat(128, 1)
    # torch_angles = torch_angles.cuda()
    torch_angles = Variable(torch_angles, requires_grad=True)
    f = RodriguesFunction()
    time_start = time.time()

    R = f(torch_angles)
    forward_time = time.time() - time_start
    # print("Rotation pytorch : {}".format(R))
    R = R.sum()
    time_start = time.time()
    R.backward()
    backward_time = time.time() - time_start
    R, jac = cv2.Rodrigues(angles[0])
    # print("Rotation Cv : {}".format(R))
    print("Jacobian pytorch: {}".format(torch_angles.grad))
    print("Jacobian CV : {}".format(torch.from_numpy(jac)))
    # print("Forward Time : {}".format(forward_time))
    # print("Backward Time : {}".format(backward_time))
    output = gradcheck(RodriguesFunction(), (torch_angles, ), eps=1e-4, atol=1e-3)
    print("Gradcheck result : {}".format(output))
    #J = torch_angles.grad.data.cpu().numpy()[0]
    #J = J.reshape((3, 3, 3))
    #print(J.sum(axis=2))
"""
