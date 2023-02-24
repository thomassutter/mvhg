import torch


class Sigmoid(torch.autograd.Function):
    """
    Surrogate gradient of the Heaviside step function.

    **Forward pass:** Heaviside step function shifted.
        .. math::
            S=\\begin{cases} 1 & \\text{if U ≥ U$_{\\rm thr}$} \\\\
            0 & \\text{if U < U$_{\\rm thr}$}
            \\end{cases}
    **Backward pass:** Gradient of sigmoid function.
        .. math::
            S&≈\\frac{1}{1 + {\\rm exp}(-kU)} \\\\
             \\frac{∂S}{∂U}&=\\frac{k {\\rm exp}(-kU)}{[{\\rm exp}(-kU)+1]^2}

    :math:`k` defaults to 25, and can be modified by calling ``surrogate.sigmoid(slope=25)``.

    Adapted from:
        *F. Zenke, S. Ganguli (2018):
            SuperSpike: Supervised Learning in Multilayer Spiking Neural
            Networks.
            Neural Computation, pp. 1514-1541.
        *
    Code adapted from https://snntorch.readthedocs.io
    """

    @staticmethod
    def forward(ctx, input_, slope=1.0):
        ctx.save_for_backward(input_)
        ctx.slope = slope
        out = (input_ >= 0).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        input_ = torch.clamp(input_, -10, 10)
        grad_input = grad_output.clone()
        grad = (
            grad_input
            * ctx.slope
            * torch.exp(-ctx.slope * input_)
            / ((torch.exp(-ctx.slope * input_) + 1) ** 2)
        )
        # if grad.isnan().any():
        #     for k in range(0, input_.shape[0]):
        #         if grad[k].isnan().any():
        #             print(input_[k])
        #             print(grad[k])
        #             print(grad_input[k])
        return grad, None


def sigmoid(slope=0.01):
    """Sigmoid surrogate gradient enclosed with a parameterized slope."""
    slope = slope

    def inner(x):
        return Sigmoid.apply(x, slope)

    return inner


if __name__ == "__main__":

    hside_approx = sigmoid(slope=0.1)
    n = 0
    x1 = torch.arange(5)
    x2 = n - x1
    m2 = 0
    check_x1 = n - x1
    check_x2 = m2 - torch.relu(n - x1)
    hside_check_x1 = hside_approx(check_x1)
    hside_check_x2 = hside_approx(check_x2)
    print(hside_check_x1)
    print(hside_check_x2)
