import softdtw_cuda
import torch
from torch.autograd import Function

class softdtwFunction(Function):
    @staticmethod
    def forward(ctx, x, y, x_len, y_len, gamma, bandwidth, warp):
        ctx.bandwidth = bandwidth
        loss, W = softdtw_cuda.forward(x, y, x_len, y_len, gamma, bandwidth, warp)
        ctx.save_for_backward(x, y, x_len, y_len, W)
        return loss

    @staticmethod
    def backward(ctx, grad):
        bandwidth = ctx.bandwidth
        x, y, x_len, y_len, W = ctx.saved_tensors
        outputs = softdtw_cuda.backward(x, y, grad, W, x_len, y_len, bandwidth);

        return outputs[0], None, None, None, None, None, None

class SoftDTW(torch.nn.Module):
    def __init__(self, gamma=1.0, bandwidth=None, warp=0):
        super(SoftDTW, self).__init__()
        self.gamma = gamma
        self.bandwidth = float(1024) if bandwidth is None else float(bandwidth)
        self.warp = warp

    def forward(self, X, Y, X_len, Y_len):
        return softdtwFunction.apply(X, Y, X_len, Y_len, self.gamma, self.bandwidth, self.warp)

