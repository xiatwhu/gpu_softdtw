# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import torch
from softdtw_loss import SoftDTW

import numpy as np
import math

# ----------------------------------------------------------------------------------------------------------------------
def timed_run(a, b, a_len, b_len, sdtw):
    """
    Runs a and b through sdtw, and times the forward and backward passes.
    Assumes that a requires gradients.
    :return: timing, forward result, backward result
    """
    from timeit import default_timer as timer

    # Forward pass
    start = timer()
    #forward = sdtw(a, b, a_len, b_len)
    s_loss = sdtw(a, b, a_len, b_len)
    #loss = (s_loss / b_len).mean()
    loss = (s_loss / b_len).mean()
    print("Loss:", loss)
    end = timer()
    t = end - start

    #grad_outputs = torch.ones_like(forward)

    # Backward
    start = timer()
    loss.backward()
    print("")
    #grads = torch.autograd.grad(forward, a, grad_outputs=grad_outputs)[0]
    end = timer()

    # Total time
    t += end - start

    return t, s_loss, None

# ----------------------------------------------------------------------------------------------------------------------
def profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward):
    sdtw = SoftDTW(gamma=0.05, bandwidth=16, warp=40)
    n_iters = 6

    print("Profiling forward() + backward() times for batch_size={}, seq_len_a={}, seq_len_b={}, dims={}...".format(batch_size, seq_len_a, seq_len_b, dims))

    times_cpu = []
    times_gpu = []

    for i in range(n_iters):
        a_cpu = torch.rand((batch_size, seq_len_a, dims), requires_grad=True)
        b_cpu = torch.rand((batch_size, seq_len_b, dims))
        a_cpu_len = torch.LongTensor([seq_len_a] * batch_size)
        b_cpu_len = torch.LongTensor([seq_len_b] * batch_size)
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        a_gpu_len = a_cpu_len.cuda()
        b_gpu_len = b_cpu_len.cuda()

        # GPU
        t_gpu, forward_gpu, backward_gpu = timed_run(a_gpu, b_gpu, a_gpu_len, b_gpu_len, sdtw)

        if i > 0:  # Ignore the first time we run, in case this is a cold start (because timings are off at a cold start of the script)
            times_gpu += [t_gpu]

    # Average and log
    avg_gpu = np.mean(times_gpu)
    print("\tGPU:     ", avg_gpu)
    print()

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    from timeit import default_timer as timer

    torch.manual_seed(1234)

    profile(32, 3, 80, 80, tol_backward=1e-4)
    #profile(512, 256, 256, 2, tol_backward=1e-3)
    profile(1, 17, 15, 2, tol_backward=1e-6)
    profile(512, 64, 64, 2, tol_backward=1e-4)
    profile(64, 1025, 1023, 2, tol_backward=1e-4)
    profile(32, 1025, 64, 2, tol_backward=1e-4)
    profile(32, 1400, 5, 2, tol_backward=1e-4)
    profile(32, 5, 1400, 2, tol_backward=1e-4)
