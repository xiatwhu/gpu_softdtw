#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#define SHFL_MASK 0xffffffff
#define warp_size 32

template <int num, typename T>
__inline__ __device__
T warp_reduce_sum(T val) {
    #pragma unroll
    for (int st = num; st > 0; st >>= 1) {
        val += __shfl_down_sync(SHFL_MASK, val, st);
    }
    return val;
}

// thread 0 return sum
template <int NT, typename T>
__inline__ __device__
T block_reduce_sum(T val, int tid) {
    static __shared__ T s_rec[NT / warp_size];
    const int lane = tid % 32;
    const int wid = tid / 32;
    val = warp_reduce_sum<warp_size / 2, T>(val);
    if (lane == 0) {
        s_rec[wid] = val;
    }
    __syncthreads();
    val = (tid < (NT / warp_size)) ? s_rec[lane] : 0;
    if (wid == 0) {
        val = warp_reduce_sum<NT / warp_size, T>(val);
    }
    return val;
}

template <int NT = 128>
__global__ void l1_distance_forward_kernel(
        const float* x, const float* y, float* D, int max_x_len, int max_y_len, int dim) {
    // x [batch, x, dim]
    // y [batch, y, dim]
    // D [batch, x, y]
    int batch_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int inner_id = blockIdx.y;

    int count = max_x_len * max_y_len;
    int stride = gridDim.y;

    x = x + batch_id * max_x_len * dim;
    y = y + batch_id * max_y_len * dim;
    D = D + batch_id * max_x_len * max_y_len;

    for (int i = inner_id; i < count; i += stride) {
        int x_index = i / max_y_len;
        int y_index = i % max_y_len;

        const float* cur_x = x + x_index * dim;
        const float* cur_y = y + y_index * dim;

        float sum = 0.f;
        int tid = thread_id;
        while (tid < dim) {
            sum += fabsf(cur_x[tid] - cur_y[tid]);
            tid += NT;
        }
        sum = block_reduce_sum<NT, float>(sum, thread_id);
        if (thread_id == 0) {
            D[x_index * max_y_len + y_index] = sum;
        }
        __syncthreads();
    }
}

template <int NT = 128>
__global__ void l1_distance_backward_kernel(
        const float* x, const float* y, const float* D_grad, float* x_grad,
        int max_x_len, int max_y_len, int dim) {
    // x [batch, x, dim]
    // y [batch, y, dim]
    // d_grad [batch, x, y]
    // x_grad [batch, x, dim]
    int batch_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int inner_id = blockIdx.y;

    int count = dim * max_x_len;
    int stride = gridDim.y;

    x = x + batch_id * max_x_len * dim;
    y = y + batch_id * max_y_len * dim;
    D_grad = D_grad + batch_id * max_x_len * max_y_len;
    x_grad = x_grad + batch_id * max_x_len * dim;

    for (int i = inner_id; i < count; i += stride) {
        int x_index = i / dim;
        int d_index = i % dim;

        const float* cur_d_grad = D_grad + x_index * max_y_len;
        float cur_x = x[x_index * dim + d_index];

        float sum = 0.0f;
        int tid = thread_id;
        while (tid < max_y_len) {
            float cur_y = y[tid * dim + d_index];
            if (cur_x > cur_y) {
                sum += cur_d_grad[tid];
            } else if (cur_x < cur_y) {
                sum -= cur_d_grad[tid];
            }
            tid += NT;
        }
        sum = block_reduce_sum<NT, float>(sum, thread_id);
        if (thread_id == 0) {
            x_grad[x_index * dim + d_index] = sum;
        }
        __syncthreads();
    }
}

__global__ void softdtw_cuda_forward_kernel(
        const float* D, const int64_t* x_len, const int64_t* y_len,
        float* r, float* loss, float* w,
        float gamma, float bandwidth, float warp,
        int max_x_len, int max_y_len) {
    // D [batch, N, M]
    // r [batch, N + 2, M + 2]
    // loss [batch]
    // weight [batch, N + 2, M + 2, 3]

    // y = -x + 0.5 + k
    // y = M/N * x + bandwidth
    // y = M/N * x - bandwidth

    int batch_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int stride = blockDim.x;

    D = D + batch_id * max_x_len * max_y_len;
    r = r + batch_id * (max_x_len + 2) * (max_y_len + 2);
    w = w + batch_id * (max_x_len + 2) * (max_y_len + 2) * 3;

    int M = y_len[batch_id];
    int N = x_len[batch_id];
    if (thread_id == 0) {
        r[0] = 0;
    }
    __syncthreads();

    float scale = 1.f * N / (N + M);
    float inv_gamma = 1.f / gamma;
    for (int i = 0; i < M + N - 1; ++i) {
        int min_x = max(0, i - M);
        int max_x = min(N - 1, i);
        int point0_x = max(min_x, static_cast<int>((i + 0.5f - bandwidth) * scale));
        // int point0_y = -point0_x + i;
        int point1_x = min(max_x, static_cast<int>((i + 0.5f + bandwidth) * scale));
        // int point1_y = -point1_x + i;
        int count = point1_x - point0_x + 1;

        for (int j = thread_id; j < count; j += stride) {
            int px = point0_x + j;
            int py = -px + i;
            float* cur_w = w + ((px + 1) * (max_y_len + 2) + py + 1) * 3;

            float r0 = -(r[px * (max_y_len + 2) + py]) * inv_gamma;
            float r1 = -(r[px * (max_y_len + 2) + py + 1] + warp) * inv_gamma;
            float r2 = -(r[(px + 1) * (max_y_len + 2) + py] + warp) * inv_gamma;
            float rmax = max(r0, max(r1, r2));
            float a = expf(r0 - rmax);
            float b = expf(r1 - rmax);
            float c = expf(r2 - rmax);
            float rsum = a + b + c;
            float inv_rsum = 1.f / rsum;
            cur_w[0] = a * inv_rsum;
            cur_w[1] = b * inv_rsum;
            cur_w[2] = c * inv_rsum;

            float softmin = -gamma * (logf(rsum) + rmax);
            r[(px + 1) * (max_y_len + 2) + py + 1] = D[px * max_y_len + py] + softmin;
        }
        __syncthreads();
    }
    if (thread_id == 0) {
        loss[batch_id] = r[N * (max_y_len + 2) + M];
    }
}

__global__ void softdtw_cuda_backward_kernel(
        float* w, const int64_t* x_len, const int64_t* y_len,
        const float* grad, float* e, float* D_grad,
        float bandwidth, int max_x_len, int max_y_len) {
    // grad [batch]
    // w [batch, N + 2, M + 2, 3]
    // e [batch, N + 2, M + 2]
    // D_grad [batch, N, M]

    // y = -x + 0.5 + k
    // y = M/N * x + bandwidth
    // y = M/N * x - bandwidth

    int batch_id = blockIdx.x;
    int thread_id = threadIdx.x;
    int stride = blockDim.x;

    D_grad = D_grad + batch_id * max_x_len * max_y_len;
    e = e + batch_id * (max_x_len + 2) * (max_y_len + 2);
    w = w + batch_id * (max_x_len + 2) * (max_y_len + 2) * 3;
    float cur_grad = grad[batch_id];

    int M = y_len[batch_id];
    int N = x_len[batch_id];
    if (thread_id == 0) {
        int offset = (N + 1) * (max_y_len + 2) + M + 1;
        e[offset] = cur_grad;
        w[3 * offset + 0] = 1.f;
    }
    __syncthreads();

    float scale = 1.f * N / (N + M);
    for (int i = M + N - 2; i >= 0 ; --i) {
        int min_x = max(0, i - M);
        int max_x = min(N - 1, i);
        int point0_x = max(min_x, static_cast<int>((i + 0.5f - bandwidth) * scale));
        // int point0_y = -point0_x + i;
        int point1_x = min(max_x, static_cast<int>((i + 0.5f + bandwidth) * scale));
        // int point1_y = -point1_x + i;
        int count = point1_x - point0_x + 1;

        for (int j = thread_id; j < count; j += stride) {
            int px = point0_x + j;
            int py = -px + i;

            float a = w[((px + 2) * (max_y_len + 2) + py + 1) * 3 + 1];
            float b = w[((px + 1) * (max_y_len + 2) + py + 2) * 3 + 2];
            float c = w[((px + 2) * (max_y_len + 2) + py + 2) * 3 + 0];
            float val =
                    e[(px + 2) * (max_y_len + 2) + py + 1] * a +
                    e[(px + 1) * (max_y_len + 2) + py + 2] * b +
                    e[(px + 2) * (max_y_len + 2) + py + 2] * c;
            e[(px + 1) * (max_y_len + 2) + py + 1] = val;
            D_grad[px * max_y_len + py] = val;
        }
        __syncthreads();
    }
}

std::vector<at::Tensor>
softdtw_cuda_forward(
        at::Tensor x, at::Tensor y,
        at::Tensor x_len, at::Tensor y_len,
        float gamma, float bandwidth, float warp) {

    at::DeviceGuard g(x.device());
    const int batch = x.size(0);
    const int max_x_len = x.size(1);
    const int dim = x.size(2);
    auto options = x.options();

    const int max_y_len = y.size(1);

    // [B, N, M]
    auto D = at::empty({batch, max_x_len, max_y_len}, options);
    // [B, N + 2, M + 2]
    auto R = at::empty({batch, max_x_len + 2, max_y_len + 2}, options);
    // [B, N + 2, M + 2, 3]
    auto W = at::zeros({batch, max_x_len + 2, max_y_len + 2, 3}, options);
    auto loss = at::empty({batch}, options);
    auto stream = at::cuda::getCurrentCUDAStream();

    R.fill_(INFINITY);

    {
        const int nt = 128;
        const int max_blocks = 40960;
        int grid_x = batch;
        int grid_y = max_blocks / grid_x;
        grid_y = std::min(grid_y, max_x_len * max_y_len);
        dim3 grads(grid_x, grid_y);

        l1_distance_forward_kernel<nt><<<grads, nt, 0, stream>>>(
                x.data_ptr<float>(),
                y.data_ptr<float>(),
                D.data_ptr<float>(),
                max_x_len,
                max_y_len,
                dim);
    }

    {
        const int nt = 1024;
        softdtw_cuda_forward_kernel<<<batch, nt, 0, stream>>>(
                D.data_ptr<float>(),
                x_len.data_ptr<int64_t>(),
                y_len.data_ptr<int64_t>(),
                R.data_ptr<float>(),
                loss.data_ptr<float>(),
                W.data_ptr<float>(),
                gamma, bandwidth, warp,
                max_x_len, max_y_len);
    }
    return {loss, W};
}

std::vector<at::Tensor>
softdtw_cuda_backward(
        at::Tensor x, at::Tensor y,
        at::Tensor grad, at::Tensor W,
        at::Tensor x_len, at::Tensor y_len,
        float bandwidth) {

    at::DeviceGuard g(x.device());
    const int batch = W.size(0);
    const int max_x_len = x.size(1);
    const int max_y_len = y.size(1);
    const int dim = x.size(2);
    auto options = grad.options();

    // [B, N, M]
    auto D_grad = at::zeros({batch, max_x_len, max_y_len}, options);
    // [B, N + 2, M + 2]
    auto E = at::zeros({batch, max_x_len + 2, max_y_len + 2}, options);
    // [B, N, dim]
    auto x_grad = at::empty({batch, max_x_len, dim}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    {
        const int nt = 1024;
        softdtw_cuda_backward_kernel<<<batch, nt, 0, stream>>>(
                W.data_ptr<float>(),
                x_len.data_ptr<int64_t>(),
                y_len.data_ptr<int64_t>(),
                grad.data_ptr<float>(),
                E.data_ptr<float>(),
                D_grad.data_ptr<float>(),
                bandwidth, max_x_len, max_y_len);
    }

    {
        const int nt = 512;
        const int max_blocks = 40960;
        int grid_x = batch;
        int grid_y = max_blocks / grid_x;
        grid_y = std::min(grid_y, max_x_len * dim);
        dim3 grads(grid_x, grid_y);

        l1_distance_backward_kernel<nt><<<grads, nt, 0, stream>>>(
                x.data_ptr<float>(),
                y.data_ptr<float>(),
                D_grad.data_ptr<float>(),
                x_grad.data_ptr<float>(),
                max_x_len,
                max_y_len,
                dim);
    }
    return {x_grad, E};
}
