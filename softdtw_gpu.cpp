#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor>
softdtw_cuda_forward(
        at::Tensor x, at::Tensor y,
        at::Tensor x_len, at::Tensor y_len,
        float gamma, float bandwidth, float warp);

std::vector<at::Tensor>
softdtw_cuda_backward(
        at::Tensor x, at::Tensor y,
        at::Tensor grad, at::Tensor W,
        at::Tensor x_len, at::Tensor y_len,
        float bandwidth);

#define CHECK_CUDA(x) \
    AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)  \
    CHECK_CUDA(x);      \
    CHECK_CONTIGUOUS(x)

std::vector<at::Tensor>
softdtw_forward(
        at::Tensor x, at::Tensor y,
        at::Tensor x_len, at::Tensor y_len,
        float gamma, float bandwidth, float warp) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(x_len);
    CHECK_INPUT(y_len);

    return softdtw_cuda_forward(x, y, x_len, y_len, gamma, bandwidth, warp);
}

std::vector<at::Tensor>
softdtw_backward(
        at::Tensor x, at::Tensor y,
        at::Tensor grad, at::Tensor W,
        at::Tensor x_len, at::Tensor y_len,
        float bandwidth) {
    CHECK_INPUT(x);
    CHECK_INPUT(y);
    CHECK_INPUT(grad);
    CHECK_INPUT(W);
    CHECK_INPUT(x_len);
    CHECK_INPUT(y_len);

    return softdtw_cuda_backward(x, y, grad, W, x_len, y_len, bandwidth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softdtw_forward, "softdtw forward (CUDA)");
    m.def("backward", &softdtw_backward, "softdtw backward (CUDA)");
}
