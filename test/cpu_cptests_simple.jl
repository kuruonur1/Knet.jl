using Base.Test
using CUDArt
using CUDNN
@show CUDNN_VERSION

import CUDNN.cudnnConvolutionForward
import CUDNN.cudnnConvolutionBackwardFilter
import CUDNN.cudnnConvolutionBackwardData
import CUDNN.cudnnGetConvolutionNdForwardOutputDim
import CUDNN.cudnnGetPoolingNdForwardOutputDim
import CUDNN.cudnnPoolingForward
import CUDNN.cudnnPoolingBackward
GPU=true
include(Pkg.dir("Knet/src/util/conv_pool_cpu.jl"))

srand(7)
@show padding=2
@show stride=2
vrange = 0:2
x = Array{Float32}(rand(vrange,8,8,1,1)); tx = CudaArray(x); @show x
w = Array{Float32}(rand(vrange,3,3,1,1)); tw = CudaArray(w); @show w
@show ydims = cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride)
@assert cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride) == cudnnGetConvolutionNdForwardOutputDim(x,w; padding=padding,stride=stride)
y = zeros(Float32,ydims); ty = CudaArray(y);
cudnnConvolutionForward(tx,tw,ty; padding=padding, stride=stride); y = to_host(ty); @show y
y2 = zeros(Float32,size(y));
cudnnConvolutionForward(x,w,y2; padding=padding, stride=stride); @show y2
@test_approx_eq y y2

# dy = rand(Float32, size(y)); tdy = CudaArray(dy); @show dy
@show x
dy = Array{Float32}(rand(vrange,size(y))); tdy = CudaArray(dy); @show dy
dw = zeros(Float32, size(w)); tdw = CudaArray(dw);
cudnnConvolutionBackwardFilter(tx,tdy,tdw; padding=padding, stride=stride); dw = to_host(tdw); @show dw
dw2 = zeros(Float32, size(w));
cudnnConvolutionBackwardFilter(x,dy,dw2; padding=padding, stride=stride); @show dw2
@test_approx_eq dw dw2

dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
cudnnConvolutionBackwardData(tw, tdy, tdx; padding=padding, stride=stride); dx = to_host(tdx); @show dx
dx2 = zeros(Float32, size(x));
cudnnConvolutionBackwardData(w, dy, dx2; padding=padding, stride=stride); @show dx2
@test_approx_eq dx dx2

#=
using CUDNN: PD, CUDNN_POOLING_MAX, cudnnGetPoolingNdForwardOutputDim
# x = rand(Float32,18,18,3,100); tx = CudaArray(x); @show x
x = reshape(Float32[1:25;], 5, 5, 1, 1); tx = CudaArray(x); @show x
psize, padding, stride = 3, 0, 3
pd1 = PD(2, psize, padding, stride, CUDNN_POOLING_MAX)
@assert cudnnGetPoolingNdForwardOutputDim(pd1, tx) == cudnnGetPoolingNdForwardOutputDim(x, window=psize, padding=padding, stride=stride, mode=0)
@show ydims = cudnnGetPoolingNdForwardOutputDim(x, window=psize, padding=padding, stride=stride, mode=0)
y = zeros(Float32, ydims); ty = CudaArray(y);
y2 = zeros(y)
cudnnPoolingForward(tx, ty; window=psize, padding=padding, stride=stride, mode=0); y = to_host(ty); @show y
cudnnPoolingForward(x, y2; window=psize, padding=padding, stride=stride, mode=0); @show y2
@test_approx_eq y y2

dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
dx2 = zeros(Float32, size(x));
dy = rand(Float32, size(y)); tdy = CudaArray(dy); @show dy
cudnnPoolingBackward(ty, tdy, tx, tdx; window=psize, padding=padding, stride=stride, mode=0); dx = to_host(tdx); @show dx
cudnnPoolingBackward(y, dy, x, dx2; window=psize, padding=padding, stride=stride, mode=0); @show dx2
@test_approx_eq dx dx2
=#
