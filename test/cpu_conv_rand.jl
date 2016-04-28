using Base.Test
using CUDArt
using CUDNN
@show CUDNN_VERSION

import CUDNN.cudnnConvolutionForward
import CUDNN.cudnnConvolutionBackwardFilter
import CUDNN.cudnnConvolutionBackwardData
import CUDNN.cudnnGetConvolutionNdForwardOutputDim
GPU=true
include(Pkg.dir("Knet/src/util/conv_pool_cpu.jl"))

srand(7)
#=
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
=#

function ctest()
    xw,xh = rand(20:30,2)
    psize = rand(6:9)
    padding = rand(0:5)
    stride = rand(1:9)
    @show xw, xh, psize, padding, stride
    x = Array{Float32}(rand(0:2,xw,xh,3,10)); tx = CudaArray(x);
    w = Array{Float32}(rand(0:2,psize,psize,3,5)); tw = CudaArray(w);
    @show ydims = cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride)
    @show cudnnGetConvolutionNdForwardOutputDim(x,w; padding=padding,stride=stride)
    @assert cudnnGetConvolutionNdForwardOutputDim(tx,tw; padding=padding,stride=stride) == cudnnGetConvolutionNdForwardOutputDim(x,w; padding=padding,stride=stride)
    y = zeros(Float32,ydims); ty = CudaArray(y);
    cudnnConvolutionForward(tx,tw,ty; padding=padding, stride=stride); y = to_host(ty);
    y2 = zeros(Float32,size(y));
    cudnnConvolutionForward(x,w,y2; padding=padding, stride=stride);
    @test_approx_eq y y2

    dy = rand(Float32, size(y)); tdy = CudaArray(dy);
    dy = Array{Float32}(rand(0:2,size(y))); tdy = CudaArray(dy);
    dw = zeros(Float32, size(w)); tdw = CudaArray(dw);
    cudnnConvolutionBackwardFilter(tx,tdy,tdw; padding=padding, stride=stride); dw = to_host(tdw)
    dw2 = zeros(Float32, size(w))
    cudnnConvolutionBackwardFilter(x,dy,dw2; padding=padding, stride=stride)
    @test_approx_eq dw dw2

    dx = zeros(Float32, size(x)); tdx = CudaArray(dx);
    cudnnConvolutionBackwardData(tw, tdy, tdx; padding=padding, stride=stride)
    dx2 = zeros(Float32, size(x));
    cudnnConvolutionBackwardData(w, dy, dx2; padding=padding, stride=stride)
    @test_approx_eq dx dx2
end

for i in 1:100 ctest() end
:ok
