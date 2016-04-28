if GPU
    import CUDNN: cudnnConvolutionForward, cudnnConvolutionBackwardFilter, cudnnConvolutionBackwardData, cudnnPoolingForward, cudnnPoolingBackward
    using CUDNN: CUDNN_CONVOLUTION, CUDNN_CROSS_CORRELATION, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, CUDNN_CONVOLUTION_FWD_ALGO_FFT, CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
else
    const CUDNN_CONVOLUTION = (UInt32)(0)
    const CUDNN_CROSS_CORRELATION = (UInt32)(1)
    const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = (UInt32)(0)
    const CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = (UInt32)(1)
    const CUDNN_CONVOLUTION_FWD_ALGO_GEMM = (UInt32)(2)
    const CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = (UInt32)(3)
    const CUDNN_CONVOLUTION_FWD_ALGO_FFT = (UInt32)(4)
    const CUDNN_POOLING_MAX = (UInt32)(0)
    const CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = (UInt32)(1)
    const CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = (UInt32)(2)
end

function _conv2_dw_gemm{T}(x0::Array{T,2}, dy::Array{T,2}, w::Array{T,2}; pad=0, stride=1, xcorr=true)
    if pad > 0 # this could be handled better...
        x=zeros(eltype(x0),map(m->2pad+m,size(x0))) 
        x[pad+1:end-pad,pad+1:end-pad] = x0
    else
        x=x0
    end
    x1l = last(collect(take(countfrom(1,stride),size(dy,1))))
    x2l = last(collect(take(countfrom(1,stride),size(dy,2))))
    widx = Int[sub2ind(size(x),i,j) for i in 1:size(w,1), j in 1:size(w,2)]
    oidx = Int[sub2ind(size(x),i,j) for i in 1:stride:x1l, j in 1:stride:x2l] # linear indexes of elements in a filter window
    destidx = Int[i+(j-1) for i in widx, j in oidx]
    return reshape(x[destidx]*(xcorr ? dy[:] : reverse(dy[:])),size(w))
end

function _conv2_dx_gemm{T}(dy::Array{T,2}, w::Array{T,2}, dx::Array{T,2}; pad=0, stride=1, xcorr=true)
    # x = y+w-1-2p  s=1
    # size_tdy = collect(size(dx)) + collect(size(w)) - 1
    size_tdy = collect(size(dx)) + collect(size(w)) - 1 + 2pad
    tdy = zeros(T, size_tdy...)

    pad1, pad2 = map(x->x-1,size(w))
    for (i,idy) in zip(countfrom(pad1+1,stride), 1:size(dy,1)), (j,jdy) in zip(countfrom(pad2+1,stride), 1:size(dy,2))
        tdy[i,j] = dy[idy,jdy]
    end
    # @show tdy
    res = _conv2_gemm(tdy, w; xcorr=true)
    # @show res
    return pad == 0 ? res : res[pad+1:end-pad,pad+1:end-pad] 
end

function _conv2_gemm{T}(x0::Array{T,2}, w::Array{T,2}; pad=0, stride=1, xcorr=false)
    if pad > 0 # this could be handled better...
        x=zeros(eltype(x0),map(m->2pad+m,size(x0))) 
        x[pad+1:end-pad,pad+1:end-pad] = x0
    else
        x=x0
    end
    window = size(w,1)
    row_extend, col_extend = floor(Int, 1 + (collect(size(x)) - collect(size(w))) / stride)
    # row_extend = size(x,1)-window+1
    # col_extend = size(x,2)-window+1
    # @show widx = Int[(j-1)*size(x,1)*stride+i for i in 1:stride:size(x,1), j in 1:col_extend] # linear indexes of filter positions in x
    widx = Int[sub2ind(size(x),i,j) for i in 1:stride:size(x,1)-window+1, j in 1:stride:size(x,2)-window+1] # linear indexes of filter positions in x

    oidx = Int[(j-1)*size(x,1)+i for i in 1:window, j in 1:window] # linear indexes of elements in a filter window
    # @show oidx = [(j-1)*size(A,1)+i for i in window:-1:1, j in window:-1:1]
    destidx = Int[i+(j-1) for i in widx, j in oidx]
    # println(x[destidx])
    # println(w[:])
    # println(reverse(w[:]))
    return reshape(x[destidx]*(xcorr ? w[:] : reverse(w[:])),row_extend,col_extend)
    # return reshape(x[destidx]*(xcorr ? w[:] : rot180(w)[:]),row_extend,col_extend)
end

function _conv2{T}(x::Array{T,2}, w::Array{T,2}; pad=0, stride=1, xcorr=false)
    max_pad = map(x->x-1-pad,size(w))
    y = conv2(x, xcorr ? rot180(w) : w)
    return y[1+max_pad[1]:stride:end-max_pad[1], 1+max_pad[2]:stride:end-max_pad[2]]
end

function cudnnConvolutionForward{T}(x::Array{T,4}, w::Array{T,4}, y::Array{T,4}; padding=0, stride=1, 
                                    upscale=1, mode=CUDNN_CONVOLUTION, cd=nothing,
                                    algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                    workSpace=0, workSpaceSizeInBytes=0, alpha=1, beta=1,im2col=1)
    # x: (W,H,C,N)
    # w: (W,H,C,K) 
    # y: (W,H,K,N) 
    fill!(y,0)
    @assert (upscale==1 && mode==CUDNN_CONVOLUTION && algorithm == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM) "$((upscale,mode,algorithm))"
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert Cx==Cw

    @inbounds for n in 1:N, k in 1:K, c in 1:Cx
        y[:,:,k,n] += _conv2_gemm(x[:,:,c,n], w[:,:,c,k]; pad=padding, stride=stride, xcorr=mode!=0)
    end
    return y
end

# dw = rot180(xcorr(x,dy))
function cudnnConvolutionBackwardFilter{T}(x::Array{T,4}, dy::Array{T,4}, dw::Array{T,4}; padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
    # x:    (Wx,Hx,Cx,N)
    # dy:   (Wy,Hy,K,N) 
    # dw:    (Ww,Hw,Cw,K) 
    fill!(dw,0)
    @assert (upscale==1&& mode==CUDNN_CONVOLUTION)
    Wx,Hx,C,Nx = size(x)
    Wy,Hy,K,Ny = size(dy)
    @inbounds for c in 1:C, k in 1:K, n in 1:Ny
        dw[:,:,c,k] += rot180(_conv2_dw_gemm(x[:,:,c,n], dy[:,:,k,n], dw[:,:,c,k]; pad=padding, stride=stride, xcorr=true)) # TODO
    end
    return dw
end

# dx = xcorr(dy, w, 'full')
function cudnnConvolutionBackwardData{T}(w::Array{T,4}, dy::Array{T,4}, dx::Array{T,4}; padding=0, stride=1, upscale=1, mode=CUDNN_CONVOLUTION)
    fill!(dx,0)
    @assert (upscale==1&& mode==CUDNN_CONVOLUTION)
    Wy,Hy,Ky,N = size(dy)
    Ww,Hw,C,Kw = size(w)
    @assert Ky==Kw
    @inbounds for n in 1:N, c in 1:C, k in 1:Kw
        t = _conv2_dx_gemm(dy[:,:,k,n], w[:,:,c,k], dx[:,:,c,n]; pad=padding, stride=stride, xcorr=true)
        dx[:,:,c,n] += t
    end
    return dx
end


function cudnnPoolingForward{T}(x::Array{T,4}, y; window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX)
    stride = isa(stride, Integer) ? (stride, stride) : stride
    window = isa(window, Integer) ? (window,window) : window
    padding = isa(padding, Integer) ? (padding,padding) : padding
    if any(map(x->x>0,padding))
        x0=x
        w,h,c,n = size(x0)
        x=zeros(eltype(x0),w+2padding[1],h+2padding[2],c,n)
        x[padding[1]+1:end-padding[1], padding[2]+1:end-padding[2],:,:] = x0
    end
    fill!(y,0)
    @assert (mode==CUDNN_POOLING_MAX)
    # x: (W,H,C,N)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert (Nx == Ny && C==K)
    # @inbounds for n in 1:Nx, c in 1:C, j in 1:stride[2]:Hx, i in 1:stride[1]:Wx
    @inbounds for n in 1:Nx, c in 1:C, jy in 1:Hy, iy in 1:Wy
        # iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
        i, j = 1+stride[1]*(iy-1), 1+stride[2]*(jy-1)
        hx_end = j+window[2]-1 > Hx ? Hx : j+window[2]-1
        wx_end = i+window[1]-1 > Wx ? Wx : i+window[1]-1
        y[iy,jy,c,n] = maximum(x[i:wx_end,j:hx_end,c,n])
    end
    return y
end

function cudnnPoolingBackward{T}(y::Array{T,4}, dy::Array{T,4}, x::Array{T,4}, dx::Array{T,4}; window=2, padding=0, stride=window, mode=CUDNN_POOLING_MAX)
    stride = isa(stride, Integer) ? (stride, stride) : stride
    window = isa(window, Integer) ? (window,window) : window
    padding = isa(padding, Integer) ? (padding,padding) : padding
    fill!(dx,0)
    @assert mode==CUDNN_POOLING_MAX
    # x: (W,H,C,N)
    if any(map(x->x>0,padding))
        x0=x
        w,h,c,n = size(x0)
        x=zeros(eltype(x0),w+2padding[1],h+2padding[2],c,n)
        x[padding[1]+1:end-padding[1], padding[2]+1:end-padding[2],:,:] = x0
    end
    dx1 = zeros(x)
    Wx,Hx,C,Nx = size(x);
    Wy,Hy,K,Ny = size(y);
    @assert (Nx == Ny && C==K)
    # @inbounds for n in 1:Nx, c in 1:C, j in 1:stride[2]:Hx, i in 1:stride[1]:Wx
    @inbounds for n in 1:Nx, c in 1:C, jy in 1:Hy, iy in 1:Wy
        #= iy, jy = div(i,stride[1])+1, div(j,stride[2])+1
        hx_end = j+window[2]-1 > Hx ? Hx : j+window[2]-1
        wx_end = i+window[1]-1 > Hx ? Hx : i+window[1]-1 =#
        i, j = 1+stride[1]*(iy-1), 1+stride[2]*(jy-1)
        hx_end = j+window[2]-1 > Hx ? Hx : j+window[2]-1
        wx_end = i+window[1]-1 > Wx ? Wx : i+window[1]-1
        a = x[i:wx_end,j:hx_end,c,n]
        di,dj = ind2sub(a,indmax(a))
        # dx[i+di-1-padding[1],j+dj-1-padding[2],c,n] += dy[iy,jy,c,n]
        dx1[i+di-1,j+dj-1,c,n] += dy[iy,jy,c,n]
    end
    dx = dx1[padding[1]+1:end-padding[1],padding[2]+1:end-padding[2]]
    return dx
end

function cudnnGetConvolutionNdForwardOutputDim{T}(x::Array{T,4}, w::Array{T,4}; padding=padding,stride=stride)
    padding = isa(padding, Integer) ? [padding,padding] : collect(padding)
    stride = isa(stride, Integer) ? [stride,stride] : collect(stride)
    Wx,Hx,Cx,N = size(x)
    Ww,Hw,Cw,K = size(w)
    @assert Cx==Cw
    Wy,Hy = floor(Int, 1 + (Int[Wx,Hx] + 2*padding - Int[Ww,Hw]) ./ stride)
    return (Wy,Hy,K,N)
end

function cudnnGetPoolingNdForwardOutputDim{T}(x::Array{T,4}; window=2, padding=0, stride=1, mode=CUDNN_POOLING_MAX)
    window = isa(window, Integer) ? (window,window) : window
    padding = isa(padding, Integer) ? (padding,padding) : padding
    stride = isa(stride, Integer) ? (stride,stride) : stride
    @assert reduce(&, [w>p for (p,w) in zip(padding,window)])
    dims = [size(x)...]
    for i=1:length(dims)-2
        # dims[i] = 1 + ceil((dims[i] + 2*padding[i] - window[i]) / stride[i])
        dims[i] = length(1:stride[i]:dims[i]+padding[i])
    end
    tuple(dims...)
end
