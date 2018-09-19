# circular convolution
using FFTW

function cconv(x::Array{Complex{T},D}, h::Array{Complex{T},D}) where {T,D}
    ifft( fft(x) .* fft(h))
end

function cconv(x::Array{T,D}, h::Array{T,D}) where {T<:Real,D}
    real(ifft( fft(x) .* fft(h)))
end

cconv(x::Array, h::Array) = cconv(promote(x,h)...)

# upsampler
function upsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D} = (fill(0,D)...,)) where {T,D}
    szx = size(x)
    output = zeros(T, szx .* factor)
    for ci in CartesianIndices(szx)
        output[((ci.I .- 1) .* factor .+ 1 .+ offset)...] = x[ci]
    end
    output
end

function downsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D} = (fill(0,D)...,)) where {T,D}
    szout = fld.(size(x), factor)
    output = similar(x, szout...)
    ci = CartesianIndices(szout)
    for idx = LinearIndices(szout)
        output[idx] = x[((ci[idx].I .- 1) .* factor .+ 1 .+ offset)...]
    end
    output
end

# matrix-formed CDFT operator for D-dimensional signal
cdftmtx(sz::NTuple) = cdftmtx(sz...)
cdftmtx(sz::Integer...) = cdftmtx(Float64, sz...)
cdftmtx(T::Type, sz::NTuple) = cdftmtx(T, sz...)
cdftmtx(::Type{Complex{T}}, sz...) where {T} = cdftmtx(T, sz...)

function cdftmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    len = prod(sz)

    imps = [ setindex!(zeros(T, sz), 1, idx) |> fft |> vec for idx in 1:len ]
    mtx = hcat(imps...)
    rm = Diagonal(Complex{T}[ exp(-1im*angle(mtx[n,end])/2) for n in 1:len ])

    rm * mtx / sqrt(T(len))
end

permdctmtx(sz::NTuple) = permdctmtx(sz...)
permdctmtx(sz::Integer...) = permdctmtx(Float64, sz...)
permdctmtx(T::Type, sz::NTuple) = permdctmtx(T, sz...)

function permdctmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    imps = [ setindex!(zeros(T, sz), 1, idx) |> dct |> vec for idx in 1:prod(sz) ]
    mtx = hcat(imps...)

    isevenids = map(ci->iseven(sum(ci.I .- 1)), CartesianIndices(sz)) |> vec
    permids = sortperm(isevenids; rev=true, alg=Base.DEFAULT_STABLE)

    vcat([ transpose(mtx[pi,:]) for pi in permids ]...)
end
