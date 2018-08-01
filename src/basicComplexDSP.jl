# circular convolution

function cconv(x::Array{Complex{T},D}, h::Array{Complex{T},D}) where {T,D}
    ifft( fft(x) .* fft(h))
end

function cconv(x::Array{T,D}, h::Array{T,D}) where {T<:Real,D}
    real(ifft( fft(x) .* fft(h)))
end

cconv(x::Array, h::Array) = cconv(promote(x,h)...)

# upsampler
function upsample(x::Array{T,D}, factor::NTuple{D}, offset::NTuple{D} = tuple(zeros(Integer,D)...)) where {T,D}
    szx = size(x)
    output = zeros(T, szx .* factor)
    for idx = 1:prod(szx)
        sub = ind2sub(szx, idx)
        output[((sub .- 1) .* factor .+ 1 .+ offset)...] = x[sub...]
    end
    output
end

function downsample(x::Array{T,D}, factor::NTuple{D}, offset::NTuple{D} = tuple(zeros(Integer,D)...)) where {T,D}
    szout = fld.(size(x), factor)
    output = Array{T,D}(szout...)
    for idx = 1:prod(szout)
        output[idx] = x[((ind2sub(szout,idx) .- 1) .* factor .+ 1 .+ offset)...]
    end
    output
end

# matrix-formed CDFT operator for D-dimensional signal
function cdftmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    len = prod(sz)

    imps = map(1:len) do idx
        u = zeros(T, sz)
        u[idx] = 1
        vec(fft(u))
    end
    mtx = hcat(imps...)

    rm = Diagonal(Complex{T}[ exp(-1im*angle(mtx[n,end])/2) for n in 1:len ])

    rm * mtx / sqrt(T(len))
end

cdftmtx(sz::Integer...) = cdftmtx(Float64, sz...)

function permdctmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    if prod(sz) == 1
        return ones(T,1,1)
    end
    len = prod(sz)

    imps = map(1:len) do idx
        u = zeros(T, sz)
        u[idx] = 1
        vec(dct(u))
    end
    mtx = hcat(imps...)

    evenIds = find(x -> iseven(sum(ind2sub(sz,x) .- 1)), 1:len)
    oddIds = find(x -> isodd(sum(ind2sub(sz,x) .- 1)), 1:len)

    evenMtx = hcat([ mtx[idx,:] for idx in evenIds]...)
    oddMtx = hcat([ mtx[idx,:] for idx in oddIds]...)

    vcat(transpose(evenMtx), transpose(oddMtx))
end

permdctmtx(sz::Integer...) = permdctmtx(Float64, sz...)
