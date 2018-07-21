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
    foreach(CartesianRange(szx)) do cr
        output[((cr.I .- 1) .* factor .+ 1 .+ offset)...] = x[cr.I...]
    end
    output
end

function downsample(x::Array{T,D}, factor::NTuple{D}, offset::NTuple{D} = tuple(zeros(Integer,D)...)) where {T,D}
    map(CartesianRange(fld.(size(x), factor))) do cr
        x[((cr.I .- 1) .* factor .+ 1 .+ offset)...]
    end
end

# matrix-formed CDFT operator for D-dimensional signal
function cdftmtx(sz::Integer...)
    len = prod(sz)

    imps = map(1:len) do idx
        u = zeros(sz)
        u[idx] = 1
        vec(fft(u))
    end
    mtx = hcat(imps...)

    rm = Diagonal([ exp(-1im*angle(mtx[n,end])/2) for n in 1:len ])

    rm * mtx / sqrt(len)
end

function permdctmtx(sz::Integer...)
    if prod(sz) == 1
        return ones(1,1)
    end
    len = prod(sz)

    imps = map(1:len) do idx
        u = zeros(sz)
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
