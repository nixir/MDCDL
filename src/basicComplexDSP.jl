# circular convolution

function cconv(x::Array{Complex{T},D}, h::Array{Complex{T},D}) where {T,D}
    ifft( fft(x) .* fft(h))
end

function cconv(x::Array{T,D}, h::Array{T,D}) where {T<:Real,D}
    real(ifft( fft(x) .* fft(h)))
end

cconv(x::Array{TX,D}, h::Array{TY,D}) where {TX,TY,D} = cconv(promote(x,h)...)

# upsampler
function upsample(x::Array{T,D}, factor::NTuple{D}, offset::NTuple{D} = tuple(zeros(Integer,D)...)) where {T,D}
    szx = size(x)
    output = zeros(T, szx .* factor)
    # for idx = 1:prod(szx)
    foreach(1:prod(szx)) do idx
        sub = ind2sub(szx,idx)
        output[((sub .- 1) .* factor .+ 1 .+ offset)...] = x[sub...]
    end
    output
end

function downsample(x::Array{T,D}, factor::NTuple{D}, offset::NTuple{D} = tuple(zeros(Integer,D)...)) where {T,D}
    szout = fld.(size(x), factor)
    output = zeros(T, szout)
    # for idx = 1:prod(szout)
    foreach(1:prod(szout)) do idx
        sub = ind2sub(szout,idx)
        output[sub...] = x[((sub .-1) .* factor .+ 1 .+ offset)...]
    end
    output
end

# multidimensional FIR filtering
function mdfilter(A::Array{T,D}, h::Array{T,D}; boundary=:circular, outputSize=:same) where {T,D}
    # center = cld.(size(h), 2)
    
    ker = zeros(T,size(A)...)
    ker[colon.(1,size(h))...] = h

    cconv(A,ker) # boundary="circular", outputsize="same", convOrCorr="conv"
end

mdfilter(A::Array{TA,D}, h::Array{TX,D}; kwargs...) where {TA,TX,D} = mdfilter(promote(A,h)...; kwargs...)

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
    len = prod(sz)

    imps = map(1:len) do idx
        u = zeros(sz)
        u[idx] = 1
        vec(dct(u))
    end
    mtx = hcat(imps...)

    evenIds = find(x -> sum(ind2sub(sz,x) .- 1) % 2 == 0, 1:len)
    oddIds = find(x -> sum(ind2sub(sz,x) .- 1) % 2 != 0, 1:len)

    evenMtx = hcat([ mtx[idx,:] for idx in evenIds]...)
    oddMtx = hcat([ mtx[idx,:] for idx in oddIds]...)

    vcat(evenMtx.', oddMtx.')
end
