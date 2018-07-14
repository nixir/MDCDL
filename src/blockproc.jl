function blockproc(A::Array{T,D}, blockSize::NTuple{D, Integer}, fun::Function) where {T,D}
    nBlocks = div.(size(A),blockSize)
    primeBlock = ntuple(n -> 1:blockSize[n], D)

    out = Array{T,D}(size(A))
    foreach(CartesianRange(nBlocks)) do cr
        block = (cr.I .- 1) .* blockSize .+ primeBlock
        out[block...] = fun(A[block...])
    end
    out
end

function blockproc!(A::Array{T,D}, blockSize::NTuple{D, Integer}, fun::Function) where {T,D}
    nBlocks = div.(size(A),blockSize)
    primeBlock = ntuple(n -> 1:blockSize[n], D)

    foreach(CartesianRange(nBlocks)) do cr
        block = (cr.I .- 1) .* blockSize .+ primeBlock
        fun(A[block...])
    end
    A
end

function butterfly(x::Matrix, p::Integer)
    vcat((x[1:p,:] + x[end-(p-1):end,:])/sqrt(2), x[p+1:end-p,:] ,(x[1:p,:] - x[end-(p-1):end,:])/sqrt(2))
end
