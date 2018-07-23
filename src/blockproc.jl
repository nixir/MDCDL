function blockproc(A::Array{T,D}, blockSize::NTuple{D, Integer}, fun::Function) where {T,D}
    nBlocks = div.(size(A),blockSize)
    primeBlock = colon.(1,blockSize)

    out = simular(A)
    foreach(CartesianRange(nBlocks)) do cr
        block = (cr.I .- 1) .* blockSize .+ primeBlock
        out[block...] = fun(A[block...])
    end
    out
end

function blockproc!(A::Array{T,D}, blockSize::NTuple{D, Integer}, fun::Function) where {T,D}
    nBlocks = div.(size(A),blockSize)
    primeBlock = colon.(1,blockSize)

    foreach(CartesianRange(nBlocks)) do cr
        block = (cr.I .- 1) .* blockSize .+ primeBlock
        fun(A[block...])
    end
    A
end

function butterfly(x::AbstractArray{T,2}, p::Integer) where T
    vcat((x[1:p,:] + x[end-(p-1):end,:])/sqrt(2), x[p+1:end-p,:] ,(x[1:p,:] - x[end-(p-1):end,:])/sqrt(2))
end
