function blockproc(fun::Function, A::AbstractArray{T,D}, blockSize::NTuple{D, Integer}) where {T,D}
     blockproc!(fun, similar(A), A, blockSize)
end

function blockproc!(fun::Function, dst::AbstractArray{T,D}, src::AbstractArray{T,D}, blockSize::NTuple{D, Integer}, ) where {T,D}
    nBlocks = div.(size(src), blockSize)
    primeBlock = colon.(1,blockSize)

    for cr = CartesianRange(nBlocks)
        block = (cr.I .- 1) .* blockSize .+ primeBlock
        dst[block...] = fun(src[block...])
    end
    dst
end
