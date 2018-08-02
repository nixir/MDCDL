function blockproc(fun::Function, A::AbstractArray{T,D}, blockSize::NTuple{D, Integer}) where {T,D}
     blockproc!(fun, similar(A), A, blockSize)
end

function blockproc!(fun::Function, dst::AbstractArray{T,D}, src::AbstractArray{T,D}, blockSize::NTuple{D, Integer}, ) where {T,D}
    nBlocks = div.(size(src), blockSize)
    primeBlock = colon.(1,blockSize)

    for idx = 1:prod(nBlocks)
        block = (CartesianIndices(nBlocks)[idx].I .- 1) .* blockSize .+ primeBlock
        dst[block...] = fun(src[block...])
    end
    dst
end
