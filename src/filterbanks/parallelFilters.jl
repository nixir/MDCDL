
struct ParallelFilters{T,D} <: FilterBank{T,D}
    decimationFactor::NTuple{D,Int}
    polyphaseOrder::NTuple{D,Int}
    nChannels::Integer

    kernelspair::NTuple{2}

    function ParallelFilters(ker::NTuple{2,Vector{A}}, df::NTuple{D,Int}, ord::NTuple{D,Int}, nch::Integer) where {T,D,A<:AbstractArray{T,D}}
        new{T,D}(df, ord, nch, ker)
    end

    function ParallelFilters(ker::Vector{A}, df::Tuple, ord::Tuple, nch::Integer) where {T,D,A<:AbstractArray{T,D}}
        ParallelFilters((ker,ker), df, ord, nch)
    end

    function ParallelFilters(::Type{T}, df::NTuple{D,Int}, ord::NTuple{D,Int}, nch::Integer) where {T,D}
        new{T,D}(df, ord, nch, ([fill(zeros(df .* (ord .+ 1)), nch) for idx=1:2 ]...,))
    end

    function ParallelFilters(fb::FilterBank)
        ParallelFilters(kernels(fb), decimations(fb), orders(fb), nchannels(fb))
    end
end

analysiskernels(pf::ParallelFilters) = kernelspair[1]
synthesiskernels(pf::ParallelFilters) = kernelspair[2]
kernels(pf::ParallelFilters) = kernelspair
