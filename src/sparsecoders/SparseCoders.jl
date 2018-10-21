module SparseCoders
    abstract type AbstractSparseCoder end
    abstract type AbstractISTA <: AbstractSparseCoder end

    include("utils.jl")
    export softshrink, hardshrink
    include("iht.jl")
    export IHT
    export iht

    include("ista.jl")
    export ISTA, FISTA

    include("pds.jl")
    export PDS
end
