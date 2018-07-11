function analyze(mlcsc::MDCDL.MultiLayerCsc{TC,D}, x::Array{TX,D}) where {TC,TX,D}
    foldl(x, mlcsc.dictionaries) do tx, dic
        analyze(dic, tx; outputMode=:augumented)
    end
end

function adjoint_synthesize(mlcsc::MDCDL.MultiLayerCsc{TC,D}, x::Array{TX,D}; isAllCoefs::Bool=false) where {TC,TX,D}
    foldl(x, mlcsc.dictionaries) do tx, dic
        adjoint_synthesize(dic, tx; outputMode=:augumented)
    end
end

function synthesize(mlcsc::MDCDL.MultiLayerCsc, y::Array; isAllCoefs::Bool=false)
    foldr(y, mlcsc.dictionaries) do dic, ty
        synthesize(dic, ty)
    end
end

function mlista(mlcsc::MDCDL.MultiLayerCsc, x, λs::Vector{T}; maxIterations::Integer=20, absTol::Real=1e-10, viewStatus::Bool=false) where T <: Real
    const L = mlcsc.nLayers
    opD  = (l, v) -> synthesize(mlcsc.dictionaries[l], v)
    opDt = (l, v) -> adjoint_synthesize(mlcsc.dictionariess[l], v; outputMode=:augumented)

    γ = Vector(L+1)
    γ[1] = x
    for l = 2:L+1
        γ[l] = opDt(l-1, γ[l-1])
    end


    for k = 1:maxIterations
        hγ = Vector(L+1)
        hγ[L+1] = γ[L+1]
        for l = L:-1:2
            hγ[l] = opD(l, γ[l+1])
        end
        hγ[1] = x

        for l = 1:L
            γ[l+1] = softshrink(hγ[l+1] - opDt(l, opD(l,hγ[l+1]) - γ[l]), λs[l])
        end
    end
    γ[2:end]
end

function mlfista(mlcsc::MDCDL.MultiLayerCsc, x, λs::Vector{T}; maxIterations::Integer=20, absTol::Real=1e-10, viewStatus::Bool=false) where T <: Real
    const L = mlcsc.nLayers
    opD  = (l, v) -> synthesize(mlcsc.dictionaries[l], v)
    opDt = (l, v) -> adjoint_synthesize(mlcsc.dictionaries[l], v; outputMode=:augumented)

    γ = Vector(L+1)
    γ[1] = x
    for l = 2:L+1
        γ[l] = opDt(l-1, γ[l-1])
    end

    tk = 1.0
    z = γ[L+1]
    for k = 1:maxIterations
        hγ = Vector(L+1)
        hγ[L+1] = z
        for l = L:-1:2
            hγ[l] = opD(l, γ[l+1])
        end
        hγ[1] = x

        glp = γ[L+1]
        for l = 1:L
            γ[l+1] = softshrink(hγ[l+1] - opDt(l, opD(l,hγ[l+1]) - γ[l]), λs[l])
        end
        tkprev = tk
        tk = (1 + sqrt(1+4*tkprev)) / 2

        z = γ[L+1] + (tkprev - 1)/tk * (γ[L+1] - glp)
        if viewStatus
            println("Iteration $k finished.")
        end
    end
    γ[2:end]
end
