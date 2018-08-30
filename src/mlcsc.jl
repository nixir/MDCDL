function analyze(mlcsc::MDCDL.MultiLayerCsc{TC,D}, x::Array{TX,D}) where {TC,TX,D}
    foldl(x, mlcsc.dictionaries) do tx, dic
        analyze(dic, tx; shape=:augumented)
    end
end

function adjoint_synthesize(mlcsc::MDCDL.MultiLayerCsc{TC,D}, x::Array{TX,D}) where {TC,TX,D}
    foldl(x, mlcsc.dictionaries) do tx, dic
        adjoint_synthesize(dic, tx; shape=:augumented)
    end
end

function synthesize(mlcsc::MDCDL.MultiLayerCsc, y::Array)
    foldr(y, mlcsc.dictionaries) do dic, ty
        synthesize(dic, ty)
    end
end

function mlista(mlcsc::MDCDL.MultiLayerCsc, x, λs::Vector{T}; maxIterations::Integer=20, absTol::Real=1e-10, viewStatus::Bool=false) where T <: Real
    L = mlcsc.nLayers
    opD  = (l, v) -> synthesize(mlcsc.dictionaries[l], v)
    opDt = (l, v) -> adjoint_synthesize(mlcsc.dictionaries[l], v; shape=:augumented)

    γ = [ x, accumulate((tx, l)->opDt(l,tx), x, 1:L)... ]

    for k = 1:maxIterations
        # hγ = [ [ opD(l, γ[l+1]) for l in 1:L ]..., γ[L+1] ]
        hγ = [ reverse(accumulate((tγ,l)->opD(l,tγ), γ[L+1], L:-1:1))..., γ[L+1] ]

        γ[1] = x
        for l = 1:L
            γ[l+1] = softshrink(hγ[l+1] - opDt(l, opD(l, hγ[l+1]) - γ[l]), λs[l])
        end
        if viewStatus
            println("Iteration $k finished. errx = $(vecnorm(x-synthesize(mlcsc,γ[L+1])))")
        end
    end
    γ[end]
end

function mlfista(mlcsc::MDCDL.MultiLayerCsc, x, λs::Vector{T}; maxIterations::Integer=20, absTol::Real=1e-10, viewStatus::Bool=false) where T <: Real
    L = mlcsc.nLayers
    opD  = (l, v) -> synthesize(mlcsc.dictionaries[l], v)
    opDt = (l, v) -> adjoint_synthesize(mlcsc.dictionaries[l], v; shape=:augumented)

    γ = [ x, accumulate((tx, l)->opDt(l,tx), x, 1:L)... ]

    tk = 1.0
    z = γ[L+1]
    for k = 1:maxIterations
        hγ = [ reverse(accumulate((tγ,l)->opD(l,tγ), z, L:-1:1))..., z ]

        γ[1] = x
        for l = 1:L
            γ[l+1] = softshrink(hγ[l+1] - opDt(l, opD(l, hγ[l+1]) - γ[l]), λs[l])
        end

        tkprev = tk
        tk = (1 + sqrt(1+4*tkprev^2)) / 2
        z = γ[L+1] + (tkprev - 1)/tk * (γ[L+1] - z)

        if viewStatus
            println("Iteration $k finished. errx = $(vecnorm(x-synthesize(mlcsc,γ[L+1])))")
        end
    end
    γ[end]
end
