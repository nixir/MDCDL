using LinearAlgebra
using Random
# import Random.rand
# import Random.rand!

rand(nsolt::AbstractNsolt, args...; kwargs...) = rand!(similar(nsolt), args...; kwargs...)

function rand!(cnsolt::Cnsolt{T,D}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {D,T}
    P = sum(cnsolt.nChannels)

    if isSymmetry
        cnsolt.symmetry .= Diagonal(exp.(1im*rand(P)))
    end
    if isInitMat
        # cnsolt.initMatrices[1] = T.(qr(rand(P,P)).Q)
        for mtx in cnsolt.initMatrices
            mtx .= qr(rand(T,size(mtx)...)).Q
        end
    end

    foreach(cnsolt.propMatrices, cnsolt.paramAngles) do pms, pas
        if isPropMat
            for mtx in pms
                mtx .= qr(rand(T,size(mtx)...)).Q
            end
        end

        if isPropAng
            for angs in pas
                angs .= rand(T,size(angs)...)
            end
        end
    end
    cnsolt
end

function rand!(rnsolt::Rnsolt{T,D}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {D,T} # "isPropAng" and "isSymmetry" are not used.
    if isInitMat
        for mtx in rnsolt.initMatrices
            mtx .= qr(rand(T,size(mtx)...)).Q
        end
    end

    for pms in rnsolt.propMatrices
        if isPropMat
            for mtx in pms
                mtx .= qr(rand(T,size(mtx)...)).Q
            end
        end
    end
    rnsolt
end

function intervals(lns::AbstractVector, offset::Integer=0)
    map(lns, cumsum(lns)) do st, ed
        UnitRange(ed - st + 1 + offset, ed + offset)
    end
end

intervals(lns::Tuple, args...) = (intervals(collect(lns), args...)...,)

vm1constraint!(ns::AbstractNsolt) = vm1constraint!(Val(istype1(ns)), ns)

function vm1constraint!(::TypeI, ns::Rnsolt)
    θ, μ = mat2rotations(ns.initMatrices[1])
    θ[1:ns.nChannels[1]-1] .= 0
    ns.initMatrices[1] .= rotations2mat(θ, μ)

    ns
end

function vm1constraint!(::TypeI, ns::Cnsolt)
    ws = foldl(ns.propMatrices,init=I) do wtmp, mts
        prod(reverse(mts[1:2:end])) * wtmp
    end
    θ, μ = mat2rotations(ns.initMatrices[1])
    θ[1:ns.nChannels-1] .= 0
    ns.initMatrices[1] .= cat(ws', Matrix(I, size(ws)...), dims=[1,2]) * rotations2mat(θ, μ)

    ns
end
