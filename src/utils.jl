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

function intervals(lns::AbstractVector{T}) where {T<:Integer}
    map(lns, cumsum(lns)) do st, ed
        UnitRange(ed - st + 1, ed)
    end
end

intervals(lns::Tuple) = (intervals(collect(lns))...,)
