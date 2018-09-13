using LinearAlgebra
using Random
import Random.rand
import Random.rand!

function rand!(cnsolt::Cnsolt{T,D}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {D,T}
    P = sum(cnsolt.nChannels)

    if isSymmetry
        cnsolt.symmetry .= Diagonal(exp.(1im*rand(P)))
    end
    if isInitMat
        cnsolt.initMatrices[1] = T.(qr(rand(P,P)).Q)
    end

    for d = 1:D
        if isPropMat
            map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
                T.(qr(rand(T,size(A))).Q)
            end
        end

        if isPropAng
            @. cnsolt.paramAngles[d] = rand(T,size(cnsolt.paramAngles[d]))
        end
    end
    cnsolt
end

function rand!(rnsolt::Rnsolt{T,D}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {D,T} # "isPropAng" and "isSymmetry" are not used.
    P = sum(rnsolt.nChannels)
    hP = fld(P,2)

    if isInitMat
        rnsolt.initMatrices[1] = T.(qr(rand(size(rnsolt.initMatrices[1])...)).Q)
        rnsolt.initMatrices[2] = T.(qr(rand(size(rnsolt.initMatrices[2])...)).Q)
    end

    for d = 1:D
        if isPropMat
            map!(rnsolt.propMatrices[d], rnsolt.propMatrices[d]) do A
                T.(qr(rand(T,size(A))).Q)
            end
        end
    end
    rnsolt
end

function rand(cb::CodeBook; kwargs...)
    rand!(deepcopy(cb); kwargs...)
end
