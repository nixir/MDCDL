function randomInit!(cnsolt::MDCDL.Cnsolt{D,1,T}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {D,T}
    P = sum(cnsolt.nChannels)

    if isSymmetry
        cnsolt.symmetry .= Diagonal(exp.(1im*rand(P)))
    end
    if isInitMat
        cnsolt.initMatrices[1] = Array{T}(qr(rand(P,P), thin=false)[1])
    end

    for d = 1:D
        if isPropMat
            map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
                Array(qr(rand(T,size(A)), thin=false)[1])
            end
        end

        if isPropAng
            @. cnsolt.paramAngles[d] = rand(T,size(cnsolt.paramAngles[d]))
        end
    end
end

function randomInit!(rnsolt::MDCDL.Rnsolt{D,1,T}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {D,T} # "isPropAng" and "isSymmetry" are not used.
    P = sum(rnsolt.nChannels)
    hP = fld(P,2)

    if isInitMat
        rnsolt.initMatrices[1] = Array{T}(qr(rand(hP,hP), thin=false)[1])
        rnsolt.initMatrices[2] = Array{T}(qr(rand(hP,hP), thin=false)[1])
    end

    for d = 1:D
        if isPropMat
            map!(rnsolt.propMatrices[d], rnsolt.propMatrices[d]) do A
                Array(qr(rand(T,size(A)), thin=false)[1])
            end
        end
    end
end
