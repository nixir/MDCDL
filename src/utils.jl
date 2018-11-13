using LinearAlgebra
using Random
# import Random.rand
# import Random.rand!

function rand_orthomtx!(mtx::AbstractMatrix{T}) where T
    mtx .= qr(rand(T, size(mtx)...)).Q
end

rand(nsolt::AbstractNsolt, args...; kwargs...) = rand!(similar(nsolt), args...; kwargs...)

function rand!(nsolt::CnsoltTypeI{T,D}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {T,D}
    if isSymmetry
        nsolt.Φ .= Diagonal(cis.(rand(nsolt.nChannels)))
    end

    if isInitMat
        rand_orthomtx!(nsolt.V0)
    end

    for d = 1:D
        if isPropMat
            foreach(nsolt.Wdks[d], nsolt.Udks[d]) do W, U
                rand_orthomtx!(W)
                rand_orthomtx!(U)
            end
        end
        if isPropAng
            foreach(nsolt.θdks[d]) do θ
                θ .= 2pi*(rand(size(θ)...) .- 0.5)
            end
        end
    end
end

function rand!(nsolt::CnsoltTypeII{T,D}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {T,D}
    if isSymmetry
        nsolt.Φ .= Diagonal(cis.(rand(nsolt.nChannels)))
    end

    if isInitMat
        rand_orthomtx!(nsolt.V0)
    end

    for d = 1:D
        if isPropMat
            foreach(nsolt.Wdks[d], nsolt.Udks[d], nsolt.Ŵdks[d], nsolt.Ûdks[d]) do W, U, Ŵ, Û
                rand_orthomtx!(W)
                rand_orthomtx!(U)
                rand_orthomtx!(Ŵ)
                rand_orthomtx!(Û)
            end
        end
        if isPropAng
            foreach(nsolt.θ1dks[d], nsolt.θ2dks[d]) do θ1, θ2
                θ1 .= 2pi*(rand(size(θ1)...) .- 0.5)
                θ2 .= 2pi*(rand(size(θ2)...) .- 0.5)
            end
        end
    end
end

function rand!(nsolt::RnsoltTypeI{T,D}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {T,D}
    if isInitMat
        rand_orthomtx!(nsolt.W0)
        rand_orthomtx!(nsolt.U0)
    end

    for d = 1:D
        if isPropMat
            foreach(nsolt.Udks[d]) do U
                rand_orthomtx!(U)
            end
        end
    end
end

function rand!(nsolt::RnsoltTypeII{T,D}; isInitMat=true, isPropMat=true, isPropAng=true, isSymmetry=true) where {T,D}

    if isInitMat
        rand_orthomtx!(nsolt.W0)
        rand_orthomtx!(nsolt.U0)
    end

    for d = 1:D
        if isPropMat
            foreach(nsolt.Wdks[d], nsolt.Udks[d]) do W, U
                rand_orthomtx!(W)
                rand_orthomtx!(U)
            end
        end
    end
end

intervals(lns::Tuple, args...) = (intervals(collect(lns), args...)...,)

vm1constraint!(ns::AbstractNsolt) = vm1constraint!(Val(istype1(ns)), ns)

function vm1constraint!(ns::RnsoltTypeI)
    θ, μ = mat2rotations(ns.initMatrices[1])
    θ[1:ns.nChannels[1]-1] .= 0
    ns.initMatrices[1] .= rotations2mat(θ, μ)

    ns
end

function vm1constraint!(ns::CnsoltTypeI)
    ws = foldl(ns.propMatrices,init=I) do wtmp, mts
        prod(reverse(mts[1:2:end])) * wtmp
    end
    θ, μ = mat2rotations(ns.initMatrices[1])
    θ[1:ns.nChannels-1] .= 0
    ns.initMatrices[1] .= cat(ws', Matrix(I, size(ws)...), dims=[1,2]) * rotations2mat(θ, μ)

    ns
end
