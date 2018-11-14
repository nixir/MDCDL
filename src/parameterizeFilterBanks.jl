setrotations!(cc::AbstractNsolt, (θ, μ)) = setrotations(cc, θ, μ)

setrotations(cc::AbstractNsolt, args...) = setrotations!(similar(cc), args...)

function getrotations(nsolt::AbstractNsolt{T,D}) where {T,D}
    initpms = getrotations_init(nsolt)
    proppms = map(enumerate(nstages(nsolt))) do (d, nstg)
        map(1:nstg) do k
            getrotations_prop(nsolt, d, k)
        end
    end
    pms = vcat(initpms, proppms...)

    return (vcat(map(t->t[1], pms)...), vcat(map(t->t[2], pms)...))
end

getrotations_init(nsolt::RnsoltTypeI) = [ mat2rotations(nsolt.W0), mat2rotations(nsolt.U0) ]
getrotations_init(nsolt::RnsoltTypeII) = [ mat2rotations(nsolt.W0), mat2rotations(nsolt.U0) ]
getrotations_init(nsolt::CnsoltTypeI) = [ mat2rotations(nsolt.V0) ]
getrotations_init(nsolt::CnsoltTypeII) = [ mat2rotations(nsolt.V0) ]

function getrotations_prop(nsolt::RnsoltTypeI, d, k)
    mat2rotations(nsolt.Udks[d][k])
end
function getrotations_prop(nsolt::RnsoltTypeII, d, k)
    pms = mat2rotations.([ nsolt.Wdks[d][k], nsolt.Udks[d][k] ])
    return (vcat(map(t->t[1], pms)...), vcat(map(t->t[2], pms)...))
end
function getrotations_prop(nsolt::CnsoltTypeI, d, k)
    pms = mat2rotations.([ nsolt.Wdks[d][k], nsolt.Udks[d][k] ])
    return (vcat(map(t->t[1], pms)..., nsolt.θdks[d][k]), vcat(map(t->t[2], pms)...))
end
function getrotations_prop(nsolt::CnsoltTypeII, d, k)
    pms = mat2rotations.([ nsolt.Wdks[d][k], nsolt.Udks[d][k], nsolt.Ŵdks[d][k], nsolt.Ûdks[d][k] ])
    return (vcat(map(t->t[1], pms)..., nsolt.θ1dks[d][k], nsolt.θ2dks[d][k]), vcat(map(t->t[2], pms)...))
end


function setrotations!(nsolt::AbstractNsolt{T,D}, θ::AbstractArray, μ::AbstractArray) where {T,D}
    npiθ, npiμ = nparamsinit(nsolt)
    nppsθ, nppsμ = nparamsperstage(nsolt)
    dlmθ = intervals([ sum(npiθ), (sum(nppsθ) .* nstages(nsolt))... ])
    dlmμ = intervals([ sum(npiμ), (sum(nppsμ) .* nstages(nsolt))... ])
    θinit, μinit = θ[dlmθ[1]], μ[dlmμ[1]]
    θprop, μprop = map(t->θ[t], dlmθ[2:end]), map(t->μ[t], dlmμ[2:end])

    setrotations_init!(nsolt, θinit, μinit)

    foreach(1:D, nstages(nsolt), θprop, μprop) do d, nstg, θd, μd
        dlmθk = intervals(fill(sum(nppsθ), nstg))
        dlmμk = intervals(fill(sum(nppsμ), nstg))
        θdk = map(k->θd[k], dlmθk)
        μdk = map(k->μd[k], dlmμk)
        foreach(1:nstg, θdk, μdk) do k, θ, μ
            setrotations_prop!(nsolt, d, k, θ, μ)
        end
    end

    return nsolt
end


function nparamsinit(nsolt::RnsoltTypeI)
    ([ ngivensangles.(nsolt.nChannels)... ], [ nsolt.nChannels... ])
end
function nparamsinit(nsolt::RnsoltTypeII)
    ([ ngivensangles.(nsolt.nChannels)... ], [ nsolt.nChannels... ])
end
function nparamsinit(nsolt::CnsoltTypeI)
    ([ ngivensangles(nsolt.nChannels) ], [ nsolt.nChannels ])
end
function nparamsinit(nsolt::CnsoltTypeII)
    ([ ngivensangles(nsolt.nChannels) ], [ nsolt.nChannels ])
end

function nparamsperstage(nsolt::RnsoltTypeI)
    ([ ngivensangles(nsolt.nChannels[2]) ], [ nsolt.nChannels[2] ])
end
function nparamsperstage(nsolt::RnsoltTypeII)
    ([ ngivensangles.(nsolt.nChannels)... ], [ nsolt.nChannels... ])
end
function nparamsperstage(nsolt::CnsoltTypeI)
    hP = fld(nsolt.nChannels, 2)
    (
        [ fill(ngivensangles(hP), 2)..., fld(nsolt.nChannels, 4) ],
        [ fill(hP, 2)... ],)
end
function nparamsperstage(nsolt::CnsoltTypeII)
    fP, cP = fld(nsolt.nChannels,2), cld(nsolt.nChannels, 2)
    (
        [ fill(ngivensangles(fP),2)..., fill(ngivensangles(cP),2)..., fill(fld(nsolt.nChannels,4),2)... ],
        [ fP, fP, cP, cP] ,)
end

function setrotations_init!(nsolt::RnsoltTypeI, θ::AbstractArray, μ::AbstractArray)
    dlsθ, dlsμ = intervals.(nparamsinit(nsolt))
    nsolt.W0 .= rotations2mat(θ[dlsθ[1]], μ[dlsμ[1]])
    nsolt.U0 .= rotations2mat(θ[dlsθ[2]], μ[dlsμ[2]])
    return nsolt
end

function setrotations_init!(nsolt::RnsoltTypeII, θ::AbstractArray, μ::AbstractArray)
    dlsθ, dlsμ = intervals.(nparamsinit(nsolt))
    nsolt.W0 .= rotations2mat(θ[dlsθ[1]], μ[dlsμ[1]])
    nsolt.U0 .= rotations2mat(θ[dlsθ[2]], μ[dlsμ[2]])
    return nsolt
end

function setrotations_init!(nsolt::CnsoltTypeI, θ::AbstractArray, μ::AbstractArray)
    nsolt.V0 .= rotations2mat(θ, μ)
    return nsolt
end

function setrotations_init!(nsolt::CnsoltTypeII, θ::AbstractArray, μ::AbstractArray)
    nsolt.V0 .= rotations2mat(θ, μ)
    return nsolt
end

function setrotations_prop!(nsolt::RnsoltTypeI, d::Integer, k::Integer, θ::AbstractArray, μ::AbstractArray)
    nsolt.Udks[d][k] .= rotations2mat(θ, μ)
    return nsolt
end

function setrotations_prop!(nsolt::RnsoltTypeII, d::Integer, k::Integer, θ::AbstractArray, μ::AbstractArray)
    dlsθ, dlsμ = intervals.(nparamsperstage(nsolt))
    nsolt.Wdks[d][k] .= rotations2mat(θ[dlsθ[1]], μ[dlsμ[1]])
    nsolt.Udks[d][k] .= rotations2mat(θ[dlsθ[2]], μ[dlsμ[2]])
    return nsolt
end

function setrotations_prop!(nsolt::CnsoltTypeI, d::Integer, k::Integer, θ::AbstractArray, μ::AbstractArray)
    dlsθ, dlsμ = intervals.(nparamsperstage(nsolt))
    nsolt.Wdks[d][k] .= rotations2mat(θ[dlsθ[1]], μ[dlsμ[1]])
    nsolt.Udks[d][k] .= rotations2mat(θ[dlsθ[2]], μ[dlsμ[2]])
    nsolt.θdks[d][k] .= θ[dlsθ[3]]
    return nsolt
end

function setrotations_prop!(nsolt::CnsoltTypeII, d::Integer, k::Integer, θ::AbstractArray, μ::AbstractArray)
    dlsθ, dlsμ = intervals.(nparamsperstage(nsolt))
    nsolt.Wdks[d][k] .= rotations2mat(θ[dlsθ[1]], μ[dlsμ[1]])
    nsolt.Udks[d][k] .= rotations2mat(θ[dlsθ[2]], μ[dlsμ[2]])
    nsolt.Ŵdks[d][k] .= rotations2mat(θ[dlsθ[3]], μ[dlsμ[3]])
    nsolt.Ûdks[d][k] .= rotations2mat(θ[dlsθ[4]], μ[dlsμ[4]])
    nsolt.θ1dks[d][k] .= θ[dlsθ[5]]
    nsolt.θ2dks[d][k] .= θ[dlsθ[6]]
    return nsolt
end
