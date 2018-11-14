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
    demils = intervals([ nparamsinit, (nparamsperstage(nsolt) .* nstages(nsolt))... ])
end

nparamsinit(nsolt::RnsoltTypeI) = sum(ngivensangles.(nsolt.nChannels))
nparamsinit(nsolt::RnsoltTypeII) = sum(ngivensangles.(nsolt.nChannels))
nparamsinit(nsolt::CnsoltTypeI) = ngivensangles(nsolt.nChannels)
nparamsinit(nsolt::CnsoltTypeII) = ngivensangles(nsolt.nChannels)

nparamsperstage(nsolt::RnsoltTypeI) = ngivensangles(nsolt.nChannels[2])
nparamsperstage(nsolt::RnsoltTypeII) = sum(ngivensangles.(nsolt.nChannels))
nparamsperstage(nsolt::CnsoltTypeI) = 2*ngivensangles(fld(nsolt.nChannels,2)) + fld(nsolt.nChannels, 4)
nparamsperstage(nsolt::CnsoltTypeII) = 2*ngivensangles(fld(nsolt.nChannels,2)) + 2*ngivensangles(cld(nsotl.nChannels,2)) + 2*fld(nsolt.nChannels,4)
