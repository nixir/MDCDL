function updateDictionary(optr_t::Type{AG}, opt_params, cb::DT, x::AbstractArray, hy::AbstractArray, params; shape::Shapes.AbstractShape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {DT<:LearningTarget,TT<:AbstractArray,TM<:AbstractArray,AG<:Optimizers.AbstractGradientDescent}
    vecpm, pminfo = decompose_params(DT, params)
    f(t) = lossfcn(cb, x, hy, compose_params(DT, t, pminfo); shape=shape)
    ∇f(t) = ForwardDiff.gradient(f, t)
    optr = optr_t(∇f; opt_params...)

    vecpm_opt = optr(vecpm)

    params_opt = compose_params(DT, vecpm_opt, pminfo)
    loss_opt = f(vecpm_opt)
    return (params_opt, loss_opt,)
end

function updateDictionary(nlopt_t::Type{Optimizers.CRS}, opt_params, cb::DT, x::AbstractArray, hy::AbstractArray, params; shape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {DT<:LearningTarget}
    vecpm, pminfo = decompose_params(DT, params)
    f(t, grad=nothing) = lossfcn(cb, x, hy, compose_params(DT, t, pminfo); shape=shape)


    nlopt = nlopt_t(f; opt_params...)

    vecpm_opt = nlopt(vecpm)
    minf = f(vecpm_opt)
    params_opt = compose_params(DT, vecpm_opt, pminfo)
    # vlevel >= 0 && println("fopt(θ) = $minf, f(θ) = $(f(minx))")
    return (params_opt, minf)
end

function updateDictionary(nlopt_t::Type{Optimizers.GlobalOpt}, opt_params, cb::DT, x::AbstractArray, hy::AbstractArray, params; shape=Shapes.Vec(size(x)), vlevel::Integer=0, kwargs...) where {DT<:LearningTarget}
    vecpm, pminfo = decompose_params(DT, params)
    f(t, grad=nothing) = lossfcn(cb, x, hy, compose_params(DT, t, pminfo); shape=shape)

    nlopt = nlopt_t(f; opt_params...)

    vecpm_opt = nlopt(vecpm)
    minf = f(vecpm_opt)
    params_opt = compose_params(DT, vecpm_opt, pminfo)
    # vlevel >= 0 && println("fopt(θ) = $minf, f(θ) = $(f(minx))")
    return (params_opt, minf)
end

function lossfcn(cb::LearningTarget, x::AbstractArray, y::AbstractArray, params; shape=Shapes.Vec(size(x))) where {TT<:AbstractArray,TM<:AbstractArray}
    cpcb = similar_dictionary(cb, params)
    syn = createTransform(cpcb, shape)
    norm(x - synthesize(syn, y))^2/2
end

function similar_dictionary(nsolt::AbstractNsolt, (θ, μ)::Tuple{TT,TM}) where {TT<:AbstractArray,TM<:AbstractArray}
    setrotations!(similar(nsolt, eltype(θ)), θ, μ)
end

similar_dictionary(target::Multiscale, params::NTuple{N}) where {N} = Multiscale(similar_dictionary.(target.filterbanks, params)...)

getParamsDictionary(nsolt::AbstractNsolt) = getrotations(nsolt)
setParamsDictionary!(nsolt::AbstractNsolt, pm::NTuple{2}) = setrotations!(nsolt, pm...)

getParamsDictionary(targets::Multiscale) = map(t->getParamsDictionary(t), targets.filterbanks)
setParamsDictionary!(targets::Multiscale, pm::NTuple) = map((t,p)->setParamsDictionary!(t, p), targets.filterbanks, pm)

getOperatorType_scs(::Type, ::Val) = AbstractOperator
getOperatorType_scs(::Type{NS}, ::Val{:convolution}) where {NS<:AbstractNsolt} = ConvolutionalOperator

decompose_params(::Type, params) = (params, ())
compose_params(::Type, params, ()) = params

decompose_params(::Type{NS}, (θ, μ)) where {NS<:AbstractNsolt} = (θ, μ)
compose_params(::Type{NS}, θ, μ) where {NS<:AbstractNsolt} = (θ, μ)

function decompose_params(::Type{Multiscale}, params::NTuple{N}) where {N} # T<:AbstractNsolt
    θs, μs = map(t->t[1], params), map(t->t[2], params)
    vcat(θs...), (length.(θs), μs)
end

function compose_params(::Type{Multiscale}, vp::AbstractArray, pminfo::Tuple) where {N}
    arrpms = map(rng->vp[rng],intervals(pminfo[1]))
    map((lhs,rhs)->(lhs,rhs,), arrpms, pminfo[2])
end
