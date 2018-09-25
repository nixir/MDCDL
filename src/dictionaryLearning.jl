using ForwardDiff
using Base.Filesystem
using Statistics
using Dates

module SparseCoders
    using MDCDL.Shapes
    abstract type SparseCoder{S} end
    struct IHT{S} <: SparseCoder{S}
        iterations::Integer
        sparsity::AbstractFloat

        filter_domain::Symbol

        IHT(; iterations=1, sparsity=0.5, filter_domain=:convolution, shape::S=Shapes.Vec()) where {S<:Shapes.AbstractShape} = new{S}(iterations, sparsity, filter_domain)
    end

    struct ScalewiseIHT{N,S} <: SparseCoder{S}
        iterations::Integer
        nonzeros::NTuple{N,Integer}

        ScalewiseIHT(; iterations::Integer=1, nonzeros::NTuple{N}=(fill(1,N)...,), shape::S=Shapes.Augumented()) where {N,S<:Shapes.AbstractShape} = new{N,S}(iterations, nonzeros)
    end
end

module Optimizers
    using MDCDL.Shapes
    abstract type AbstractOptimizer{S} end
    abstract type AbstractGradientDescent{S} <: AbstractOptimizer{S} end
    struct Steepest{S} <: AbstractGradientDescent{S}
        iterations::Integer
        rate::AbstractFloat
        Steepest(; iterations=1, rate=1e-3, shape::S=Shapes.Vec()) where {S<:Shapes.AbstractShape}= new{S}(iterations, rate)
    end

    struct Momentum{S} <: AbstractGradientDescent{S}
        iterations::Integer
        rate::AbstractFloat
        β::AbstractFloat
        Momentum(; iterations=1, rate=1e-3, beta=1e-8, shape::S=Shapes.Vec()) where {S<:Shapes.AbstractShape}= new{S}(iterations, rate, beta)
    end

    struct AdaGrad{S} <: AbstractGradientDescent{S}
        iterations::Integer
        rate::AbstractFloat
        ϵ::AbstractFloat
        AdaGrad(; iterations=1, rate=1e-3, epsilon=1e-8, shape::S=Shapes.Vec()) where {S<:Shapes.AbstractShape} = new{S}(iterations, rate, epsilon)
    end

    struct Adam{S} <: AbstractGradientDescent{S}
        iterations::Integer
        rate::AbstractFloat
        β1::AbstractFloat
        β2::AbstractFloat
        ϵ::AbstractFloat

        Adam(; iterations=1, rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, shape=Shapes.Vec()) = new{shape}(iterations, rate, beta1, beta2, epsilon)
    end

end

iterations(abopt::Optimizers.AbstractOptimizer) = abopt.iterations

LearningTarget{N} = Union{CodeBook, NTuple{N, CodeBook}}

function train!(target::LearningTarget, trainingSet::AbstractArray; epochs::Integer=1, sparsecoder::SparseCoders.SparseCoder{S}=SparseCoders.IHT(), optimizer::Optimizers.AbstractOptimizer{S}=Optimizers.Steepest(), verbose::Union{Integer,Symbol}=1, logdir=Union{Nothing,AbstractString}=nothing) where {S<:Shapes.AbstractShape}
    vlevel = verboselevel(verbose)

    savesettings(logdir, target, trainingSet;
        vlevel=vlevel,
        epochs=epochs)

    vlevel >= 1 && println("beginning dictionary training...")

    params_dic = getParamsDictionary(target)
    for itr = 1:epochs
        K = length(trainingSet)
        loss_sps = fill(Inf, K)
        loss_dus = fill(Inf, K)
        for k = 1:K
            x = trainingSet[k]

            vlevel >= 3 && println("start Sparse Coding Stage.")
            sparse_coefs, loss_sps[k] = stepSparseCoding(sparsecoder, target, x; vlevel=vlevel)
            vlevel >= 3 && println("end Sparse Coding Stage.")

            vlevel >= 3 && println("start Dictionary Update.")
            params_dic, loss_dus[k] = updateDictionary(optimizer, target, x, sparse_coefs, params_dic; vlevel=vlevel)
            vlevel >= 3 && println("end Dictionary Update Stage.")

            vlevel >= 2 && println("epoch #$itr, data #$k: loss(Sparse coding) = $(loss_sps[k]), loss(Dic. update) = $(loss_dus[k]).")

            setParamsDictionary!(target, params_dic)
        end
        if vlevel >= 1
            println("--- epoch #$itr, sum(loss) = $(sum(loss_dus)), var(loss) = $(var(loss_dus))")
        end
        savelogs(logdir, target, itr;
            vlevel=vlevel,
            time=string(now()),
            loss_sparse_coding=loss_sps,
            loss_dictionary_update=loss_dus)
    end
    vlevel >= 1 && println("training finished.")
    return setParamsDictionary!(target, params_dic)
end

function stepSparseCoding(ihtsc::SparseCoders.IHT{S}, cb::DT, x::AbstractArray; vlevel::Integer=0, kwargs...) where {S<:Shapes.AbstractShape,DT<:LearningTarget}
    TP = getOperatorType_scs(DT, Val(ihtsc.filter_domain))

    ana = createAnalyzer(TP, cb, size(x); shape=S())
    syn = createSynthesizer(TP, cb, size(x); shape=S())

    # initial sparse vector y0
    y0 = analyze(ana, x)
    # number of non-zero coefficients
    K = trunc(Int, ihtsc.sparsity * length(x))

    y_opt, loss_iht = iht(syn, ana, x, y0, K; iterations=ihtsc.iterations, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end

function stepSparseCoding(ihtsc::SparseCoders.ScalewiseIHT{N,S}, targets::NTuple{N,CB}, x::AbstractArray; vlevel::Integer=0, kwargs...) where {N,S<:Shapes.AbstractShape,CB<:CodeBook}
    ana = createAnalyzer(targets, size(x); shape=S())
    syn = createSynthesizer(targets, size(x); shape=S())

    y0 = analyze(ana, x)

    y_opt, loss_iht = iht(syn, ana, x, y0, ihtsc.nonzeros; iterations=ihtsc.iterations, isverbose=(vlevel>=3))

    return (y_opt, loss_iht)
end

updateDictionary(::Optimizers.AbstractOptimizer, cb::LearningTarget, x::AbstractArray, hy::AbstractArray; kwargs...) = updateDictionary(cb, x, hy, getParamsDictionary(cb), kwargs...)

function updateDictionary(optr::Optimizers.AbstractGradientDescent{S}, cb::DT, x::AbstractArray, hy::AbstractArray, params; vlevel::Integer=0, kwargs...) where {S<:Shapes.AbstractShape,DT<:LearningTarget,TT<:AbstractArray,TM<:AbstractArray}
    vecpm, pminfo = decompose_params(DT, params)
    f(t) = lossfcn(cb, x, hy, compose_params(DT, t, pminfo); shape=S())
    ∇f(t) = ForwardDiff.gradient(f, t)

    vecpm_opt = vecpm
    optim_state = initialstate(optr, ∇f, vecpm)
    for itr = 1:iterations(optr)
        Δvecpm = ∇f(vecpm_opt)
        upm, optim_state = updateamount(optr, Δvecpm, itr, optim_state)
        vecpm_opt -= upm

        vlevel >= 3 && println("Dic. Up. Stage: #Iter. = $itr, ||∇loss|| = $(norm(Δvecpm))")
    end
    params_opt = compose_params(DT, vecpm_opt, pminfo)
    loss_opt = f(vecpm_opt)
    return (params_opt, loss_opt,)
end

savesettings(::Nothing, args...; kwargs...) = nothing
function savesettings(dirname::AbstractString, nsolt::AbstractNsolt{T,D}, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), filename="settings", kwargs...) where {T,D}
    filepath = joinpath(dirname, filename)

    txt_general = """
        Settings of $(namestring(nsolt)) Dictionary Learning:
        Element type := $T
        Decimation factor := $(nsolt.decimationFactor)
        Polyphase order := $(nsolt.polyphaseOrder)
        Number of channels := $(nsolt.nChannels)

        Number of training data := $(length(trainingset))
        Epochs := $epochs

        User-defined options for Sparse Coding Stage := $sc_options
        User-defined options for Dictionary Update Stage := $du_options

        """
    txt_others = [ "Keyword $(arg.first) := $(arg.second)\n" for arg in kwargs]

    open(filepath, write=true) do io
        println(io, txt_general, txt_others...)
    end
    vlevel >= 2 && println("Settings was written in $filename.")
    nothing
end

function savesettings(dirname::AbstractString, targets::NTuple{N,CB}, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), filename="settings", kwargs...) where {N,CB<:CodeBook,T,D}
    for idx = 1:N
        savesettings(dirname, targets[idx], trainingset; vlevel=vlevel, epochs=epochs, sc_options=sc_options, du_options=du_options, filename=string(filename,"_",idx))
    end
end

savelogs(::Nothing, args...; kwargs...) = nothing
function savelogs(dirname::AbstractString, nsolt::AbstractNsolt, epoch::Integer; params...)
    filename_logs = joinpath(dirname, "log")
    filename_nsolt = joinpath(dirname, "nsolt.json")

    strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
    strlogs = string("epoch $epoch:", strparams...)
    open(filename_logs, append=true) do io
        println(io, strlogs)
    end
    save(nsolt, filename_nsolt)
    nothing
end

function savelogs(dirname::AbstractString, targets::NTuple{N,CB}, epoch::Integer; params...) where {N,CB<:AbstractNsolt}
    for idx = 1:N
        filename_nsolt = joinpath(dirname, string("nsolt_", idx, ".json"))
        save(targets[idx], filename_nsolt)
    end
    return nothing
    # filename_logs = joinpath(dirname, "log")
    # filename_nsolt = joinpath(dirname, "nsolt.json")
    #
    # strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
    # strlogs = string("epoch $epoch:", strparams...)
    # open(filename_logs, append=true) do io
    #     println(io, strlogs)
    # end
    # save(nsolt, filename_nsolt)
end

function lossfcn(cb::LearningTarget, x::AbstractArray, y::AbstractArray, params; shape=Shapes.Vec()) where {TT<:AbstractArray,TM<:AbstractArray}
    cpcb = similar_dictionary(cb, params)
    syn = createSynthesizer(cpcb, size(x); shape=shape)
    norm(x - synthesize(syn, y))^2/2
end

function similar_dictionary(nsolt::AbstractNsolt, (θ, μ)::Tuple{TT,TM}) where {TT<:AbstractArray,TM<:AbstractArray}
    setrotations!(similar(nsolt, eltype(θ)), θ, μ)
end

similar_dictionary(target::NTuple{N}, params::NTuple{N}) where {N} = similar_dictionary.(target, params)

getParamsDictionary(nsolt::AbstractNsolt) = getrotations(nsolt)
setParamsDictionary!(nsolt::AbstractNsolt, pm::NTuple{2}) = setrotations!(nsolt, pm...)

getParamsDictionary(targets::NTuple) = getParamsDictionary.(targets)
setParamsDictionary!(targets::NTuple, pm::NTuple) = setParamsDictionary!.(targets, pm)

getOperatorType_scs(::Type, ::Val) = AbstractOperator
getOperatorType_scs(::Type{NS}, ::Val{:convolution}) where {NS<:AbstractNsolt} = ConvolutionalOperator

decompose_params(::Type, params) = (params, ())
compose_params(::Type, params, ()) = params

decompose_params(::Type{NS}, (θ, μ)) where {NS<:AbstractNsolt} = (θ, μ)
compose_params(::Type{NS}, θ, μ) where {NS<:AbstractNsolt} = (θ, μ)

function decompose_params(::Type{NTuple{N,T}}, params::NTuple{N}) where {N,T<:AbstractNsolt}
    θs, μs = map(t->t[1], params), map(t->t[2], params)
    vcat(θs...), (length.(θs), μs)
end

function compose_params(::Type{NTuple{N,T}}, vp::AbstractArray, pminfo::Tuple) where {N,T<:AbstractNsolt}
    arrpms = map(rng->vp[rng],intervals(pminfo[1]))
    map((lhs,rhs)->(lhs,rhs,), arrpms, pminfo[2])
end

initialstate(spr::Optimizers.Steepest, args...) = ()
initialstate(spr::Optimizers.AdaGrad, args...) = 0.0
initialstate(adam::Optimizers.Adam, ∇f::Function, vecpm::AbstractArray) = (zero(vecpm), 0.0)

updateamount(stp::Optimizers.Steepest, grad, args...) = (stp.rate .* grad, ())
function updateamount(agd::Optimizers.AdaGrad, grad::AbstractArray, ::Integer, v::AbstractFloat)
    v = v + norm(grad)^2
    upm = agd.rate / (sqrt(v) + agd.ϵ) * grad
    (upm, v)
end
function updateamount(adam::Optimizers.Adam, grad::AbstractArray, itr::Integer, (m, v)::Tuple{TM,TV}) where {TM<:AbstractArray,TV<:AbstractFloat}
    m = adam.β1 * m + (1 - adam.β1) * grad
    v = adam.β2 * v + (1 - adam.β2) * norm(grad)^2
    m̂ = m / (1 - adam.β1^(itr))
    v̂ = v / (1 - adam.β2^(itr))

    upm = adam.rate * m̂ / (sqrt(v̂) + adam.ϵ)
    (upm, (m, v))
end

namestring(nsolt::Rnsolt) = namestring(nsolt, "Real NSOLT")
namestring(nsolt::Cnsolt) = namestring(nsolt, "Complex NSOLT")
namestring(nsolt::AbstractNsolt{T,D}, strnsolt::AbstractString) where {T,D} = "$D-dimensional $(ifelse(istype1(nsolt), "Type-I", "Type-II")) $strnsolt"

verbosenames() = Dict(:none => 0, :standard => 1, :specified => 2, :loquacious => 3)
verboselevel(sym::Symbol) = verbosenames()[sym]
verboselevel(lv::Integer) = lv
