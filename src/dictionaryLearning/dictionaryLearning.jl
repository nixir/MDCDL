using ForwardDiff
using Base.Filesystem
using Statistics
using Dates
using FileIO: load
using ImageCore: channelview
using ColorTypes
using NLopt

LearningTarget{N} = Union{CodeBook, NTuple{N, CodeBook}, Multiscale}

function train!(target::LearningTarget, trainingSet::AbstractArray;
    epochs::Integer=1, shape=nothing,
    sparsecoder=SparseCoders.IHT,
    sparsecoder_options=(),
    optimizer=Optimizers.Steepest,
    optimizer_options=(),
    verbose::Union{Integer,Symbol}=1, logdir=Union{Nothing,AbstractString}=nothing,
    plot_function=t->nothing)

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
            x = gettrainingdata(trainingSet[k])
            shapek = getvalidshape(shape, target, sparsecoder, optimizer, x)

            vlevel >= 3 && println("start Sparse Coding Stage.")
            sparse_coefs, loss_sps[k] = stepSparseCoding(sparsecoder, sparsecoder_options, target, x; shape=shapek, vlevel=vlevel)
            vlevel >= 3 && println("end Sparse Coding Stage.")

            vlevel >= 3 && println("start Dictionary Update.")
            params_dic, loss_dus[k] = updateDictionary(optimizer, optimizer_options, target, x, sparse_coefs, params_dic; shape=shapek, vlevel=vlevel)
            vlevel >= 3 && println("end Dictionary Update Stage.")

            vlevel >= 2 && println("epoch #$itr, data #$k: loss(Sparse coding) = $(loss_sps[k]), loss(Dic. update) = $(loss_dus[k]).")

            setParamsDictionary!(target, params_dic)
        end
        if vlevel >= 1
            println("--- epoch #$itr, sum(loss) = $(sum(loss_sps)), var(loss) = $(var(loss_sps))")
        end
        # plot_function(target)

        savelogs(logdir, target, itr;
            vlevel=vlevel,
            time=string(now()),
            loss_sparse_coding=loss_sps,
            loss_dictionary_update=loss_dus)
    end
    vlevel >= 1 && println("training finished.")
    return setParamsDictionary!(target, params_dic)
end

gettrainingdata(filename::AbstractString) = gettrainingdata(FileIO.load(filename))
gettrainingdata(td::AbstractArray{T}) where {T<:AbstractFloat} = td
gettrainingdata(td::AbstractArray{Complex{T}}) where {T<:AbstractFloat} = td
function gettrainingdata(td::AbstractArray{T,D}, TP::Type=Float64) where {D,T<:Color}
    channelview(td) .|> TP
end

getvalidshape(shape::Shapes.AbstractShape, args...) = shape
getvalidshape(::Nothing, ::LearningTarget, sc, du, x::AbstractArray) = Shapes.Vec(size(x))
getvalidshape(::Nothing, ::Multiscale, ::SparseCoders.IHT{T}, args...) where {N,T<:NTuple{N}} = Shapes.Arrayed()

verbosenames() = Dict(:none => 0, :standard => 1, :specified => 2, :loquacious => 3)
verboselevel(sym::Symbol) = verbosenames()[sym]
verboselevel(lv::Integer) = lv

display_filters(::Val{false}, args...; kwargs...) = nothing

display_filters(::Val{true}, nsolt::AbstractNsolt) = display(plot(nsolt))
