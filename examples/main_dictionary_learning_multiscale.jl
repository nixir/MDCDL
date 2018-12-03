using MDCDL
using LinearAlgebra
using Images, TestImages
using Plots

using Base.Filesystem
using Dates

function main()
    ########## Configurations #########
    # choose NSOLT type: (Rnsolt | Cnsolt)
    Nsolt = Rnsolt
    # TP := eltype(Nsolt) (<:AbstractFloat)
    TP = Float64
    # decimation factor: (<:NTuple{D,Int} where D is #dims)
    df = (2,2)
    # polyphase order: (<:NTuple{D,Int} where D)
    ord = (2,2)
    # number of channels: (<:Union{Integer,Tuple{Int,Int}} for Rnsolt)
    #                     (<:Integer for Cnsolt)
    nch = 8
    # number of tree level (<: Integer)
    level = 3

    # size of minibatches (<:NTuple{D,Int})
    szx = (16,16)
    # number of minibatches (<:Integer)
    nSubData = 16

    # path of log files (do nothing if isa(logdir, Nothing))
    # logdir = joinpath(@__DIR__, "results", Dates.format(now(), "yyyymmdd_HH_MM_SS_sss"))
    logdir = nothing
    # save minibatches?
    do_save_trainingset = false
    do_export_atoms = false

    # options for sparse coding
    sparsity = 0.25
    sparsecoder = SparseCoders.IHT
    sparsecoder_options = (
        iterations = 400,
        nonzeros = trunc(Int, sparsity * prod(szx)), filter_domain=:convolution,)
    # options for dictionary update
    optimizer = Optimizers.Steepest
    optimizer_options = (
        rate = 1e-4,
        iterations=10,)

    # general options of dictionary learning
    options = ( epochs  = 1000,
                verbose = :standard, # :none, :standard, :specified, :loquacious
                sparsecoder = sparsecoder,
                sparsecoder_options = sparsecoder_options,
                optimizer = optimizer,
                optimizer_options = optimizer_options,
                logdir = logdir,)
    ####################################
    logdir != nothing && !isdir(logdir) && mkpath(logdir)

    # original image
    orgImg = testimage("cameraman")

    # generate minibatches of orgImage
    trainingSet = map(1:nSubData) do nsd
        pos = rand.(UnitRange.(0,size(orgImg) .- szx))
        orgImg[UnitRange.(1 .+ pos, szx .+ pos)...]
    end

    if logdir != nothing && do_save_trainingset
        datadir = joinpath(logdir, "data")
        !isdir(datadir) && mkpath(datadir)
        map(idx->save(joinpath(datadir, "$idx.png"), trainingSet[idx]), 1:nSubData)
    end

    # create NSOLT instance
    nsolt = Nsolt(TP, df, ord, nch)
    # set random orthonormal matrices to the initial matrices.
    MDCDL.rand!(nsolt, isPropMat = false, isPropAng = false, isSymmetry = false)
    istype1(nsolt) && MDCDL.vm1constraint!(nsolt)

    msnsolt = Multiscale([ deepcopy(nsolt) for l in 1:level ]...)
    # rand!(msnsolt.filterbanks[level], isSymmetry = false, isPropAng=false, isPropMat = false)

    # dictionary learning
    MDCDL.train!(msnsolt, trainingSet; options...)

    if logdir != nothing && do_export_atoms
        for idx = 1:length(msnsolt)
            png(plot(msnsolt.filterbanks[idx]), joinpath(logdir, string("nsolt_",idx,".png")))
        end
    end

    return msnsolt
end

main()
