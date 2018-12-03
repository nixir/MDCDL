using MDCDL
using LinearAlgebra

using TestImages: testimage
using Plots: plot, png

using Base.Filesystem
using Dates: format, now

function main()
    ########## Configurations #########
    # choose NSOLT type: (Rnsolt | Cnsolt)
    Nsolt = Rnsolt
    # TP := eltype(Nsolt) (<:AbstractFloat)
    TP = Float64
    # NSOLT dimensions
    D = 2
    # decimation factor: (<:NTuple{D,Int} where D)
    df = (2,2)
    # polyphase order: (<:NTuple{D,Int} where D)
    ord = (4,4)
    # number of channels: (<:Union{Integer,Tuple{Int,Int}} for Rnsolt)
    #                     (<:Integer for Cnsolt)
    nch = 8

    # size of minibatches (<:NTuple{D,Int})
    szx = (16,16)
    # number of minibatches (<:Integer)
    nSubData = 32

    # path of log files (do nothing if isa(logdir, Nothing))
    # logdir = begin
    #     tm = format(now(), "yyyymmdd_HH_MM_SS_sss")
    #     joinpath(@__DIR__, "results", tm)
    # end
    logdir = nothing
    # save minibatches?
    do_save_trainingset = false

    # options for sparse coding
    sc_options1 = ( iterations = 500, sparsity = 0.5, filter_domain=:convolution)
    # options for dictionary update
    du_options1 = ( iterations = 100, stepsize = 1e-3,)

    # general options of dictionary learning
    options1 = ( epochs  = 30,
                verbose = :standard,
                sc_options = sc_options1,
                du_options = du_options1,
                logdir = logdir,)

    # options for sparse coding
    sc_options2 = ( iterations = 500, sparsity = 0.5, filter_domain=:convolution)
    # options for dictionary update
    du_options2 = ( iterations = 100, stepsize = 1e-4,)

    # general options of dictionary learning
    options2 = ( epochs  = 300,
                verbose = :standard,
                sc_options = sc_options2,
                du_options = du_options2,
                logdir = logdir,)
    ####################################
    logdir != nothing && !isdir(logdir) && mkpath(logdir)
    do_save_trainingset = do_save_trainingset && logdir != nothing

    # original image
    orgImg = TP.(testimage("cameraman"))

    # generate minibatches of orgImage
    trainingSet = map(1:nSubData) do nsd
        pos = rand.(UnitRange.(0,size(orgImg) .- szx))
        orgImg[UnitRange.(1 .+ pos, szx .+ pos)...]
    end

    if do_save_trainingset
        datadir = joinpath(logdir, "data")
        !isdir(datadir) && mkpath(datadir)
        map(idx->save(joinpath(datadir, "$idx.png"), trainingSet[idx]), 1:nSubData)
    end

    nsolt_core = Nsolt(TP, df, (fill(0,D)...,), nch)
    MDCDL.rand!(nsolt_core, isSymmetry = false)
    MDCDL.train!(nsolt_core, trainingSet; options1...)

    # create NSOLT instance
    nsolt = similar(nsolt_core, TP, df, ord, nch)
    nsolt.initMatrices .= nsolt_core.initMatrices

    # dictionary learning
    MDCDL.train!(nsolt, trainingSet; options2...)

    #
    #plot(nsolt)

    return nsolt
end

main()
