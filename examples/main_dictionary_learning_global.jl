using MDCDL
using LinearAlgebra

using TestImages: testimage
using Images: imresize
using Plots: plot, png

using Base.Filesystem
using Dates: format, now

function main()
    ########## Configurations #########
    # choose NSOLT type: (Rnsolt | Cnsolt)
    Nsolt = Cnsolt
    # TP := eltype(Nsolt) (<:AbstractFloat)
    TP = Float64
    # decimation factor: (<:NTuple{D,Int} where D is #dims)
    df = (2,2)
    # polyphase order: (<:NTuple{D,Int} where D)
    ord = (2,2)
    # number of channels: (<:Union{Integer,Tuple{Int,Int}} for Rnsolt)
    #                     (<:Integer for Cnsolt)
    nch = 8

    # size of minibatches (<:NTuple{D,Int})
    szx = (64,64)

    # path of log files (do nothing if isa(logdir, Nothing))
    # logdir = joinpath(@__DIR__, "results", format(now(), "yyyymmdd_HH_MM_SS_sss"))
    logdir = nothing
    # save minibatches?
    do_save_trainingset = false
    do_export_atoms = false

    # options for sparse coding
    sparsity = 0.5
    sparsecoder = SparseCoders.IHT
    sparsecoder_options = (
        iterations = 10000,
        nonzeros = trunc(Int, sparsity * prod(szx)), filter_domain=:convolution,)
    # options for dictionary update
    # optimizer = Optimizers.Steepest
    # optimizer_options = (
    #     rate = 1e-4,
    #     iterations=1,)
    optimizer = Optimizers.GlobalOpt
    optimizer_options = (
        alg = :GN_DIRECT_L_RAND,
        iterations = 10000,
    )

    # general options of dictionary learning
    options = ( epochs  = 10,
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
    trainingSet = [ imresize(orgImg, szx) .|> Float64 ]

    # create NSOLT instance
    nsolt = Nsolt(TP, df, ord, nch)
    # set random orthonormal matrices to the initial matrices.
    # MDCDL.rand!(nsolt, isPropMat = false, isPropAng = false, isSymmetry = false)
    θ, μ = getrotations(nsolt)
    θ += π/6*randn(size(θ)...)
    setrotations!(nsolt, θ, μ)

    # dictionary learning
    MDCDL.train!(nsolt, trainingSet; options...)

    if logdir != nothing && do_export_atoms
        png(plot(nsolt), joinpath(logdir, "atoms.png"))
    end

    return nsolt
end

nsolt = main()
