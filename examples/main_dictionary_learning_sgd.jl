# RNSOLT dictionary learning with 1-vanishing moment

using LinearAlgebra
using MDCDL
using ForwardDiff
using TestImages
# using Images
using ColorTypes
using Statistics
using Printf: @printf, @sprintf
using Plots

using Base.Filesystem
using Dates

########## Configurations #########
cnt = 0

# output file name
doWriteResults = true

# choose NSOLT type (Rnsolt | Cnsolt)
Nsolt = Cnsolt

# data dimension
D = 2
# decimation factor
df = (2,2)
# polyphase order
ord = (4,4)
# number of symmetric/antisymmetric channel
nch = 8

dt = Float64

η = 1e-4

szSubData = tuple(fill(32 ,D)...)
nSubData = 64
nEpoch = 200

sparsity = 0.6 # ∈ [0, 1.0]
#######################

resultsdir_parent = joinpath(@__DIR__, "results")
!isdir(resultsdir_parent) && mkdir(resultsdir_parent)

tm = Dates.format(now(), "yyyy_mm_dd_SS_sss")
resultsdir = joinpath(resultsdir_parent, tm)
!isdir(resultsdir) && mkdir(resultsdir)

logfile = joinpath(resultsdir, "log")
datafile = joinpath(resultsdir, "nsolt")

nsolt = Nsolt(dt, df, ord, nch)
MDCDL.rand!(nsolt; isInitMat=true, isPropMat=false, isPropAng=false, isSymmetry=false)
orgNsolt = deepcopy(nsolt)

orgImg = Array{dt}(testimage("cameraman"))
trainingIds = map(1:nSubData) do nsd
    pos = rand.(UnitRange.(0,size(orgImg) .- szSubData))
    UnitRange.(1 .+ pos, szSubData .+ pos)
end

analyzer = createAnalyzer(nsolt, szSubData; shape=:vector)
y0 = analyzer(orgImg[trainingIds[1]...])
# nSparseCoefs = fld(length(y0), 2)
nSparseCoefs = floor(Int, sparsity*prod(szSubData))

angs0, mus0 = getAngleParameters(nsolt)
angs0s = angs0[nch[1]:end]

y = y0
serrs = dt[]
svars = dt[]
for epoch = 1:nEpoch
    errt = Vector{dt}(undef, length(trainingIds))
    for nd in 1:length(trainingIds)
        global y
        subx = trainingIds[nd]
        x = orgImg[subx...]
        hy = MDCDL.iht(nsolt, x, y, nSparseCoefs; iterations=100, isverbose=false)

        pvx = mdarray2polyphase(x, df)
        pvy = mdarray2polyphase(reshape(hy, fld.(szSubData, df)..., sum(nch)))
        θ, μ = getAngleParameters(nsolt)

        f(t) = norm(pvx.data - synthesize(Nsolt(df, ord, nch, t, μ), pvy).data)^2/2
        g(t) = ForwardDiff.gradient(f, t)

        θ -= η*g(θ)
        setAngleParameters!(nsolt, θ, μ)

        synthesizer = createSynthesizer(nsolt, x; shape=:vector)
        adjsyn = createAnalyzer(nsolt, x; shape=:vector)
        y = adjsyn(x)
        errt[nd] = norm(x - synthesizer(hy))^2/2
        # println("Epoch: $epoch, No.: $nd, cost = $(errt[nd])")
    end
    push!(serrs, sum(errt))
    push!(svars, var(errt))
    msg = @sprintf("Epoch %5d finished. sum(cost) = %.4e, svars= %.4e.", epoch, serrs[epoch], svars[epoch])
    println(msg)

    if doWriteResults
        open(logfile, append=true) do io
            println(io, msg)
        end
        MDCDL.save(nsolt, datafile)
    end
end

plot(nsolt)
