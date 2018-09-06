# RNSOLT dictionary learning with 1-vanishing moment

using LinearAlgebra
using MDCDL
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
const doWriteResults = true

# data dimension
const D = 2
# decimation factor
const df = (4,4)
# polyphase order
const ord = (2,2)
# number of symmetric/antisymmetric channel
const nch = 18

const dt = Float64

const η = 1e-4

const szSubData = tuple(fill(16 ,D)...)
const nSubData = 32
const nEpoch = 2

const sparsity = 0.6 # ∈ [0, 1.0]
#####################################

resultsdir_parent = joinpath(@__DIR__, "results")
!isdir(resultsdir_parent) && mkdir(resultsdir_parent)

tm = Dates.format(now(), "yyyy_mm_dd_SS_sss")
resultsdir = joinpath(resultsdir_parent, tm)
!isdir(resultsdir) && mkdir(resultsdir)

logfile = joinpath(resultsdir, "log")
datafile = joinpath(resultsdir, "nsolt")

nsolt = Cnsolt(dt, df, ord, nch)
MDCDL.rand!(nsolt; isInitMat=true, isPropMat=true)
orgNsolt = deepcopy(nsolt)

orgImg = Array{dt}(testimage("cameraman"))
trainingIds = map(1:nSubData) do nsd
    pos = rand.(UnitRange.(0,size(orgImg) .- szSubData))
    UnitRange.(1 .+ pos, szSubData .+ pos)
end

analyzer = createAnalyzer(nsolt, szSubData; shape=:vector)
y0 = analyzer(orgImg[trainingIds[1]...])
# nSparseCoefs = fld(length(y0), 2)
nSparseCoefs = floor(Int, sparsity*length(y0))

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

        grad = MDCDL.gradSqrdError(nsolt, pvx, pvy)
        angs, mus = getAngleParameters(nsolt)
        angs = angs - η*grad
        setAngleParameters!(nsolt, angs, mus)

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
