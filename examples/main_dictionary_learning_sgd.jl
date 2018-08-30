# RNSOLT dictionary learning with 1-vanishing moment

# using NLopt
using LinearAlgebra
using MDCDL
using TestImages
# using Images
using ColorTypes
# using Plots
cnt = 0

# output file name
filename = ""

# data dimension
D = 2
# decimation factor
df = (2,2)
# polyphase order
ord = (4,4)
# number of symmetric/antisymmetric channel
nch = (4,4)

dt = Float64

η = 1e-5

szSubData = tuple(fill(32,D)...)
nSubData = 16
nEpoch = 30

nsolt = Rnsolt(dt, df, ord, nch)
# include(joinpath(Pkg.dir(),"MDCDL","test","randomInit.jl"))
MDCDL.rand!(nsolt)

orgImg = Array{dt}(testimage("cameraman"))
trainingIds = map(1:nSubData) do nsd
        pos = rand.(UnitRange.(0,size(orgImg) .- szSubData))
        UnitRange.(1 .+ pos, szSubData .+ pos)
end

y0 = analyze(nsolt, orgImg[trainingIds[1]...]; shape=:vector)
sparsity = fld(length(y0), 4)

angs0, mus0 = getAngleParameters(nsolt)
angs0s = angs0[nch[1]:end]

y = y0
serrs = Vector{dt}(undef, nEpoch)
for epoch = 1:nEpoch
    errt = Vector{dt}(undef, length(trainingIds))
    for nd in 1:length(trainingIds)
        global y
        subx = trainingIds[nd]
        x = orgImg[subx...]
        hy = MDCDL.iht(nsolt, x, y, sparsity; maxIterations=100, viewStatus=false, lt=(lhs,rhs)->isless(norm(lhs), norm(rhs)))

        pvx = mdarray2polyphase(x, df)
        pvy = mdarray2polyphase(reshape(hy, fld.(szSubData, df)..., sum(nch)))

        grad = MDCDL.gradSqrdError(nsolt, pvx, pvy)
        angs, mus = getAngleParameters(nsolt)
        angs = angs - η*grad
        setAngleParameters!(nsolt, angs, mus)

        y = analyze(nsolt, x; shape=:vector)
        errt[nd] = norm(x-synthesize(nsolt,hy,size(x)))^2/2
        println("Epoch: $epoch, No.: $nd, cost = $(errt[nd])")
    end
    serrs[epoch] = sum(errt)
    println("Epoch $epoch finished. sum(cost) = $(serrs[epoch])")
end

# atmimshow(nsolt)

if !isempty(filename)
    MDCDL.save(filename, nsolt)
end
