# RNSOLT dictionary learning with 1-vanishing moment

using NLopt
using MDCDL
using TestImages, Images
using Plots
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
nch = (16,16)

dt = Float64

η = 1e-5

szSubData = tuple(fill(64,D)...)
nSubData = fld(prod(szSubData),4)
nEpoch = 1

nsolt = Rnsolt(dt, df, ord, nch)
include(joinpath(Pkg.dir(),"MDCDL","test","randomInit.jl"))
randomInit!(nsolt)

orgImg = Array{dt}(testimage("cameraman"))
trainingIds = [ (colon.(1,szSubData) .+ rand.(colon.(0,size(orgImg) .- szSubData))) for nsd in 1:nSubData ]

y0 = analyze(nsolt, orgImg[trainingIds[1]...]; outputMode=:vector)
sparsity = fld(length(y0), 8)

angs0, mus0 = getAngleParameters(nsolt)
angs0s = angs0[nch[1]:end]

y = y0
for epoch = 1:nEpoch, nd in 1:length(trainingIds)
    subx = trainingIds[nd]
    x = orgImg[subx...]
    hy = MDCDL.iht(nsolt, x, y, sparsity; maxIterations=1000, viewStatus=true, lt=(lhs,rhs)->isless(norm(lhs), norm(rhs)))

    pvx = mdarray2polyphase(x, df)
    pvy = mdarray2polyphase(reshape(hy, fld.(szSubData, df)..., sum(nch)))

    grad = MDCDL.gradSqrdError(nsolt, pvx, pvy)
    angs, mus = getAngleParameters(nsolt)
    angs = angs - η*grad
    setAngleParameters!(nsolt, angs, mus)

    y = analyze(nsolt, x; outputMode=:vector)
    println("Epoch: $epoch, No.: $nd, cost = $(vecnorm(x-synthesize(nsolt,hy,size(x)))^2/2)")
    sleep(1)
end

atmimshow(nsolt)

if !isempty(filename)
    MDCDL.save(filename, nsolt)
end
