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
ord = (0,0)
# number of symmetric/antisymmetric channel
nch = (16,16)

dt = Float64

# η = 1e-5

szSubData = tuple(fill(64,D)...)
nSubData = 32
nEpoch = 10

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
    hy = MDCDL.iht(nsolt, x, y, sparsity; maxIterations=100, viewStatus=false, lt=(lhs,rhs)->isless(norm(lhs), norm(rhs)))

    pvx = mdarray2polyphase(x, df)
    pvy = mdarray2polyphase(reshape(hy, fld.(szSubData, df)..., sum(nch)))

    cnt = 0
    # tfb = Rnsolt(dt, df, ord, nch)
    objfunc = (angs::Vector, grad::Vector) -> begin
        global cnt
        cnt::Int += 1
        tfb = Rnsolt(dt, df, ord, nch)

        angsvm1 = vcat(zeros(dt, sum(nch)-1), angs)
        setAngleParameters!(tfb, angsvm1, mus0)

        grad .= MDCDL.gradSqrdError(tfb, pvx, pvy)

        dist = pvx.data - synthesize(tfb, pvy).data
        cst = vecnorm(dist)^2

        println("f_$(cnt):\t cost = $(cst),\t |grad| = $(vecnorm(grad))")
        # sleep(0.2)
        cst
    end

    # opt = Opt(:LN_COBYLA, length(angs0s))
    lopt = Opt(:LD_MMA, length(angs0s))
    maxeval!(lopt,100)
    min_objective!(lopt, objfunc)
    (minf, minx, ret) = optimize(lopt, angs0s)

    ##### global optimization #####
    # gopt = Opt(:GD_MLSL, length(angs0s))
    # lower_bounds!(gopt, -1*pi*ones(size(angs0s)))
    # upper_bounds!(gopt,  1*pi*ones(size(angs0s)))
    # local_optimizer!(gopt, lopt)
    # min_objective!(gopt, objfunc)
    # (minf, minx, ret) = optimize(gopt, angs0s)
    ##### #####

    minxt = vcat(zeros(dt, sum(nch)-1), minx);
    setAngleParameters!(nsolt, minxt, mus0)
    y = analyze(nsolt, x; outputMode=:vector)
    println("Epoch: $epoch, No.: $nd, cost = $(minf)")
end

atmimshow(nsolt)

if !isempty(filename)
    MDCDL.save(filename, nsolt)
end
