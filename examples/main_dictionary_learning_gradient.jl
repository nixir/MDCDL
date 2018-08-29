# RNSOLT dictionary learning with 1-vanishing moment

using NLopt
using MDCDL
using TestImages, Images
using Plots
using Base.Printf: @printf
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
# include(joinpath(Pkg.dir(),"MDCDL","test","randomInit.jl"))
MDCDL.rand!(nsolt)

orgImg = Array{dt}(testimage("cameraman"))
trainingIds = map(1:nSubData) do nsd
        pos = rand.(UnitRange.(0,size(orgImg) .- szSubData))
        UnitRange.(1 .+ pos, szSubData .+ pos)
end

y0 = analyze(nsolt, orgImg[trainingIds[1]...]; outputMode=:vector)
sparsity = fld(length(y0), 8)

angs0, mus0 = getAngleParameters(nsolt)
# angs0s = angs0[nch[1]:end]

y = y0
for epoch = 1:nEpoch, nd in 1:length(trainingIds)
    global y
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

        setAngleParameters!(tfb, angs, mus0)

        grad .= MDCDL.gradSqrdError(tfb, pvx, pvy)

        dist = pvx.data - synthesize(tfb, pvy).data
        println(dist)
        cst = vecnorm(dist)^2

        # println("f_$(cnt):\t cost = $(cst),\t |grad| = $(vecnorm(grad))")
        # @printf("f_%4d): cost = %.6e, |grad| = %.6e\n", cnt, cst, vecnorm(grad))
        # sleep(0.2)
        cst
    end

    lopt = Opt(:LN_COBYLA, length(angs0))
    # lopt = Opt(:LD_MMA, length(angs0))
    maxeval!(lopt,100)
    min_objective!(lopt, objfunc)
    (minf, minx, ret) = optimize(lopt, angs0)

    ##### global optimization #####
    # gopt = Opt(:GD_MLSL, length(angs0s))
    # lower_bounds!(gopt, -1*pi*ones(size(angs0s)))
    # upper_bounds!(gopt,  1*pi*ones(size(angs0s)))
    # local_optimizer!(gopt, lopt)
    # min_objective!(gopt, objfunc)
    # (minf, minx, ret) = optimize(gopt, angs0s)
    ##### #####

    # minxt = vcat(zeros(dt, sum(nch)-1), minx);
    setAngleParameters!(nsolt, minx, mus0)
    y = analyze(nsolt, x; outputMode=:vector)
    # println("Epoch: $epoch, No.: $nd, cost = $(minf)")
    @printf("Epoch: %4d, No.: %3d, cost = %.6e\n", epoch, nd, minf)
end

atmimshow(nsolt)

if !isempty(filename)
    MDCDL.save(filename, nsolt)
end
