# RNSOLT dictionary learning with 1-vanishing moment

using NLopt
using MDCDL
using TestImages, Images
using Plots
using LinearAlgebra
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
# tree level
lv = 3

szSubData = tuple(fill(8,D)...)
nSubData = 64
nEpoch = 10

nsolt = Rnsolt(df, ord, nch)
# include(joinpath(Pkg.dir(),"MDCDL","test","randomInit.jl"))
# rand!(nsolt)
msnsolt = Multiscale(nsolt, lv)

orgImg = Array{RGB{Float64}}(testimage("lena"))
trainingIds = [ (colon.(1,szSubData) .+ rand.(colon.(0,size(orgImg) .- szSubData))) for nsd in 1:nSubData ]

y0 = analyze(msnsolt, orgImg[trainingIds[1]...]; shape=:vector)
sparsity = fld(length(y0),4)

angs0, mus0 = getAngleParameters(msnsolt.filterBank)
angs0s = angs0[sum(nch):end]

opt = Opt(:LN_COBYLA, length(angs0s))
# opt = Opt(:LD_MMA, length(angs0))
lower_bounds!(opt, -1*pi*ones(size(angs0s)))
upper_bounds!(opt,  1*pi*ones(size(angs0s)))
xtol_rel!(opt,1e-4)
maxeval!(opt,400)

# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

# (minf, minx, ret) = optimize(opt, [1.234, 5.678])
y = y0
for idx = 1:nEpoch, subx in trainingIds
    x = orgImg[subx...]
    hy = MDCDL.iht(msnsolt, x, y, sparsity; maxIterations=400, viewStatus=true, lt=(lhs,rhs)->isless(norm(lhs),norm(rhs)))
    cnt = 0
    objfunc = (angs::Vector, grad::Vector) -> begin
        global cnt
        cnt::Int += 1

        angsvm1 = vcat(zeros(sum(nch)-1), angs)
        setAngleParameters!(msnsolt.filterBank, angsvm1, mus0)
        dist = x .- synthesize(msnsolt, hy, size(x))
        cst = norm(dist)^2

        println("f_$(cnt): cost=$(cst)")

        cst
    end
    min_objective!(opt, objfunc)
    (minf, minx, ret) = optimize(opt, angs0s)

    minxt = vcat(zeros(sum(nch)-1), minx);
    setAngleParameters!(msnsolt.filterBank, minxt, mus0)
    y = analyze(msnsolt, x; shape=:vector)
    println("Iterations $idx finished.")
end

atmimshow(msnsolt.filterBank)

if !isempty(filename)
    MDCDL.save(filename, msnsolt.filterBank)
end
