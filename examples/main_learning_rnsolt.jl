using MDCDL
using NLopt
using Images, ImageView, TestImages

include(joinpath(Pkg.dir(),"MDCDL","test","randomInit.jl"))
srand(23485729)

##### configurations #####
D = 2
lv = 3
szSubData = ntuple( d -> 64, D)
nSubData = 64
nEpoch = 10
nIterIht = 300
dt = Float64

# nnzCoefs = fld(prod(szSubData), 32)
nnzCoefs = 256
##########################


orgImg = testimage("cameraman")
normalizedImg = Array{Float64}(orgImg)
# normalizedImg = orgImg .- (sum(orgImg) / length(orgImg))
x = [ normalizedImg[(colon.(1,szSubData) .+ rand.(colon.(0,size(normalizedImg) .- szSubData)))...] for nsd in 1:nSubData ]

df = tuple(fill(2,D)...)
ord = tuple(fill(2,D)...)
nch = (cld(prod(df),2), fld(prod(df),2)) .+ (1,1)
nsolt = Rnsolt(dt, df, ord, nch)
randomInit!(nsolt; isSymmetry = false)

for epoch = 1:nEpoch, k = 1:length(x)
    println("epoch = $epoch, #dataset = $k.")
    xk = x[k]

    # sparse coding
    println("*** Sparse Coding Stage ***")
    y = analyze(nsolt, xk, lv; outputMode=:augumented)
    # TODO
    hy = MDCDL.iht((ty)->synthesize(nsolt, ty, lv), (ty)->adjoint_synthesize(nsolt, ty, lv; outputMode=:augumented), xk, y, fill(nnzCoefs,lv); viewStatus = true, maxIterations = nIterIht)

    println("*** Dictionary Learning Stage ***")
    # dictionary learning
    angles0, mus0 = MDCDL.getAngleParameters(nsolt)

    nzk = fld(length(xk),16)
    count = 0
    objfunc = (angs, grad) -> begin
        count += 1

        setAngleParameters!(nsolt, angs, mus0)

        tx = synthesize(nsolt, hy, lv)
        diffx = xk - tx

        cst = vecnorm(diffx)^2
        println("f_$(count): cost=$(cst)")

        cst
    end

    opt = Opt(:LN_COBYLA, length(angles0))
    lower_bounds!(opt, -1*pi*ones(size(angles0)))
    upper_bounds!(opt,  1*pi*ones(size(angles0)))
    xtol_rel!(opt,1e-4)
    maxeval!(opt,200)

    min_objective!(opt, objfunc)
    (minf, minx, ret) = optimize(opt, angles0)

    MDCDL.setAngleParameters!(nsolt, minx, mus0)
end

MDCDL.save(joinpath(Pkg.dir(),"MDCDL","examples","design",string(now())))
