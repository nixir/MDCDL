using MDCDL
using NLopt
using Images, ImageView, TestImages

include("../test/randomInit.jl")

##### configurations #####
D = 2
nl = 3
szSubData = ntuple( d -> 64, D)
nSubData = 2
nEpoch = 2
dt = Float64

# nnzCoefs = fld(prod(szSubData), 32)
nnzCoefs = 256
##########################


orgImg = testimage("cameraman")
cplxImg = complex.(Array{Float64}(orgImg))
normalizedImg = cplxImg .- (sum(cplxImg) / length(cplxImg))
x = [ normalizedImg[(colon.(1,szSubData) .+ rand.(colon.(0,size(cplxImg) .- szSubData)))...] for nsd in 1:nSubData]

mlcsc = MultiLayerCsc{Complex{Float64},D}(nl)

for l = 1:nl
    Dl = D + l - 1
    # df = ntuple( d -> 2, Dl)
    df = tuple(fill(2,D)..., ones(Integer,l-1)...)
    nch = prod(df) + 2
    # nch = (fld(prod(df),2) + 1, fld(prod(df),2) + 1)
    ord = ntuple( d -> 2, Dl)

    # cnsolt = Cnsolt(df, nch, ord, dataType=dt)
    cnsolt = Cnsolt(dt, df, nch, ord)
    randomInit!(cnsolt; isSymmetry = false)
    mlcsc.dictionaries[l] = cnsolt
end

for epoch = 1:nEpoch, k = 1:length(x)
    xk = x[k]

    # sparse coding
    println("*** Sparse Coding Stage ***")
    y = MDCDL.analyze(mlcsc, xk)
    hy = MDCDL.iht(mlcsc, xk, complex.(zeros(size(y))), nnzCoefs; viewStatus = true, maxIterations = 30)

    println("*** Dictionary Learning Stage ***")
    # dictionary learning
    for l = nl:-1:1
        println("epoch = $epoch, #dataset = $k, layer = $l.")
        submlcsc = MDCDL.MultiLayerCsc{Float64,D}(l)
        submlcsc.dictionaries .= mlcsc.dictionaries[1:l]

        angles0, mus0 = MDCDL.getAngleParameters(submlcsc.dictionaries[l])

        nzk = fld(length(xk),16)
        count = 0
        objfunc = (angs, grad) -> begin
            count += 1

            setAngleParameters!(submlcsc.dictionaries[l], angs, mus0)

            γ = Vector(submlcsc.nLayers)
            # γ[submlcsc.nLayers+1] = hy
            γ[submlcsc.nLayers] = MDCDL.synthesize( submlcsc.dictionaries[ submlcsc.nLayers ], hy)
            for lfun = submlcsc.nLayers-1:-1:1
                dic = submlcsc.dictionaries[lfun]
                γ[lfun] = MDCDL.synthesize(dic, γ[lfun+1])
            end

            # diffx = xk - MDCDL.synthesize(submlcsc, hy)
            diffx = xk - γ[1]

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

        MDCDL.setAngleParameters!(mlcsc.dictionaries[l], minx, mus0)

        hy = MDCDL.synthesize(submlcsc.dictionaries[l], hy)
    end
end

gs = MDCDL.mlfista(mlcsc, x[1], fill(1e-6,nl))
recx = MDCDL.synthesize(mlcsc, gs[nl])

errx = vecnorm(recx - x[1])
println(errx)

 

# rx = MDCDL.synthesize(mlcsc, hy)
# errx = vecnorm(rx - x)
#
# println("error = $errx")
#
# find(v -> abs(v) < 1e-5, hy)
# find(v -> abs(v) < 1e-5, y0)
#
# length(hy)
#
# length(find(v -> abs(v) > 1e-5, hy)) / length(hy) * 100
