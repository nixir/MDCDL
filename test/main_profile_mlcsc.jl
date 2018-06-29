using MDCDL
using NLopt
using Images, ImageView, TestImages

# configurations
D = 2
nl = 3
szSubData = ntuple( d -> 32, D)
nSubData = 1

dt = Float64

orgImg = testimage("cameraman")
cplxImg = complex.(Array{Float64}(orgImg))
normalizedImg = cplxImg .- (sum(cplxImg) / length(cplxImg))
x = [ normalizedImg[(colon.(1,szSubData) .+ rand.(colon.(0,size(cplxImg) .- szSubData)))...] for nsd in 1:nSubData]


mlcsc = MultiLayerCsc{Float64,D}(nl)

for l = 1:nl
    Dl = D + l - 1
    # df = ntuple( d -> 2, Dl)
    df = tuple(fill(2,D)..., ones(Integer,l-1)...)
    nch = prod(df) + 2
    ord = ntuple( d -> 2, Dl)
    cnsolt = Cnsolt(df, nch, ord, dataType=dt)

    # initialize parameter matrices
    # cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
    cnsolt.initMatrices[1] = Array{dt}(qr(rand(nch,nch), thin=false)[1])

    for d = 1:Dl
        map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
            Array(qr(rand(dt,size(A)), thin=false)[1])
        end

        @. cnsolt.paramAngles[d] = rand(dt,size(cnsolt.paramAngles[d]))
    end

    mlcsc.dictionaries[l] = cnsolt
end

# nnzCoefs = fld(prod(szSubData), 32)
nnzCoefs = 256

nEpoch = 2
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

            gamma = Vector(submlcsc.nLayers)
            # gamma[submlcsc.nLayers+1] = hy
            gamma[submlcsc.nLayers] = if l != 1
                MDCDL.hardshrink(MDCDL.stepSynthesisBank(submlcsc.dictionaries[submlcsc.nLayers], hy, inputMode=:augumented), nzk)
            else
                MDCDL.stepSynthesisBank(submlcsc.dictionaries[submlcsc.nLayers], hy, inputMode=:augumented)
            end
            for lfun = submlcsc.nLayers-1:-1:1
                dic = submlcsc.dictionaries[lfun]
                gamma[lfun] = MDCDL.stepSynthesisBank(dic, gamma[lfun+1], inputMode=:augumented)
            end

            # diffx = xk - MDCDL.synthesize(submlcsc, hy)
            diffx = xk - gamma[1]

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

        hy = MDCDL.stepSynthesisBank(submlcsc.dictionaries[l], hy; inputMode=:augumented)
    end
end

gs = MDCDL.mlfista(mlcsc, x[1], [1e-8, 1e-6, 1e-3])
recx = MDCDL.synthesize(mlcsc, gs[nl])

errx = vecnorm(recx - x[1])
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
