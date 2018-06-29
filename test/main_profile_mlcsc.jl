include("NSOLT/CnsoltSystem.jl")
using CnsoltSystem
using NLopt
using Images, ImageView, TestImages

# configurations
const D = 2
const nl = 3
const szSubData = ntuple( d -> 64, D)
const nSubData = 16

const dt = Float64

const orgImg = testimage("cameraman")
const cplxImg = complex.(Array{Float64}(orgImg))
const normalizedImg = cplxImg .- (sum(cplxImg) / length(cplxImg))
const x = [ normalizedImg[(colon.(1,szSubData) .+ rand.(colon.(0,size(cplxImg) .- szSubData)))...] for nsd in 1:nSubData]


mlcsc = MultiLayerCsc{Float64,D}(nl)

for l = 1:nl
    const Dl = D + l - 1
    # df = ntuple( d -> 2, Dl)
    df = tuple(fill(2,D)..., ones(Integer,l-1)...)
    nch = prod(df) + 2
    ord = ntuple( d -> 2, Dl)
    cnsolt = Cnsolt(df, nch, ord, dataType=dt)

    # initialize parameter matrices
    # cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
    cnsolt.initMatrices[1] = Array{dt}(qr(rand(nch,nch), thin=false)[1])

    # for d = 1:Dl
    #     map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
    #         Array(qr(rand(dt,size(A)), thin=false)[1])
    #     end
    #
    #     @. cnsolt.paramAngles[d] = rand(dt,size(cnsolt.paramAngles[d]))
    # end

    mlcsc.dictionaries[l] = cnsolt
end

# nnzCoefs = fld(prod(szSubData), 32)
nnzCoefs = 256

nEpoch = 2
for epoch = 1:nEpoch, k = 1:length(x)
    const xk = x[k]

    # sparse coding
    println("*** Sparse Coding Stage ***")
    y = CnsoltSystem.analyze(mlcsc, xk)
    hy = CnsoltSystem.iht(mlcsc, xk, complex.(zeros(size(y))), nnzCoefs; viewStatus = true, maxIterations = 30)

    println("*** Dictionary Learning Stage ***")
    # dictionary learning
    for l = nl:-1:1
        println("epoch = $epoch, #dataset = $k, layer = $l.")
        submlcsc = CnsoltSystem.MultiLayerCsc{Float64,D}(l)
        submlcsc.dictionaries .= mlcsc.dictionaries[1:l]

        angles0, mus0 = CnsoltSystem.getAngleParameters(submlcsc.dictionaries[l])

        nzk = fld(length(xk),16)
        count = 0
        objfunc = (angs, grad) -> begin
            count += 1

            setAngleParameters!(submlcsc.dictionaries[l], angs, mus0)

            gamma = Vector(submlcsc.nLayers)
            # gamma[submlcsc.nLayers+1] = hy
            gamma[submlcsc.nLayers] = if l != 1
                CnsoltSystem.hardshrink(CnsoltSystem.stepSynthesisBank(submlcsc.dictionaries[submlcsc.nLayers], hy, inputMode="augumented"), nzk)
            else
                CnsoltSystem.stepSynthesisBank(submlcsc.dictionaries[submlcsc.nLayers], hy, inputMode="augumented")
            end
            for lfun = submlcsc.nLayers-1:-1:1
                dic = submlcsc.dictionaries[lfun]
                gamma[lfun] = CnsoltSystem.stepSynthesisBank(dic, gamma[lfun+1], inputMode="augumented")
            end

            # diffx = xk - CnsoltSystem.synthesize(submlcsc, hy)
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

        CnsoltSystem.setAngleParameters!(mlcsc.dictionaries[l], minx, mus0)

        hy = CnsoltSystem.stepSynthesisBank(submlcsc.dictionaries[l], hy; inputMode="augumented")
    end
end

gs = CnsoltSystem.mlfista(mlcsc, x[1], [1e-8, 1e-6, 1e-3])
recx = CnsoltSystem.synthesize(mlcsc, gs[nl])

errx = vecnorm(recx - x[1])
# rx = CnsoltSystem.synthesize(mlcsc, hy)
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
