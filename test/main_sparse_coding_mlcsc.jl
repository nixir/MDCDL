include("NSOLT/MDCDL.jl")
using MDCDL
using NLopt
using Images, ImageView, TestImages

# configurations
const D = 2
const nl = 3

const orgImg = testimage("cameraman")[(1:128)+128,(1:128)+128]
const cplxImg = complex.(Array{Float64}(orgImg))
const x = cplxImg

const dt = Float64

mlcsc = MultiLayerCsc{Float64,D}(nl)

for l = 1:nl
    const Dl = D + l - 1
    # df = ntuple( d -> 2, Dl)
    df = tuple(fill(2,D)..., ones(Integer,l-1)...)
    nch = prod(df) + 2
    ord = ntuple( d -> 0, Dl)
    cnsolt = Cnsolt(df, nch, ord, dataType=dt)

    # initialize parameter matrices
    cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
    cnsolt.initMatrices[1] = Array{dt}(qr(rand(nch,nch), thin=false)[1])

    for d = 1:Dl
        map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
            Array(qr(rand(dt,size(A)), thin=false)[1])
        end

        @. cnsolt.paramAngles[d] = rand(dt,size(cnsolt.paramAngles[d]))
    end

    mlcsc.dictionaries[l] = cnsolt
end

gradOfLossFcn = (yy) -> begin
    - MDCDL.analyze(mlcsc, x - MDCDL.synthesize(mlcsc, yy))
end

# lambda = 1e-4
K = fld(length(x), 4)

# proxFcn = (x, a) -> MDCDL.softshrink(x, a*lambda)

# sparseCoding = v -> MDCDL.fista(gradOfLossFcn, proxFcn, v; viewStatus = true, stepSize = 1.0, maxIterations=50)[1]
sparseCoding = v -> MDCDL.iht(gradOfLossFcn, v, K; viewStatus = true, maxIterations=20)[1]

nItr = 10
for k = 1:nItr
    # sparse coding
    y = analyze(mlcsc, x)
    hy = sparseCoding(y)

    # dictionary learning
    for l = nl:-1:1
        submlcsc = MDCDL.MultiLayerCsc{Float64,D}(l)
        submlcsc.dictionaries .= mlcsc.dictionaries[1:l]

        angles0, mus0 = MDCDL.getAngleParameters(submlcsc.dictionaries[l])

        count = 0
        objfunc = (angs, grad) -> begin
            count += 1

            setAngleParameters!(submlcsc.dictionaries[l], angs, mus0)

            diffx = x - MDCDL.synthesize(submlcsc, hy)

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

        hy = MDCDL.stepSynthesisBank(submlcsc.dictionaries[l], hy; inputMode="augumented")
    end
end


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
