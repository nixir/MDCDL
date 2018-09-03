# RNSOLT dictionary learning with 1-vanishing moment

using NLopt
using MDCDL
using TestImages, Images
using Plots
using LinearAlgebra
using Printf: @printf
# cnt = 0

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

szSubData = tuple(fill(16,D)...)
nSubData = 64
nEpoch = 10

nsolt = Rnsolt(df, ord, nch)
MDCDL.rand!(nsolt; isPropMat=false)

msanalyzer = MDCDL.createMultiscaleAnalyzer(nsolt, szSubData; level=lv, shape=:vector)
mssynthesizer = MDCDL.createMultiscaleSynthesizer(nsolt, szSubData; level=lv, shape=:vector)

orgImg = Array{RGB{Float64}}(testimage("lena"))
trainingIds = map(1:nSubData) do nsd
    pos = rand.(UnitRange.(0,size(orgImg) .- szSubData))
    UnitRange.(1 .+ pos, szSubData .+ pos)
end

# y0 = analyze(msnsolt, orgImg[trainingIds[1]...]; shape=:vector)
y0 = msanalyzer(orgImg[trainingIds[1]...])
sparsity = fld(length(y0),4)

angs0, mus0 = getAngleParameters(nsolt)
angs0s = angs0[sum(nch):end]

opt = Opt(:GN_CRS2_LM, length(angs0s))
# opt = Opt(:LD_MMA, length(angs0))
lower_bounds!(opt, -1*pi*ones(size(angs0s)))
upper_bounds!(opt,  1*pi*ones(size(angs0s)))
xtol_rel!(opt,1e-4)
maxeval!(opt,400)

# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

# (minf, minx, ret) = optimize(opt, [1.234, 5.678])
y = y0
for epoch = 1:nEpoch
    errs = fill(Inf, nSubData)
    for idx = 1:nSubData
        subx = trainingIds[idx]
        global y
        x = orgImg[subx...]
        hy = iht(mssynthesizer, msanalyzer, x, y, sparsity; iterations=400, isverbose=false, lt=(lhs,rhs)->isless(norm(lhs),norm(rhs)))
        # cnt = 0
        objfunc = (angs::Vector, grad::Vector) -> begin
            # global cnt
            # cnt::Int += 1

            angsvm1 = vcat(zeros(sum(nch)-1), angs)
            setAngleParameters!(nsolt, angsvm1, mus0)
            dist = x .- mssynthesizer(hy)
            cst = norm(dist)^2

            # println("f_$(cnt): cost=$(cst)")

            cst
        end
        min_objective!(opt, objfunc)
        (minf, minx, ret) = optimize(opt, angs0s)
        errs[idx] = minf

        minxt = vcat(zeros(sum(nch)-1), minx);
        setAngleParameters!(nsolt, minxt, mus0)
        y = msanalyzer(x)
        @printf("dataset %3d: cost = %.4e\n", idx, errs[idx])
    end
    @printf("epoch %3d finished, total cost = %.4e\n", epoch, sum(errs))
end

atmimshow(nsolt; scale=0.6)

if !isempty(filename)
    MDCDL.save(filename, msnsolt.filterBank)
end
