include("NSOLT/MDCDL.jl")

using NLopt
using MDCDL
using TestImages, Images

count = 0

D = 2
df = (2,2)
nch = 6
ord = (4,4)
lv = 3

cnsolt = MDCDL.Cnsolt(df, nch, ord)
# cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
cnsolt.initMatrices[1] = Array(qr(rand(nch,nch), thin=false)[1])

for d = 1:D
    map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
        Array(qr(rand(size(A)), thin=false)[1])
    end

    @. cnsolt.paramAngles[d] = rand(size(cnsolt.paramAngles[d]))
end

xraw = testimage("cameraman")
x = complex(Array{Float64}(xraw))
y0, sc = analyze(cnsolt, x, lv)

sparsity = fld.(length.(y0),8)

# y = copy(y0)
# y = MDCDL.softshrink.(y0,3e-5)
# vecnorm(vecnorm.(y-y0))

angs0, mus0 = getAngleParameters(cnsolt)

function objfunc(angs::Vector, grad::Vector)
    global count
    count::Int += 1

    setAngleParameters!(cnsolt, angs, mus0)

    y, sc = analyze(cnsolt, x, lv)

    hy = MDCDL.hardshrink.(y, sparsity)
    # hy = y

    tx = synthesize(cnsolt, hy, sc, lv)
    dist = x - tx
    cst = vecnorm(x - tx)^2

    # #TODO: 勾配の計算式を間違えている可能性がある．
    # #TODO: 非常に処理が重いので計算方法を考える
    # if length(grad) > 0
    #     grad = map(1:length(angs)) do idx
    #         cc = MDCDL.Cnsolt(df, nch, ord)
    #         angd = zeros(size(angs))
    #         angd[idx] = angs[idx] + pi/2
    #
    #         setAngleParameters!(cc, angs, mus0)
    #         dx = synthesize(cc, y, sc, lv)
    #
    #         # 多次元データに対する内積の簡潔な表現に直す
    #         -2*real( sum(conj(dist) .* dx) )
    #     end
    # end

    println("f_$(count): cost=$(cst)")

    cst
end

myfunc = objfunc

opt = Opt(:LN_COBYLA, length(angs0))
# opt = Opt(:LD_MMA, length(angs0))
lower_bounds!(opt, -1*pi*ones(size(angs0)))
upper_bounds!(opt,  1*pi*ones(size(angs0)))
xtol_rel!(opt,1e-4)
maxeval!(opt,500)

min_objective!(opt, myfunc)
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

# (minf, minx, ret) = optimize(opt, [1.234, 5.678])
(minf, minx, ret) = optimize(opt, angs0)

println("got $minf at $minx after $count iterations (returned $ret)")

setAngleParameters!(cnsolt, minx, mus0)

af = getAnalysisFilters(cnsolt)


afimg1 = Array{Gray{N0f8}}(min.(abs.(af[6]),1.0))
