using NLopt
using MDCDL
using TestImages, Images

include(joinpath(Pkg.dir(),"MDCDL/test/randomInit.jl"))

count = 0

D = 2
df = (2,2)
ord = (2,2)
nch = (3,3)
lv = 3
nIters = 20

nsolt = Rnsolt(df, ord, nch)
randomInit!(nsolt; isSymmetry = false)

xraw = testimage("cameraman")[((1:128,1:128) .+ (64,168))...]
x = complex(Array{Float64}(xraw))

y0 = analyze(nsolt, x, lv; outputMode = :augumented)
sparsity = fld.(length.(y0),4)

angs0, mus0 = getAngleParameters(nsolt)


opt = Opt(:LN_COBYLA, length(angs0))
# opt = Opt(:LD_MMA, length(angs0))
lower_bounds!(opt, -1*pi*ones(size(angs0)))
upper_bounds!(opt,  1*pi*ones(size(angs0)))
xtol_rel!(opt,1e-4)
maxeval!(opt,200)

# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

# (minf, minx, ret) = optimize(opt, [1.234, 5.678])
y = y0
for idx = 1:nIters
    # hy = map((ys, sp) -> MDCDL.iht(nsol, x, ys, sp), y, sparsity)
    # hy = y
    hy = MDCDL.iht(
        (ty)->synthesize(nsolt, ty, lv),
        (tx)->adjoint_synthesize(nsolt, tx, lv; outputMode=:augumented),
        x, y, sparsity; viewStatus=true
    )
    count = 0
    objfunc = (angs::Vector, grad::Vector) -> begin
        global count
        count::Int += 1

        setAngleParameters!(nsolt, angs, mus0)
        dist = x - synthesize(nsolt, hy, lv)
        cst = vecnorm(dist)^2

        # #TODO: 勾配の計算式を間違えている可能性がある．
        # #TODO: 非常に処理が重いので計算方法を考える
        # if length(grad) > 0
        #     grad = map(1:length(angs)) do idx
        #         cc = MDCDL.nsolt(df, nch, ord)
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
    min_objective!(opt, objfunc)
    (minf, minx, ret) = optimize(opt, angs0)

    setAngleParameters!(nsolt, minx, mus0)
    y = analyze(nsolt, x, lv; outputMode=:augumented)
    println("Iterations $idx finished.")
end

# println("got $minf at $minx after $count iterations (returned $ret)")
