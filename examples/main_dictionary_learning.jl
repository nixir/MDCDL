using NLopt
using MDCDL
using TestImages, Images

count = 0

D = 2
df = (2,2)
ord = (2,2)
nch = (3,3)
lv = 3

szSubData = tuple(fill(64,D)...)
nSubData = 16
nEpoch = 20

nsolt = Rnsolt(df, ord, nch)
msnsolt = Multiscale(nsolt, lv)

orgImg = Array{RGB{Float64}}(testimage("lena"))
trainingSet = [ orgImg[(colon.(1,szSubData) .+ rand.(colon.(0,size(orgImg) .- szSubData)))...] for nsd in 1:nSubData ]

y0 = analyze(msnsolt, trainingSet[1]; outputMode=:vector)
sparsity = fld(length(y0),8)

angs0, mus0 = getAngleParameters(msnsolt.filterBank)

opt = Opt(:LN_COBYLA, length(angs0))
# opt = Opt(:LD_MMA, length(angs0))
lower_bounds!(opt, -1*pi*ones(size(angs0)))
upper_bounds!(opt,  1*pi*ones(size(angs0)))
xtol_rel!(opt,1e-4)
maxeval!(opt,400)

# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

# (minf, minx, ret) = optimize(opt, [1.234, 5.678])
y = y0
for idx = 1:nEpoch, x in trainingSet
    # hy = map((ys, sp) -> MDCDL.iht(nsol, x, ys, sp), y, sparsity)
    # hy = y
    hy = MDCDL.iht(msnsolt, x, y, sparsity; maxIterations=200, viewStatus=true, lt=(lhs,rhs)->isless(norm(lhs),norm(rhs)))
    count = 0
    objfunc = (angs::Vector, grad::Vector) -> begin
        global count
        count::Int += 1

        setAngleParameters!(msnsolt.filterBank, angs, mus0)
        dist = x .- synthesize(msnsolt, hy, size(x))
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

    setAngleParameters!(msnsolt.filterBank, minx, mus0)
    y = analyze(msnsolt, x; outputMode=:vector)
    println("Iterations $idx finished.")
end

# println("got $minf at $minx after $count iterations (returned $ret)")
