# RNSOLT dictionary learning with 1-vanishing moment

using NLopt
using MDCDL
using TestImages, Images
using Plots
cnt = 0

# output file name
filename = ""

# data dimension
D = 2
# decimation factor
df = (2,2)
# polyphase order
ord = (2,2)
# number of symmetric/antisymmetric channel
nch = (4,4)

dt = Float64

# η = 1e-5

szSubData = tuple(fill(32,D)...)
nSubData = 32
nEpoch = 10

nsolt = Rnsolt(dt, df, ord, nch)
include(joinpath(Pkg.dir(),"MDCDL","test","randomInit.jl"))
randomInit!(nsolt)

orgImg = Array{dt}(testimage("cameraman"))
trainingIds = [ (colon.(1,szSubData) .+ rand.(colon.(0,size(orgImg) .- szSubData))) for nsd in 1:nSubData ]

y0 = analyze(nsolt, orgImg[trainingIds[1]...]; outputMode=:vector)
sparsity = fld(length(y0), 8)

angs0, mus0 = getAngleParameters(nsolt)
angs0s = angs0[nch[1]:end]

# opt = Opt(:LN_COBYLA, length(angs0s))
opt = Opt(:LD_CCSAQ, length(angs0s))
# lower_bounds!(opt, -1*pi*ones(size(angs0s)))
# upper_bounds!(opt,  1*pi*ones(size(angs0s)))
xtol_rel!(opt,1e-4)
maxeval!(opt,2000)

# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,2,0), 1e-8)
# inequality_constraint!(opt, (x,g) -> myconstraint(x,g,-1,1), 1e-8)

# (minf, minx, ret) = optimize(opt, [1.234, 5.678])
y = y0
for idx = 1:nEpoch, subx in trainingIds
    x = orgImg[subx...]
    hy = MDCDL.iht(nsolt, x, y, sparsity; maxIterations=500, viewStatus=true, lt=(lhs,rhs)->isless(norm(lhs), norm(rhs)))
    cnt = 0
    objfunc = (angs::Vector, grad::Vector) -> begin
        global cnt
        cnt::Int += 1

        # println(typeof(angs))
        angsvm1 = vcat(zeros(dt, sum(nch)-1), angs)
        # println(typeof(angsvm1))
        ###########################

        tfb = Rnsolt(dt, df, ord, nch)
        setAngleParameters!(tfb, angsvm1, mus0)
        rx = synthesize(tfb, hy, size(x))

        nBlocks = mdarray2polyphase(x - rx, df).nBlocks
        et = mdarray2polyphase(x - rx, df).data

        yt = begin
            yaug = reshape(hy, fld.(size(x),df)..., sum(nch))
            mdarray2polyphase(yaug).data
        end

        fet = tfb.matrixC * flipdim(et, 1)

        M = prod(df)
        cM, fM = cld(M,2), fld(M,2)

        Pu, Pl = eye(dt, nch[1], cM), eye(dt, nch[2], fM)
        P = zeros(dt, sum(nch), M)
        P[1:nch[1],1:cM] = Pu
        P[nch[1]+1:end,cM+1:end] = Pl

        pet = P*fet
        # rand!(pet)

        # println(η*max(maximum(gdw), maximum(gdu)))
        W0 = copy(tfb.initMatrices[1])
        U0 = copy(tfb.initMatrices[2])

        angsw, musw = MDCDL.mat2rotations(W0)
        angsu, musu = MDCDL.mat2rotations(U0)

        gdw = -MDCDL.scalarGradOfOrthonormalMatrix(yt[1:nch[1],:], pet[1:nch[1],:], angsw, musw)
        gdu = -MDCDL.scalarGradOfOrthonormalMatrix(yt[nch[1]+1:end,:], pet[nch[1]+1:end,:], angsu, musu)

        # tfb.initMatrices[1] .= MDCDL.rotations2mat(angsw - η*gdw, musw)
        # tfb.initMatrices[2] .= MDCDL.rotations2mat(angsu - η*gdu, musu)

        pet[1:nch[1],:] = W0*pet[1:nch[1],:]
        pet[nch[1]+1:end,:] = U0*pet[nch[1]+1:end,:]

        hP = tfb.nChannels[1]

        L = fld(nch[2]*(nch[2]-1),2)
        gdudk = [ fill(zeros(dt,L), o) for o in ord ]
        for d = 1:D
            nShift = fld(size(pet,2), nBlocks[1])
            # pvx = permutedims(pvx)
            pvx = MDCDL.permutedims(PolyphaseVector(pet, nBlocks))
            typeof(pvx)
            pet, nBlocks = pvx.data, pvx.nBlocks
            # submatrices
            xu = view(pet, 1:hP, :)
            xl = view(pet, (1:hP)+hP, :)

            for k = 1:tfb.polyphaseOrder[d]
                tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
                xu .= tu; xl .= tl

                if isodd(k)
                    xl .= circshift(xl, (0, nShift))
                else
                    xu .= circshift(xu, (0, -nShift))
                end
                tu, tl = (xu + xl, xu - xl) ./ sqrt(2)
                xu .= tu; xl .= tl

                txl = tfb.propMatrices[d][k] * xl

                angsu, musu = MDCDL.mat2rotations(tfb.propMatrices[d][k])
                hoge = -MDCDL.scalarGradOfOrthonormalMatrix(yt[nch[1]+1:end,:], xl, angsu, musu)
                gdudk[d][k]  .= hoge /vecnorm(x)
                # tfb.propMatrices[d][k] .= MDCDL.rotations2mat(angsu - η*gdudk[d][k], musu)
                # println(gdu)

                xl .= txl
            end
        end

        grad .= vcat(gdw[nch[1]:end], gdu, vcat(vcat.(gdudk...)...))
        # println(typeof(vcat(vcat.(gdudk)...)))

        # println(size(hoge))
        # println(size(grad))
        setAngleParameters!(nsolt, angsvm1, mus0)
        dist = x .- synthesize(nsolt, hy, size(x))
        cst = vecnorm(dist)^2

        println("f_$(cnt):\t cost = $(cst),\t |grad| = $(vecnorm(grad))")
        sleep(0.2)
        cst
    end
    min_objective!(opt, objfunc)
    (minf, minx, ret) = optimize(opt, angs0s)

    minxt = vcat(zeros(dt, sum(nch)-1), minx);
    setAngleParameters!(nsolt, minxt, mus0)
    y = analyze(nsolt, x; outputMode=:vector)
    println("Iterations $idx finished.")
end

atmimshow(nsolt)

if !isempty(filename)
    MDCDL.save(filename, nsolt)
end
