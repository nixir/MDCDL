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
# randomInit!(nsolt)

orgImg = Array{dt}(testimage("cameraman"))
trainingIds = [ (colon.(1,szSubData) .+ rand.(colon.(0,size(orgImg) .- szSubData))) for nsd in 1:nSubData ]

y0 = analyze(nsolt, orgImg[trainingIds[1]...]; outputMode=:vector)
sparsity = fld(length(y0), 8)

angs0, mus0 = getAngleParameters(nsolt)
angs0s = angs0[nch[1]:end]

y = y0
for epoch = 1:nEpoch, nd in 1:length(trainingIds)
    subx = trainingIds[nd]
    x = orgImg[subx...]
    hy = MDCDL.iht(nsolt, x, y, sparsity; maxIterations=100, viewStatus=false, lt=(lhs,rhs)->isless(norm(lhs), norm(rhs)))
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
        yt = mdarray2polyphase(rx, df).data

        M = prod(df)
        cM, fM = cld(M,2), fld(M,2)

        Pu, Pl = eye(dt, nch[1], cM), eye(dt, nch[2], fM)
        P = zeros(dt, sum(nch), M)
        P[1:nch[1],1:cM] = Pu
        P[nch[1]+1:end,cM+1:end] = Pl

        pet = P * nsolt.matrixC * et
        pyt = P * nsolt.matrixC * yt
        # rand!(pet)

        # println(η*max(maximum(gdw), maximum(gdu)))
        W0 = copy(nsolt.initMatrices[1])
        U0 = copy(nsolt.initMatrices[2])

        pyt[1:nch[1],:] = W0 * pyt[1:nch[1],:]
        pyt[(nch[1]+1):end,:] = U0 * pyt[(nch[1]+1):end,:]

        angsw, musw = MDCDL.mat2rotations(W0)
        angsu, musu = MDCDL.mat2rotations(U0)

        gdw = -MDCDL.scalarGradOfOrthonormalMatrix(pyt[1:nch[1],:], pet[1:nch[1],:], angsw, musw)
        gdu = -MDCDL.scalarGradOfOrthonormalMatrix(pyt[nch[1]+1:end,:], pet[nch[1]+1:end,:], angsu, musu)

        # nsolt.initMatrices[1] .= MDCDL.rotations2mat(angsw - η*gdw, musw)
        # nsolt.initMatrices[2] .= MDCDL.rotations2mat(angsu - η*gdu, musu)

        pet[1:nch[1],:] = W0*pet[1:nch[1],:]
        pet[nch[1]+1:end,:] = U0*pet[nch[1]+1:end,:]

        hP = nsolt.nChannels[1]

        L = fld(nch[2]*(nch[2]-1),2)
        gdudk = [ fill(zeros(dt,L), o) for o in ord ]
        for d = 1:D
            nShift = fld(size(pet,2), nBlocks[1])
            # pvx = permutedims(pvx)
            pvxe = MDCDL.permutedims(PolyphaseVector(pet, nBlocks))
            pet = pvxe.data
            pvy  = MDCDL.permutedims(PolyphaseVector(pyt, nBlocks))
            pyt, nBlocks = pvy.data, pvy.nBlocks
            # submatrices
            xeu = view(pet, 1:hP, :)
            xel = view(pet, (1:hP)+hP, :)
            xyu = view(pyt, 1:hP, :)
            xyl = view(pyt, (1:hP)+hP, :)
            for k = 1:nsolt.polyphaseOrder[d]
                teu, tel = (xeu + xel, xeu - xel) ./ sqrt(2)
                xeu .= teu; xel .= tel

                tyu, tyl = (xyu + xyl, xyu - xyl) ./ sqrt(2)
                xyu .= tyu; xyl .= tyl

                if isodd(k)
                    xel .= circshift(xel, (0, nShift))
                    xyl .= circshift(xyl, (0, nShift))
                else
                    xeu .= circshift(xeu, (0, -nShift))
                    xyu .= circshift(xyu, (0, -nShift))
                end

                teu, tel = (xeu + xel, xeu - xel) ./ sqrt(2)
                xeu .= teu; xel .= tel

                tyu, tyl = (xyu + xyl, xyu - xyl) ./ sqrt(2)
                xyu .= tyu; xyl .= tyl

                # txl = nsolt.propMatrices[d][k] * xl
                Uk = nsolt.propMatrices[d][k]
                xyl .= Uk * xyl

                angsu, musu = MDCDL.mat2rotations(nsolt.propMatrices[d][k])
                gdudk[d][k] = -MDCDL.scalarGradOfOrthonormalMatrix(xyl, xel, angsu, musu)
                # nsolt.propMatrices[d][k] .= MDCDL.rotations2mat(angsu - η*gdu, musu)

                xel .= Uk * xel
            end
        end


        grad .= vcat(gdw[nch[1]:end], gdu, vcat(vcat.(gdudk...)...))*vecnorm(x)^(-2)

        setAngleParameters!(nsolt, angsvm1, mus0)
        dist = x .- synthesize(nsolt, hy, size(x))
        cst = vecnorm(dist)^2

        println("f_$(cnt):\t cost = $(cst),\t |grad| = $(vecnorm(grad))")
        # sleep(0.2)
        cst
    end

    # opt = Opt(:LN_COBYLA, length(angs0s))
    lopt = Opt(:LD_MMA, length(angs0s))
    maxeval!(lopt,100)
    min_objective!(lopt, objfunc)
    (minf, minx, ret) = optimize(lopt, angs0s)

    ##### global optimization #####
    # gopt = Opt(:GD_MLSL, length(angs0s))
    # lower_bounds!(gopt, -1*pi*ones(size(angs0s)))
    # upper_bounds!(gopt,  1*pi*ones(size(angs0s)))
    # local_optimizer!(gopt, lopt)
    # min_objective!(gopt, objfunc)
    # (minf, minx, ret) = optimize(gopt, angs0s)
    #################

    minxt = vcat(zeros(dt, sum(nch)-1), minx);
    setAngleParameters!(nsolt, minxt, mus0)
    y = analyze(nsolt, x; outputMode=:vector)
    println("Epoch: $epoch, No.: $nd, cost = $(minf)")
end

atmimshow(nsolt)

if !isempty(filename)
    MDCDL.save(filename, nsolt)
end
