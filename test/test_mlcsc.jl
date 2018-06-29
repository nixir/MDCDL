using MDCDL

# configurations
D = 2
nl = 3

szx = ntuple( d -> 32, D)

dt = Float64

mlcsc = MultiLayerCsc{Float64,D}(nl)

for l = 1:nl
    Dl = D + l - 1
    df = ntuple( d -> 2, Dl)
    # df = (2,8)
    # nch = prod(df) + 2
    nch = prod(df)
    ord = ntuple( d -> 0, Dl)
    # ord = (4,2)
    # create a CNSOLT object
    cnsolt = Cnsolt(df, nch, ord, dataType=dt)

    # initialize parameter matrices
    # cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
    # cnsolt.initMatrices[1] = Array{dt}(qr(rand(nch,nch), thin=false)[1])
    #
    # for d = 1:Dl
    #     map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
    #         Array(qr(rand(dt,size(A)), thin=false)[1])
    #     end
    #
    #     @. cnsolt.paramAngles[d] = rand(dt,size(cnsolt.paramAngles[d]))
    # end

    mlcsc.dictionaries[l] = cnsolt
end

x = MDCDL.crand(dt,szx)

y = MDCDL.analyze(mlcsc, x)
rx = MDCDL.synthesize(mlcsc, y)

errx = vecnorm(rx - x)

println("error = $errx")
