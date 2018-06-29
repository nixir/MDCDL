using MDCDL

include("randomInit.jl")

# configurations
D = 2
nl = 3

szx = ntuple( d -> 32, D)

dt = Float64

mlcsc = MultiLayerCsc{Float64,D}(nl)

for l = 1:nl
    Dl = D + l - 1
    df = tuple(fill(2,D)..., ones(Integer,l-1)...)
    # df = (2,8)
    nch = prod(df) + 2
    # nch = (cld(prod(df),2) + 1, fld(prod(df),2) + 1)
    ord = ntuple( d -> 0, Dl)
    # ord = (4,2)
    # create a CNSOLT object
    cnsolt = Cnsolt(df, nch, ord, dataType=dt)
    randomInit!(cnsolt)

    mlcsc.dictionaries[l] = cnsolt
end

x = rand(dt,szx)

y = MDCDL.analyze(mlcsc, x)
rx = MDCDL.synthesize(mlcsc, y)

errx = vecnorm(rx - x)

println("error = $errx")
