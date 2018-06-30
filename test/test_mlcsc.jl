using MDCDL

include("randomInit.jl")

# configurations
D = 3
nl = 2

szx = ntuple( d -> 64, D)

dt = Float64

mlcsc = MultiLayerCsc{Float64,D}(nl)

for l = 1:nl
    Dl = D + l - 1
    df = tuple(fill(2,D)..., ones(Integer,l-1)...)
    # df = tuple(fill(2,Dl)...)
    # df = (2,8)
    nch = prod(df) + 2
    # nch = (cld(prod(df),2) + 1, fld(prod(df),2) + 1)
    ord = ntuple( d -> 2, Dl)
    # ord = (4,2)
    # create a CNSOLT object
    cnsolt = Cnsolt(df, nch, ord, dataType=dt)
    randomInit!(cnsolt)

    mlcsc.dictionaries[l] = cnsolt
end

# x = rand(dt,szx)
x = zeros(dt,szx)
x[1] = 1

y = MDCDL.analyze(mlcsc, x)
rx = MDCDL.synthesize(mlcsc, y)

errx = vecnorm(rx - x)

println("error = $errx")
