using MDCDL

include("randomInit.jl")

D = 2
# df = ntuple(v -> 2, D) # must be even
# df = tuple(rand(1:4,D)...)
df = (2,3)
# df = (2,2)
# nch = prod(df)
nch = (cld(prod(df),2), fld(prod(df),2)) .+ (1,4)
# nch = (3,3)
# ord = ntuple(v -> 4, D) # any element must be even number
ord = tuple(rand(0:2:4,D)...)

# create a NSOLT object
nsolt = Rnsolt(df, nch, ord)
randomInit!(nsolt)

# atmimshow(nsolt)

szx = df .* (ord .+ 1) .* 4
# x = rand(szx) + 1im*rand(szx)
x = reshape(convert.(Complex{Float64},collect(1:prod(szx))),szx...)

ya, sc = analyze(nsolt,x)

typeof(ya)

afs = getAnalysisFilters(nsolt)

offseta = df .- 1

yf = [ circshift(MDCDL.downsample(MDCDL.mdfilter(x, f; operation=:conv), df, offseta), 0) for f in afs ]

# err_y = [ ya[1][p,:] - vec.(yf)[p] for p in 1:nch ]

err_y = ya[1] .- yf

#########################3
println("norm(err_y) = $(sum(vecnorm.(err_y)))")

rxa = synthesize(nsolt, ya, sc)

sfs = getSynthesisFilters(nsolt)

rxfm = sum([MDCDL.mdfilter( MDCDL.upsample(yf[p], df), sfs[p]; operation=:conv) for p in 1:sum(nch) ])
rxf = circshift(rxfm, -1 .* df .* ord)

err_rx = vecnorm(rxf - x)
println("norm(err_rx) = $(vecnorm(err_rx))")

println("process succeeded!")
