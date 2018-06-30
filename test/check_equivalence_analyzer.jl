using MDCDL

include("randomInit.jl")

D = 2
# df = ntuple(v -> 2, D) # must be even
df = (2,2)
# df = (2,2)
nch = prod(df) + 2
# nch = (3,3)
# ord = ntuple(v -> 4, D) # any element must be even number
ord = (2,2)

# create a CNSOLT object
cnsolt = Cnsolt(df, nch, ord)
randomInit!(cnsolt)

# atmimshow(cnsolt)

szx = df .* (ord .+ 1) .* 4
# x = rand(szx) + 1im*rand(szx)
x = reshape(convert.(Complex{Float64},collect(1:prod(szx))),szx...)

ya, sc = analyze(cnsolt,x)

typeof(ya)

afs = getAnalysisFilters(cnsolt)

offseta = df .- 1

yf = [ circshift(MDCDL.downsample(MDCDL.mdfilter(x,f),df,offseta),0) for f in afs ]

# err_y = [ ya[1][p,:] - vec.(yf)[p] for p in 1:nch ]

err_y = ya[1] .- yf

#########################3
println("norm(err_y) = $(sum(vecnorm.(err_y)))")

rxa = synthesize(cnsolt, ya, sc)

sfs = getSynthesisFilters(cnsolt)

offsetu = df .* 0
rxfm = sum([MDCDL.mdfilter( MDCDL.upsample(yf[p],df,offsetu),sfs[p]) for p in 1:sum(nch) ])
rxf = circshift(rxfm, -1 .* df .* ord)

err_rx = vecnorm(rxf - x)
println("norm(err_rx) = $(vecnorm(err_rx))")

println("process succeeded!")
