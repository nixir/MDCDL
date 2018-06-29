include("NSOLT/MDCDL.jl")

using MDCDL

D = 1
# df = ntuple(v -> 2, D) # must be even
df = (2,)
# df = (2,2)
nch = prod(df) + 2
# ord = ntuple(v -> 4, D) # any element must be even number
ord = (2,)

# create a CNSOLT object
cnsolt = Cnsolt(df, nch, ord)

# set random values to CNSOLT's parameters
if false
    # initialize parameter matrices
    cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
    cnsolt.initMatrices[1] = Array(qr(rand(nch,nch))[1])

    for d = 1:D
        map!(cnsolt.PropMatrices[d], cnsolt.PropMatrices[d]) do A
            Array(qr(rand(size(A)))[1])
        end

        @. cnsolt.paramAngles[d] = rand(size(cnsolt.paramAngles[d]))
    end
end

# atmimshow(cnsolt)

szx = df .* (ord .+ 1) .* 4
# x = rand(szx) + 1im*rand(szx)
x = reshape(convert.(Complex{Float64},collect(1:prod(szx))),szx...)

ya, sc = analyze(cnsolt,x)

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
rxfm = sum([MDCDL.mdfilter( MDCDL.upsample(yf[p],df,offsetu),sfs[p]) for p in 1:nch ])
rxf = circshift(rxfm, -1 .* df .* ord)

err_rx = vecnorm(rxf - x)
println("norm(err_rx) = $(vecnorm(err_rx))")

println("process succeeded!")
