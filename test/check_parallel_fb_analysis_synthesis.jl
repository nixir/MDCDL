using MDCDL

D = 3
df = ntuple(v -> 2, D) # must be even
nch = prod(df) + 2
ord = ntuple(v -> 2, D) # any element must be even number
# ord = (2,)

# create a CNSOLT object
cnsolt = Cnsolt(df, nch, ord)

# set random values to CNSOLT's parameters
if true
    # initialize parameter matrices
    cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
    cnsolt.initMatrices[1] = Array(qr(rand(nch,nch))[1])

    for d = 1:D
        map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
            Array(qr(rand(size(A)))[1])
        end

        @. cnsolt.paramAngles[d] = rand(size(cnsolt.paramAngles[d]))
    end
end

# atmimshow(cnsolt)

pfb = ParallelFB(df, nch, ord, dataType = Complex{Float64})
pfb.analysisFilters .= getAnalysisFilters(cnsolt)
pfb.synthesisFilters .= getSynthesisFilters(cnsolt)

########################

szx = df .* (ord .+ 1) .* 4
# x = rand(szx) + 1im*rand(szx)
x = reshape(convert.(Complex{Float64},collect(1:prod(szx))),szx...)

ya, sc = analyze(cnsolt,x)
yf, scp = analyze(pfb, x)

err_y = ya[1] .- yf[1]

#########################3
println("norm(err_y) = $(sum(vecnorm.(err_y)))")

rxa = synthesize(cnsolt, ya, sc)
rxf = synthesize(pfb, yf, scp)

err_rx = vecnorm(rxf - x)
println("norm(err_rx) = $(vecnorm(err_rx))")

println("process succeeded!")
