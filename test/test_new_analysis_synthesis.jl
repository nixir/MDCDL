include("NSOLT/MDCDL.jl")
using MDCDL

# configurations
D = 2
df = ntuple( d -> 2, D)
# df = (2,8)
nch = prod(df) + 2
ord = ntuple( d -> 4, D)
# ord = (4,2)
lv = 1

szx = df .* (ord .+ 1)

dt = Float64

# create a CNSOLT object
cnsolt = Cnsolt(df, nch, ord, dataType=dt)

# initialize parameter matrices
cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
cnsolt.initMatrices[1] = Array{dt}(qr(rand(nch,nch), thin=false)[1])

for d = 1:D
    map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
        Array(qr(rand(dt,size(A)), thin=false)[1])
    end

    @. cnsolt.paramAngles[d] = rand(dt,size(cnsolt.paramAngles[d]))
end

println("CNSOLT Configurations: #Dimensions=$D, Decimation factor=$df, #Channels=$nch, Polyphase order=$ord, Tree levels=$lv")
# show atomic images
# atmimshow(cnsolt)

x = rand(Complex{dt},szx)


y, sc = MDCDL.analyze(cnsolt, x, lv)
rx = MDCDL.synthesize(cnsolt, y, sc, lv)

errx = vecnorm(rx - x)

println("error = $errx")

afs = getAnalysisFilters(cnsolt)
sfs = getSynthesisFilters(cnsolt)

sztx = df .* (ord .+ 1)
tx = rand(Complex{dt}, sztx)

offsetd = df .- 1
offsetu = df .* 0

ys = [ MDCDL.downsample(MDCDL.cconv( afs[p],tx), df, offsetd) for p in 1:nch]

rtx = circshift(sum( MDCDL.cconv.(sfs,[ MDCDL.upsample(yyy, df, offsetu) for yyy in ys])), df)

recerrf = vecnorm(rtx - tx)
println("Reconstruction error by filtering: $recerrf")
