using MDCDL

include("randomInit.jl")

# configurations
D = 2
df = ntuple( d -> 2, D)
# df = (2,8)
nch = prod(df) + 2
# nch = (cld(prod(df),2)+1,fld(prod(df),2)+1)
ord = ntuple( d -> 2, D)
# ord = (4,2)
lv = 1

# szx = 3 .* ((df .^ lv) .* (ord .+ 1))
szx = tuple(fill(32,D)...)

dt = Float64

# create a CNSOLT object
cnsolt = Cnsolt(df, nch, ord, dataType=dt)
randomInit!(cnsolt)

println("CNSOLT Configurations: #Dimensions=$D, Decimation factor=$df, #Channels=$nch, Polyphase order=$ord, Tree levels=$lv")
# show atomic images
# atmimshow(cnsolt)

x = rand(dt,szx)

y, sc = MDCDL.analyze(cnsolt, x, lv)
rx = MDCDL.synthesize(cnsolt, y, sc, lv)

errx = vecnorm(rx - x)

println("error = $errx")

afs = getAnalysisFilters(cnsolt)
sfs = getSynthesisFilters(cnsolt)

sztx = df .* (ord .+ 1)
tx = rand(dt, sztx)

offsetd = df .- 1
offsetu = df .* 0

ys = [ MDCDL.downsample(MDCDL.cconv( afs[p],tx), df, offsetd) for p in 1:sum(nch)]

rtx = circshift(sum( MDCDL.cconv.(sfs,[ MDCDL.upsample(yyy, df, offsetu) for yyy in ys])), df)

recerrf = vecnorm(rtx - tx)
println("Reconstruction error by filtering: $recerrf")
