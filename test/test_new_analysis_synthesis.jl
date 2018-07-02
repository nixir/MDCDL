using MDCDL

include("randomInit.jl")

# configurations
D = 2
# df = tuple(rand(1:4,D)...)
df = tuple(fill(3,D)...)
nch = (cld(prod(df),2), fld(prod(df),2)) .+ (1, 2)
# nch = prod(df) + 2 + (prod(df) % 2)
# nch = (cld(prod(df),2)+1,cld(prod(df),2)+1)
# ord = tuple(rand(0:2:4,D)...)
ord = (0,0)
lv = 2

szx = 4 .* ((df .^ lv) .* (ord .+ 1))

dt = Float64

# create a nsolt object
nsolt = Rnsolt(df, nch, ord, dataType=dt)
# randomInit!(nsolt)

println("nsolt Configurations: #Dimensions=$D, Decimation factor=$df, #Channels=$nch, Polyphase order=$ord, Tree levels=$lv")
# show atomic images
# atmimshow(nsolt)

# x = rand(dt,szx)
x = reshape(collect(1:prod(szx)),szx...)

y, sc = MDCDL.analyze(nsolt, x, lv)
rx = MDCDL.synthesize(nsolt, y, sc, lv)

errx = vecnorm(rx - x)

println("error = $errx")

afs = getAnalysisFilters(nsolt)
sfs = getSynthesisFilters(nsolt)

offsetd = df .- 1
nShifts = -1 .* ord .* df

ys = [ MDCDL.downsample(MDCDL.mdfilter(x, afs[p]; boundary=:circular, operation=:conv), df, offsetd) for p in 1:sum(nch) ]

rtx = circshift(sum( MDCDL.mdfilter.([ MDCDL.upsample(yyy, df) for yyy in ys], sfs; boundary=:circular, operation=:conv)), nShifts)

recerrf = vecnorm(rtx - x)
println("Reconstruction error by filtering: $recerrf")
