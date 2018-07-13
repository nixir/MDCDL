using MDCDL
using Images, ImageView, TestImages
using ImageFiltering

### configurations ###
dirNsolt = joinpath(Pkg.dir(),"MDCDL","examples","design","sample.json")
σ_noise = 1e-3
λ = 1e-6
lv = 2 # tree level of NSOLT
######################

### setup observed image
orgImg = testimage("lena")
orgImgFloat = Array{ColorTypes.RGB{Float64}}(orgImg)
u = orgImg

# AWGN
w = mapc.( v -> σ_noise * randn(), u)

# blur kernel
h = centered([ 0 1 0; 1 -4 1; 0 1 0 ])

# observed image
x = imfilter(u,h,"circular") + w

### setup NSOLT
nsolt = MDCDL.load(dirNsolt)
println("NSOLT configurations:")
println(" - Type = $(typeof(nsolt))")
println(" - Decimation Factor = $(nsolt.decimationFactor)")
println(" - Number Of Channels = $(nsolt.nChannels)")
println(" - Polyphase Order = $(nsolt.polyphaseOrder)")

# analysis/synthesis filter set
pfb = ParallelFB(nsolt)

y = analyze(pfb, x, lv)

rx = synthesize(pfb, y, lv)

errx = vecnorm(rx - x)

println("error: $errx")
