using MDCDL
using Images, ImageView, TestImages
using ImageFiltering

### configurations ###
dirNsolt = joinpath(Pkg.dir(),"MDCDL","examples","design","sample.json")
σ_noise = 3e-2
λ = 1e-2
lv = 1 # tree level of NSOLT
######################

### setup observed image
orgImg = testimage("cameraman")
u = Array{Float64}(imresize(orgImg,(256,256)))

# AWGN
w = σ_noise * randn(size(u))

# Degradation (pixel loss)
P = map((v)-> v > 0.1 ? 1 : 0, rand(size(u)))

# Observation
x = P.*u + w

### setup NSOLT
nsolt = MDCDL.load(dirNsolt)
println("NSOLT configurations:")
println(" - Type = $(typeof(nsolt))")
println(" - Decimation Factor = $(nsolt.decimationFactor)")
println(" - Number of Channels = $(nsolt.nChannels)")
println(" - Polyphase Order = $(nsolt.polyphaseOrder)")

# analysis/synthesis filter set
pfb = ParallelFB(nsolt)

y0 = analyze(pfb, x, lv)

gradOfLossFcn = (ty) -> begin
    - analyze(pfb, P.*(x - synthesize(pfb, ty, lv)), lv)
end
proxFcn = (ty, η) -> MDCDL.softshrink(ty, λ*η)
viewFcn = (itrs, ty, err) -> begin
    println("# Iterations=$itrs, error=$err")
end

hy = MDCDL.fista(gradOfLossFcn, proxFcn, 1.0, y0; maxIterations=200, viewFunction=viewFcn, absTol=eps())

ru = Array{Gray{Float64}}(synthesize(pfb, hy, lv))

errx = vecnorm(ru - u)

println("error: $errx")
