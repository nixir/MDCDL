using MDCDL
using Images, TestImages
using ImageFiltering
using Plots

### configurations ###
dirNsolt = joinpath(Pkg.dir(),"MDCDL","examples","design","sample.json")
σ_noise = 1e-2 # variance of AWGN
λ = 1e-2 # FISTA parameter
lv = 3 # tree level of NSOLT
######################

### setup observed image
# orgImg = testimage("cameraman")
# u = Array{Float64}(imresize(orgImg,(256,256)))
orgImg = testimage("lena")
u = Array{RGB{Float64}}(orgImg)

# AWGN
w = mapc.((v)->σ_noise * randn(), u)

# Degradation (pixel loss)
P = map((v)-> v > 0.01 ? 1 : 0, rand(size(u)))

# Observation
x = P.*u + w

### setup NSOLT
nsolt = MDCDL.load(dirNsolt)
println("NSOLT configurations:")
println(" - Type = $(typeof(nsolt))")
println(" - Decimation Factor = $(nsolt.decimationFactor)")
println(" - Number of Channels = $(nsolt.nChannels)")
println(" - Polyphase Order = $(nsolt.polyphaseOrder)")
### show analysis filters
atmimshow(nsolt)

# setup Multiscale NSOLT
mlpfb = Multiscale(ParallelFB(nsolt), lv)

# setup FISTA
y0 = analyze(mlpfb, x; shape = :vector)

gradOfLossFcn = (ty) -> begin
    - analyze(mlpfb, P.*(x - synthesize(mlpfb, ty, size(x))); shape = :vector)
end
proxFcn = (ty, η) -> MDCDL.softshrink(ty, λ*η)
viewFcn = (itrs, ty, err) -> begin
    println("# Iterations=$itrs, error=$err")
end

hy = MDCDL.fista(gradOfLossFcn, proxFcn, 1.0, y0; maxIterations=400, viewFunction=viewFcn, absTol=eps())

# restored image
ru = synthesize(mlpfb, hy, size(x))

psnr = (a, ref) -> 10*log10(1/(vecnorm(a-ref)^2/length(a)))
errx = vecnorm(ru - u)

println("error: $errx")

plotOrg = plot(u ; xlabel=string("Original Image"), aspect_ratio=:equal)
plotObs = plot(x ; xlabel=string("Observed Image\nPSNR=",trunc(psnr(x,u),3)), aspect_ratio=:equal)
plotRes = plot(ru; xlabel=string("Restored Image\nPSNR=",trunc(psnr(ru,u),3)), aspect_ratio=:equal)
plot(plotOrg, plotObs, plotRes; aspect_ratio=:equal, layout=(1,3))
# atmimshow(msnsolt.filterBank)
# imshow(ru)
