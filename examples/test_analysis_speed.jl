using MDCDL
using LinearAlgebra
using Plots

function test_analysis_speed(ts::TransformSystem, dataset::Vector)
    N = length(dataset)
    val, t, bytes, gctime, memallocs = @timed for n = 1:N
        analyze(ts, dataset[n])
    end
    return t
end

szxs = [ (8,8), (16,16) ,(32,32), (64,64), (128,128), (256,256), (512,512) ]
lnszxs = length(szxs)

lnx = 100

df = (2,2)
ord = (2,2)
nch = 8

nsolt = Rnsolt(df, ord, nch)
rand!(nsolt)

tms = Array{Float64}(undef, lnszxs)
for idx = 1:lnszxs
    transform = createTransform(nsolt, Shapes.Vec(szxs[idx]...))
    xs = [ rand(szxs[idx]...) for n = 1:lnx ]
    tms[idx] = test_analysis_speed(transform, xs)
end

plotopts = (    xticks = (collect(1:lnszxs), string.(szxs)),
                xlabel = "Image size [pixels]",
                ylabel = "Real time [s]",
                yscale = :log10,
                legend = :topleft,
                label = "MDCDL (Julia)", )
plot(tms; plotopts...)
