using MDCDL
using Plots

function test_analysis_speed(ts::TransformSystem, dataset::Vector)
    N = length(dataset)
    val, t, bytes, gctime, memallocs = @timed for n = 1:N
        analyze(ts, dataset[n])
    end
    return t
end

# image size
szxs = [ (8,8), (16,16) ,(32,32), (64,64), (128,128), (256,256), (512,512) ]
lnszxs = length(szxs)

lnx = 100       # Number of elements of dataset

df = (2,2)      # Decimation factor
ord = (2,2)     # Polyphase order
nch = 8         # Number of channels

nsolt = Rnsolt(df, ord, nch)    # create instance of Real-NSOLT (RNSOLT)
rand!(nsolt)    # randomize RNSOLT angle parameteres

tms = Array{Float64}(undef, lnszxs) # elapsed times
for idx = 1:lnszxs
    transform = createTransform(nsolt, Shapes.Vec(szxs[idx]...))
    xs = [ rand(szxs[idx]...) for n = 1:lnx ]   # generate dataset
    tms[idx] = test_analysis_speed(transform, xs)
end
