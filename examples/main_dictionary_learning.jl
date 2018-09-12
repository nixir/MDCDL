using MDCDL
using LinearAlgebra
using TestImages

Nsolt = Rnsolt
dt = Float64
df = (2,2)
ord = (4,4)
nch = 8

szx = (32,32)
nSubData = 32

orgImg = Array{dt}(testimage("cameraman"))
trainingIds = map(1:nSubData) do nsd
    pos = rand.(UnitRange.(0,size(orgImg) .- szx))
    UnitRange.(1 .+ pos, szx .+ pos)
end

trainingSet = map(idx -> orgImg[idx...], trainingIds)

nsolt = Nsolt(df, ord, nch)
MDCDL.rand!(nsolt, isPropMat = false, isSymmetry = false)

sc_options = ( iterations = 400, sparsity = 0.4, filter_domain=:convolution)
du_options = ( iterations = 1, stepsize = 1e-3,)

options = ( epochs  = 100,
            verbose = :standard,
            sc_options = sc_options,
            du_options = du_options,)

MDCDL.train!(nsolt, trainingSet; options...)
