

########################################

# include("OrthonormalMatrixSystem.jl")

using MDCDL
# using OrthonormalMatrixSystem

dimension = 2
x = randn(ntuple(t -> 64, dimension)...)
typeof(size(x))

A = OrthonormalMatrixSystem.rotations2mat(pi*rand(3),ones(3))

dec = ntuple(x -> 2, dimension) # Decimation factor
nch = prod(dec) # Number of Channels
po = ntuple(x -> 0, dimension) # Polyphase order
lv = 4 # Number of levels

cc = MDCDL.Cnsolt(dec,nch,po)

# y,sc = MDCDL.myanalyze(cc,x,lv)
# dx = MDCDL.mysynthesize(cc,y,sc,lv)
#
# vecnorm(dx - x)^2 # check PR property

ty,tsc = MDCDL.analyze(cc,x,lv)
tx = MDCDL.synthesize(cc,ty,tsc,lv)

vecnorm(tx - x)^2 # check PR property

workspace()
