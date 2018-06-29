using MDCDL

# configurations
D = 3
df = ntuple( d -> 2, D)
nch = prod(df) + 2
ord = ntuple( d -> 2, D) # ord[d] > 0 for all d

dt = Float64

# create a CNSOLT object
cnsoltSrc = Cnsolt(df, nch, ord, dataType=dt)
cnsoltDst = Cnsolt(df, nch, ord, dataType=dt)

# initialize parameter matrices
cnsoltSrc.symmetry .= Diagonal(exp.(1im*rand(nch)))
cnsoltSrc.initMatrices[1] = Array{dt}(qr(rand(nch,nch),thin=false)[1])

for d = 1:D
    map!(cnsoltSrc.propMatrices[d], cnsoltSrc.propMatrices[d]) do A
        Array(qr(rand(dt,size(A)),thin=false)[1])
    end

    @. cnsoltSrc.paramAngles[d] = 6*rand(dt,size(cnsoltSrc.paramAngles[d]))
end

println("CNSOLT Configurations: #Dimensions=$D, Decimation factor=$df, #Channels=$nch, Polyphase order=$ord, Tree levels=1")
# show atomic images
# atmimshow(cnsolt)

angs, mus = getAngleParameters(cnsoltSrc)
setAngleParameters!(cnsoltDst, angs, mus)

errInitMat = vecnorm(cnsoltSrc.initMatrices[1] - cnsoltDst.initMatrices[1])
errPropMat = sum(vecnorm.(cnsoltSrc.propMatrices .- cnsoltDst.propMatrices))
errAngs = sum(vecnorm.(cnsoltSrc.paramAngles .- cnsoltDst.paramAngles))

println("err. initial matrices: $errInitMat")
println("err. prop matrices: $errPropMat")
println("err. angles: $errAngs")
