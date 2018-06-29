using Images, ImageView
using TestImages
using MDCDL
using Gtk.ShortNames

# function myshrink(x, lambda::Real)
#     max(1.0 - lambda/vecnorm(x), 0.0) * x
# end

df = (2,2)
nch = 6
ord = (2,2)
lv = 3
isRandomizedParams = true

orgImg = testimage("cameraman")[(1:256)+128,(1:256)+128]
cplxImg = complex.(Array{Float64}(orgImg))
sigma_awgn = 1e-2

crandn = (args...) -> (randn(args...) + 1im*randn(args...))/2
# cplxImg += sigma_awgn*randn(Complex{Float64},size(cplxImg)) # this code does not work in currently version (v0.6.3)
cplxImg += sigma_awgn*crandn(Float64,size(cplxImg))

cnsolt = Cnsolt(df,nch,ord)

if isRandomizedParams
    cnsolt.symmetry .= Diagonal(exp.(1im*rand(nch)))
    WW = eye(fld(nch,2))
    for d = 1:2
        map!(cnsolt.propMatrices[d], cnsolt.propMatrices[d]) do A
            Array(qr(rand(size(A)))[1])
        end

        @. cnsolt.paramAngles[d] = rand(size(cnsolt.paramAngles[d]))
        WW = foldl((x,y) -> y*x, cnsolt.propMatrices[d]) * WW
    end
    V0 = eye(nch)
    V0[2:end,2:end] = qr(rand(nch-1,nch-1))[1]
    WT = eye(nch)
    WT[1:fld(nch,2),1:fld(nch,2)] = WW'
    cnsolt.initMatrices[1] = WT * V0

end

y, sc = analyze(cnsolt, cplxImg, lv)

lambda = 3e-2
prox = (yin, eta) -> MDCDL.softshrink.(yin, lambda*eta)
gradOfLossFcn = (yin) -> - analyze(cnsolt, cplxImg - synthesize(cnsolt, yin, sc, lv), lv)[1]

(sy, err) = MDCDL.ista(gradOfLossFcn, prox, y, maxIterations=20, viewStatus=true)


resCplxImg = synthesize(cnsolt, sy, sc, lv)
resImgAbs = Array{Gray{N0f8}}(min.(vecnorm.(resCplxImg),1.0))
resImgAng = Array{Gray{N0f8}}(max.(min.(angle.(resCplxImg) ./ pi + 0.5, 1.0), 0.0))

imshow(resImgAbs)
