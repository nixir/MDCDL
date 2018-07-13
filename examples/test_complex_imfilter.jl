using MDCDL
using ImageFiltering

D = 2
df = (3,2)
ch = 1
ord = (2,4)

pfb = ParallelFB(Complex{Float64},df,ch,ord)

afs = [ rand((Complex{Float64},df.*(ord.+1))...) for p in 1:ch ]
# afs = [ zeros(df.*(ord.+1)) ]
# afs[1][1,1] = 1

pfb.analysisFilters .= afs

# x = rand(fill(64,D)...)
x = Array{Complex{Float64}}(reshape(collect(1:(3*16)^D),fill(3*16,D)...))

yfb = analyze(pfb, x)[1]
# yim = circshift(imfilter(x, reflect(afs[1]), "circular"),-1.*(fld.(ord,2).+1))
typeof(x)
yim = downsample(imfilter(x, reflect(centered(conj(afs[1]))), "circular"),df, fld.(df,2))

vecnorm(yim-yfb)
