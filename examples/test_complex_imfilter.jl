using MDCDL
using ImageFiltering
using Images
using OffsetArrays

D = 2
df = (2,2)
ch = 1
ord = (3,5)

szImg = tuple(fill(32,D)...)

typeKernel = Complex{Float64}
typeSignal = Complex{Float64}

pfb = ParallelFB(typeKernel,df,ch,ord)

# afs = [ rand(typeKernel,(df.*(ord.+1))...) for p in 1:ch ]
afs = [ zeros(typeKernel, df.*(ord.+1)) ]
afs[1][1,1] = 1

pfb.analysisFilters .= afs

x = if typeSignal <: ColorTypes.Colorant
	map(typeSignal,[ rand(eltype(typeSignal),szImg...) for cmp in 1:length(typeSignal) ]...)
else
	# rand(typeSignal,szImg...)
	Array{typeSignal}(reshape(collect(1:prod(szImg)),szImg...))
end

offset = df .- 1
ymd = circshift(MDCDL.downsample(mdfilter(x, afs[1];operation=:conv),df, offset), -1.*(fld.(ord,2)))

region = colon.(1,df.*(ord.+1)) .- df.*fld.(ord,2) .- 1
println(region)
yim = MDCDL.downsample(imfilter(x, reflect(OffsetArray(afs[1],region...)),"circular",ImageFiltering.FIR()),df, offset)
# yim = MDCDL.downsample(imfilter(x, reflect(centered(afs[1])),"circular",ImageFiltering.FIR()),df, fld.(df,2))


erry = vecnorm(yim-ymd)
println(erry)
