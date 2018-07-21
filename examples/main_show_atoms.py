import julia
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

j = julia.Julia()
j.using("MDCDL")
j.using("Base")
j.using("Images")
j.using("TestImages")

# TODO: 可能な限りPythonのネイティブコードに書き換える
nsolt = j.eval('MDCDL.load(joinpath(Pkg.dir(),"MDCDL","examples","design","sample.json"))')

# number of channels (2-tuple)
nChs = nsolt.nChannels
# decimation factor (D-tuple for D-dimensional RNSOLT)
df = nsolt.decimationFactor
# polyphase order (D-tuple for D-dimensional NSOLT)
ord = nsolt.polyphaseOrder

# generate analysis filter (atomic image) set of RNSOLT
analysisFilters = j.getAnalysisFilters(nsolt)

# show atomic images of RNSOLT
fig1, axes1 = plt.subplots(2, max(nChs));
for ps in range(0,nChs[0]):
    axes1[0,ps].imshow(analysisFilters[ps], cmap="Greys", vmin=-0.5, vmax=0.5)
for pa in range(0,nChs[1]):
    axes1[1,pa].imshow(analysisFilters[nChs[0]+pa], cmap="Greys", vmin=-0.5, vmax=0.5)
fig1

# number of wavelet tree level
lv = 3
fig2, axes2 = plt.subplots(sum(nChs), lv);

# create multi-scale RNSOLT object
msnsolt = j.Multiscale(nsolt, lv)
# 可能な限りPythonのネイティブコードに書き換える
x = j.eval('Array{Float64}(testimage("cameraman"))')

y = j.analyze(msnsolt, x)

for l in range(0,lv-1):
    for p in range(1,sum(nChs)):
        axes2[p,l].imshow(y[l][p-1], cmap="Greys", vmin=-0.5, vmax=0.5)

for p in range(0,sum(nChs)):
    axes2[p,lv-1].imshow(y[lv-1][p], cmap="Greys", vmin=-0.5, vmax=0.5)
fig2
rx = j.synthesize(msnsolt, y)

# evaluate reconstruction error
errx = np.linalg.norm(rx - x)
errx
