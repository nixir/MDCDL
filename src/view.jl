using Images, ImageView, Gtk.ShortNames

function atmimshow(cc::MDCDL.Cnsolt{2,S,T}, clim = CLim(-0.5,0.5)) where {S,T}
    P = cc.nChannels

    afs = MDCDL.getAnalysisFilters(cc)

    afsRe, afsIm = reim(afs)
    sigclim = ImageView.Signal(clim)

    grid, frames, canvases = canvasgrid((2, P))
    for p = 1:P
        zrr, sdr = roi(afsRe[p], ImageView.default_axes(afsRe[p]))
        zri, sdi = roi(afsIm[p], ImageView.default_axes(afsIm[p]))
        imshow(frames[1,p], canvases[1,p], afsRe[p], sigclim, zrr, sdr)
        imshow(frames[2,p], canvases[2,p], afsIm[p], sigclim, zri, sdi)
    end
    win = Window(grid)
    showall(win)
end

function atmimshow(cc::MDCDL.Rnsolt{2,S,T}, clim = CLim(-0.5,0.5)) where {S,T}
    P = sum(cc.nChannels)

    afs = MDCDL.getAnalysisFilters(cc)

    sigclim = ImageView.Signal(clim)

    grid, frames, canvases = canvasgrid((2, P))
    for p = 1:P
        zr, sd = roi(afs[p], ImageView.default_axes(afs[p]))
        imshow(frames[1,p], canvases[1,p], afs[p], sigclim, zr, sd)
    end
    win = Window(grid)
    showall(win)
end

# function atmimshow(mlcsc::MDCDL.MultiLayerCsc, args...)
#     for l = mlcsc.nLayers
#         atmimshow(mlcsc.dictionaries[l], args...)
#     end
# end
