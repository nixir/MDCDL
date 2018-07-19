# import ImageView, Gtk

function atmimshow(cc::MDCDL.Cnsolt{T,2,S}, clim = ImageView.CLim(-0.5,0.5)) where {S,T}
    P = cc.nChannels

    afs = MDCDL.getAnalysisFilters(cc)

    afsRe, afsIm = reim(afs)
    sigclim = ImageView.Signal(clim)

    grid, frames, canvases = ImageView.canvasgrid((2, P))
    for p = 1:P
        zrr, sdr = ImageView.roi(afsRe[p], ImageView.default_axes(afsRe[p]))
        zri, sdi = ImageView.roi(afsIm[p], ImageView.default_axes(afsIm[p]))
        ImageView.imshow(frames[1,p], canvases[1,p], afsRe[p], sigclim, zrr, sdr)
        ImageView.imshow(frames[2,p], canvases[2,p], afsIm[p], sigclim, zri, sdi)
    end
    win = Gtk.GtkWindow(grid)
    ImageView.showall(win)
end

function atmimshow(cc::MDCDL.Rnsolt{T,2,S}, clim = ImageView.CLim(-0.5,0.5)) where {S,T}
    P = sum(cc.nChannels)

    afs = MDCDL.getAnalysisFilters(cc)

    sigclim = ImageView.Signal(clim)

    grid, frames, canvases = ImageView.canvasgrid((2, P))
    for p = 1:P
        zr, sd = ImageView.roi(afs[p], ImageView.default_axes(afs[p]))
        ImageView.imshow(frames[1,p], canvases[1,p], afs[p], sigclim, zr, sd)
    end
    win = Gtk.GtkWindow(grid)
    ImageView.showall(win)
end

# function atmimshow(mlcsc::MDCDL.MultiLayerCsc, args...)
#     for l = mlcsc.nLayers
#         atmimshow(mlcsc.dictionaries[l], args...)
#     end
# end
