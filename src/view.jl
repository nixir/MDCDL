using Images, ImageView, Gtk.ShortNames

function atmimshow(cc::MDCDL.Cnsolt{2,S,T}; scale::Real = 1.0, offset::Real=0.5) where {S,T}
    P = cc.nChannels
    lowerBound=0.0
    upperBound=1.0


    afs = MDCDL.getAnalysisFilters(cc)

    afsViewRe, afsViewIm = map([real(afs), imag(afs)]) do afspart
        afsScaled = scale .* afspart .+ offset

        Array{Gray{N0f8}}[ max.(min.(f,upperBound),lowerBound) for f in afsScaled ]
    end
    # resImgAbs = Array{Gray{N0f8}}(min.(vecnorm.(resCplxImg),1.0))

    grid, frames, canvases = canvasgrid((2, P))
    for p = 1:P
        #TODO: imshow()の内部処理によって自動的にスケールが変更されてしまう．データの値をそのまま解釈して表示するように修正する
        # imshow(canvases[1,p], afsViewRe[p])
        # imshow(canvases[2,p], afsViewIm[p])
    end
    win = Window(grid)
    showall(win)
end
