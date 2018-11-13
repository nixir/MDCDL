using TiledIteration: TileIterator
using FFTW: fft, dct

# upsampler
function upsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D} = (fill(0,D)...,)) where {T,D}
    szx = size(x)
    mtx = zeros(T, szx .* factor)
    for ci in CartesianIndices(szx)
        mtx[((ci.I .- 1) .* factor .+ 1 .+ offset)...] = x[ci]
    end
    mtx
end

function downsample(x::AbstractArray{T,D}, factor::NTuple{D}, offset::NTuple{D}=(fill(0,D)...,)) where {T,D}
    map(CartesianIndices(fld.(size(x), factor))) do ci
        x[((ci.I .- 1) .* factor .+ 1 .+ offset)...]
    end
end

representationmatrix(f, sz::NTuple) = representationmatrix(f, sz...)
function representationmatrix(f::Function, sz::Integer...)
    hcat([ setindex!(zeros(sz), 1, idx) |> f |> vec for idx in 1:prod(sz) ]...)
end

Base.@pure function haarbasis(d::Integer)
    w = walsh(d)
    [ reshape(@view(w[p,:]), fill(2,d)...) |> Array for p in 1:size(w, 1)]
end

Base.@pure function walsh(n::Integer)
    # ifelse(n >= 0, sub_walsh(Val(n)), error("n must to be a positive")) # this code is not work correctly.
    if n >= 0; sub_walsh(Val(n)) else error("n must to be a positive") end
end

function sub_walsh(::Val{N}) where {N}
    w = sub_walsh(Val(N-1))
    return [ w w ; w -w ]
end
sub_walsh(::Val{0}) = 1

# matrix-formed CDFT operator for D-dimensional signal
cdftmtx(sz::NTuple) = cdftmtx(sz...)
cdftmtx(sz::Integer...) = cdftmtx(Float64, sz...)
cdftmtx(T::Type, sz::NTuple) = cdftmtx(T, sz...)
cdftmtx(::Type, sz::Integer...) = cdftmtx(Float64, sz...)

cdftmtx(::Type{Complex{T}}, sz...) where {T} = cdftmtx(T, sz...)

Base.@pure function cdftmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    len = prod(sz)

    mtx = representationmatrix(x->fft(T.(x)), sz)
    rm = Diagonal(Complex{T}[ exp(-1im*angle(mtx[n,end])/2) for n in 1:len ])

    complex(T).(rm * mtx / sqrt(len))
end

permdctmtx(sz::NTuple) = permdctmtx(sz...)
permdctmtx(sz::Integer...) = permdctmtx(Float64, sz...)
permdctmtx(T::Type, sz::NTuple) = permdctmtx(T, sz...)

permdctmtx(::Type, sz::Integer...) = permdctmtx(Float64, sz...)

Base.@pure function permdctmtx(::Type{T}, sz::Integer...) where T<:AbstractFloat
    mtx = representationmatrix(x->dct(T.(x)), sz)

    isevenids = map(ci->iseven(sum(ci.I .- 1)), CartesianIndices(sz)) |> vec
    permids = sortperm(isevenids; rev=true, alg=Base.DEFAULT_STABLE)

    vcat([ transpose(@view mtx[pi,:]) for pi in permids ]...)
end

function getMatrixB(P::Integer, angs::AbstractVector{T}) where T
    hP = fld(P,2)
    psangs = (2 .* angs .+ pi) ./ 4
    cs = cos.(psangs)
    ss = sin.(psangs)

    LC = [[ (-1im*cs[n]) (-1im*ss[n]); (cs[n]) (-ss[n]) ] for n in 1:fld(hP,2) ]
    LS = [[ (ss[n]) (cs[n]); (1im*ss[n]) (-1im*cs[n]) ] for n in 1:fld(hP,2) ]

    pbm = ones(fill(hP % 2,2)...)
    C = cat(LC..., pbm; dims=[1,2])
    S = cat(LS..., 1im*pbm; dims=[1,2])

    [ C conj(C); S conj(S) ] / sqrt(convert(T,2))
end

function analysisbank(cc::CnsoltTypeI{T,D}) where {D,T}
    df = cc.decimationFactor
    P = cc.nChannels
    M = prod(df)
    ord = orders(cc)

    # output
    ppm = zeros(complex(T), P, prod(df .* (ord .+ 1)))
    ppm[1:M,1:M] = cc.FJ

    # Initial matrix process
    ppm = cc.V0 * ppm

    nStrides = [1, cumprod(collect(ord[1:end-1] .+ 1))... ] .* M
    for d = 1:D
        # angs = cc.paramAngles[d]
        # propMats = cc.propMatrices[d]
        ppmup = @view ppm[1:fld(P,2),:]
        ppmlw = @view ppm[fld(P,2)+1:P,:]
        for k = 1:ord[d]
            B = getMatrixB(P, cc.θdks[d][k])

            # B Λ(z_d) B'
            ppm .= B' * ppm
            ppmlw .= circshift(ppmlw, (0, nStrides[d]))
            ppm .= B * ppm

            ppmup .= cc.Wdks[d][k] * ppmup
            ppmlw .= cc.Udks[d][k] * ppmlw
        end
    end
    cc.Φ * ppm
end

function analysisbank(cc::CnsoltTypeII{T,D}) where {D,T}
    df = cc.decimationFactor
    P = cc.nChannels
    M = prod(df)
    ord = orders(cc)
    nStages = cc.nStages

    # output
    ppm = zeros(complex(T), P, prod(df .* (ord .+ 1)))
    ppm[1:M,1:M] = cc.FJ

    # Initial matrix process
    ppm = cc.V0 * ppm

    nStrides = [1, cumprod(collect(ord[1:end-1] .+ 1))... ] .* M
    for d = 1:D
        ppmev = @view ppm[1:P-1,:]
        ppmup1 = @view ppm[1:fld(P,2),:]
        ppmlw1 = @view ppm[fld(P,2)+1:P-1,:]
        ppmup2 = @view ppm[1:cld(P,2),:]
        ppmlw2 = @view ppm[cld(P,2):P,:]
        for k = 1:nStages[d]
            # first step
            B = getMatrixB(P, cc.θ1dks[d][k])

            # B Λ(z_d) B'
            ppmev .= B' * ppmev
            ppmlw1 .= circshift(ppmlw1, (0, nStrides[d]))
            ppmev .= B * ppmev

            ppmup1 .= cc.Wdks[d][k] * ppmup1
            ppmlw1 .= cc.Udks[d][k] * ppmlw1

            # second step

            B = getMatrixB(P, cc.θ2dks[d][k])

            # B Λ(z_d) B'
            ppmev .= B' * ppmev
            ppmlw2 .= circshift(ppmlw2, (0, nStrides[d]))
            ppmev .= B * ppmev

            ppmlw2 .= cc.Ûdks[d][k] * ppmlw2
            ppmup2 .= cc.Ŵdks[d][k] * ppmup2
        end
    end
    cc.Φ * ppm
end

function analysisbank(rc::RnsoltTypeI{T,D}) where {D,T}
    df = rc.decimationFactor
    nch = rc.nChannels
    P = sum(nch)
    M = prod(df)
    ord = orders(rc)

    rngUpper = (1:nch[1], :)
    rngLower = (nch[1]+1:P, :)

    # output
    ppm = zeros(T, P, prod(df .* (ord .+ 1)))
    ppm[1:cld(M,2), 1:M] = @view rc.CJ[1:cld(M,2),:]
    ppm[nch[1].+(1:fld(M,2)), 1:M] = @view rc.CJ[cld(M,2)+1:end,:]

    # Initial matrix process
    ppm[rngUpper...] = rc.W0 * @view ppm[rngUpper...]
    ppm[rngLower...] = rc.U0 * @view ppm[rngLower...]

    nStrides = [1, cumprod(collect(ord[1:end-1] .+ 1))... ] .* M
    for d = 1:D
        ppmup = @view ppm[1:nch[1],:]
        ppmlw = @view ppm[nch[1]+1:P,:]
        for k = 1:ord[d]
            # B Λ(z_d) B'
            butterfly!(ppm, nch[1])
            ppmlw .= circshift(ppmlw, (0, nStrides[d]))
            butterfly!(ppm, nch[1])

            ppmlw .= rc.Udks[d][k] * ppmlw
        end
    end
    ppm
end

function analysisbank(rc::RnsoltTypeII{T,D}) where {D,T}
    df = rc.decimationFactor
    M = prod(df)
    ord = orders(rc)
    nStages = rc.nStages
    nch = rc.nChannels
    P = sum(nch)
    # maxP, minP, chMajor, chMinor = if rc.nChannels[1] > rc.nChannels[2]
    #     (rc.nChannels[1], rc.nChannels[2], 1:rc.nChannels[1], (rc.nChannels[1]+1):P)
    # else
    #     (rc.nChannels[2], rc.nChannels[1], (rc.nChannels[1]+1):P, 1:rc.nChannels[1])
    # end
    minP, maxP = minmax(rc.nChannels)

    # output
    ppm = zeros(T, P, prod(df .* (ord .+ 1)))
    # ppm[1:M,1:M] = rc.matrixF
    ppm[1:cld(M,2), 1:M] = @view rc.CJ[1:cld(M,2),:]
    ppm[nch[1] .+ (1:fld(M,2)), 1:M] = @view rc.CJ[cld(M,2)+1:end,:]

    # Initial matrix process
    ppmiu = @view ppm[1:nch[1],:]
    ppmil = @view ppm[(nch[1]+1):end,:]
    ppmiu .= rc.W0 * ppmiu
    ppmil .= rc.U0 * ppmil

    nStrides = [1, cumprod(collect(ord[1:end-1] .+ 1))... ] .* M
    for d = 1:D
        ppmlw1 = @view ppm[(minP+1):end,:]
        ppmmn  = @view ppm[chMinor,:]
        ppmlw2 = @view ppm[(maxP+1):end,:]
        ppmmx  = @view ppm[chMajor,:]
        for k = 1:nStages[d]
            # first step

            # B Λ(z_d) B'
            butterfly!(ppm, minP)
            ppmlw1 .= circshift(ppmlw1, (0, nStrides[d]))
            butterfly!(ppm, minP)

            # ppmmn .= propMats[2k-1] * ppmmn


            # second step
            # B Λ(z_d) B'
            butterfly!(ppm, minP)
            ppmlw2 .= circshift(ppmlw2, (0, nStrides[d]))
            butterfly!(ppm, minP)

            ppmmx .= propMats[2k] * ppmmx
        end
    end
    ppm
end

kernels(pfb::PolyphaseFB) = (analysiskernels(pfb), synthesiskernels(pfb))

function analysiskernels(pfb::PolyphaseFB)
    df = decimations(pfb)

    afb = analysisbank(pfb)
    return map([ @view(afb[p,:]) for p in 1:size(afb, 1) ]) do vf
        out = similar(vf, kernelsize(pfb)...)
        tilesout = collect(TileIterator(axes(out), df))

        foldl(1:length(tilesout), init=out) do mtx, idx
            sub = (1:prod(df)) .+ ((idx - 1) * prod(df))
            setindex!(mtx, reshape(@view(vf[sub]), df...), tilesout[idx]...)
        end
    end
end

function synthesiskernels(cc::AbstractNsolt)
    map(analysiskernels(cc)) do af
        reshape(af .|> conj |> vec |> reverse, size(af))
    end
end

function mdarray2polyphase(x::AbstractArray{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    nBlocks = fld.(size(x), szBlock)
    if any(size(x) .% szBlock .!= 0)
        error("size error. input data: $(size(x)), block size: $(szBlock).")
    end

    outdata = similar(x, prod(szBlock), prod(nBlocks))
    tiles = collect(TileIterator(axes(x), szBlock))
    for idx in 1:length(tiles)
        outdata[:,idx] = vec(@view x[tiles[idx]...])
    end
    PolyphaseVector(outdata, nBlocks)
end

function mdarray2polyphase(x::AbstractArray{T,D}) where {T,D}
    nBlocks = size(x)[1:D-1]
    outdata = transpose(reshape(x, prod(nBlocks), size(x,D)))
    PolyphaseVector(outdata, nBlocks)
end

function polyphase2mdarray(x::PolyphaseVector{TX,D}, szBlock::NTuple{D,TS}) where {TX,D,TS<:Integer}
    if size(x.data,1) != prod(szBlock)
        throw(ArgumentError("size mismatch! 'prod(szBlock)' must be equal to $(size(x.data,1))."))
    end

    out = similar(x.data, (x.nBlocks .* szBlock)...)
    tiles = collect(TileIterator(axes(out), szBlock))
    for idx in 1:length(tiles)
        out[tiles[idx]...] = reshape(@view(x.data[:,idx]), szBlock...)
    end
    out
end

function polyphase2mdarray(x::PolyphaseVector{T,D}) where {T,D}
    reshape(transpose(x.data), x.nBlocks..., size(x.data, 1))
end

function shiftdimspv(x::AbstractMatrix, nBlocks::Integer)
    hcat([ @view x[:, (1:nBlocks:end) .+ idx] for idx = 0:nBlocks-1 ]...)
end

ishiftdimspv(x::AbstractMatrix, nBlocks::Integer) = shiftdimspv(x, fld(size(x, 2), nBlocks))

function butterfly!(x::AbstractMatrix, p::Integer)
    xu = x[1:p,:]
    xl = x[end-(p-1):end,:]

    x[1:p,:]           .= (xu + xl) / sqrt(2)
    x[end-(p-1):end,:] .= (xu - xl) / sqrt(2)
end

@inline function unnormalized_butterfly!(xu::T, xl::T) where {T<:AbstractMatrix}
    tu, tl = (xu + xl, xu - xl)
    xu .= tu
    xl .= tl
    nothing
end

@inline function half_butterfly!(xu::T, xl::T) where {T<:AbstractMatrix}
    tu, tl = (xu + xl, xu - xl) ./ 2
    xu .= tu
    xl .= tl
    nothing
end

function shiftforward!(::Val{:circular}, mtx::AbstractMatrix, nShift::Integer)
    mtx .= circshift(mtx, (0, nShift))
end
shiftbackward!(tp::Val{:circular}, mtx, nShift) = shiftforward!(tp, mtx, -nShift)

function shiftforward!(::Val{:zero}, mtx::AbstractMatrix, nShift::Integer)
    mtx[:,1+nShift:end] .= @view mtx[:,1:end-nShift]
    mtx[:,1:nShift] .= 0
    mtx
end

function shiftbackward!(::Val{:zero}, mtx::AbstractMatrix, nShift::Integer)
    mtx[:,1:end-nShift] .= @view mtx[:,1+nShift:end]
    mtx[:,end-nShift+1:end] .= 0
    mtx
end

shiftforward(tp::Val, mtx::AbstractMatrix, nShift) = shiftforward!(tp, deepcopy(mtx), nShift)

shiftbackward(tp::Val, mtx::AbstractMatrix, nShift) = shiftbackward!(tp, deepcopy(mtx), nShift)

# function lifting(nsolt::RnsoltTypeI{T,D}) where {T,D}
#     Pw = nsolt.nChannels[1]
#     Pu = nsolt.nChannels[2]
#     P = Pw + Pu
#     M = prod(nsolt.decimationFactor)
#     cM, fM = cld(M,2), fld(M,2)
#
#     initmtx = begin
#         perms = [ collect(1:cM)...,
#                   collect((1:Pw-cM) .+ M)...,
#                   collect((1:fM) .+ cM)...,
#                   collect((1:Pu-fM) .+ (Pw+fM))... ]
#
#         pmtx = foldl(1:length(perms), init=zeros(T,P,P)) do mtx, idx
#             setindex!(mtx, 1, idx, perms[idx])
#         end
#
#         V0 = cat(nsolt.initMatrices[1], nsolt.initMatrices[2], dims=[1,2])
#
#         [ V0 * pmtx ]
#     end
#
#     propMatrices = map(nsolt.propMatrices) do ppmold
#         ppmnew = [ [ Matrix{T}(I, Pw, Pw), copy(mtx) ] for mtx in ppmold ]
#
#         reduce( (lhs, rhs)->[lhs..., rhs...] , ppmnew)
#     end
#
#     paramAngs = Vector{Vector{T}}[ [ zeros(fld(nchannels(nsolt),4)) for n in 1:nsolt.polyphaseOrder[pd] ] for pd in 1:D ]
#
#     symmetry = [ ones(nsolt.nChannels[1]); -1im * ones(nsolt.nChannels[2]) ];
#
#     matrixF = diagm(0=>[ ones(cM); 1im * ones(fM) ]) * nsolt.matrixC
#
#     Cnsolt(decimations(nsolt), orders(nsolt), nchannels(nsolt), initmtx, propMatrices, paramAngs; symmetry=symmetry, matrixF=matrixF)
# end
