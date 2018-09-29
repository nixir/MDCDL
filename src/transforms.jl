function reshape_polyvec(::Shapes.Default, ::NsoltOperator, pvy::PolyphaseVector)
    [ reshape(pvy.data[p,:], pvy.nBlocks) for p in 1:size(pvy.data,1) ]
end

function reshape_polyvec(::Shapes.Arrayed, ::NsoltOperator, pvy::PolyphaseVector)
    polyphase2mdarray(pvy)
end

function reshape_polyvec(::Shapes.Vec, ::NsoltOperator, pvy::PolyphaseVector)
    vec(transpose(pvy.data))
end

function reshape_polyvec(::Shapes.Default, ::ConvolutionalOperator, y::AbstractArray)
    y
end

function reshape_polyvec(::Shapes.Arrayed, ::ConvolutionalOperator{TF,D}, y::AbstractArray{TY,D}) where {TF,TY,D}
    cat(D+1, y...)
end

function reshape_polyvec(::Shapes.Vec, ::ConvolutionalOperator, y::AbstractArray)
    vcat(vec.(y)...)
end


function reshape_coefs(::Shapes.Default, ::NsoltOperator, y::AbstractArray)
    PolyphaseVector(hcat(vec.(y)...) |> transpose |> Matrix, size(y[1]))
end

function reshape_coefs(::Shapes.Arrayed, ::NsoltOperator, y::AbstractArray)
    mdarray2polyphase(y)
end

function reshape_coefs(::Shapes.Vec, nsop::NsoltOperator, y::AbstractArray)
    szout = fld.(nsop.insize, decimations(nsop))
    ty = reshape(y, szout..., nchannels(nsop))
    mdarray2polyphase(ty)
end

function reshape_coefs(::Shapes.Default, ::ConvolutionalOperator, y::AbstractArray)
    y
end

function reshape_coefs(::Shapes.Arrayed, co::ConvolutionalOperator{T,D}, y::AbstractArray) where {T,D}
    [ y[fill(:,D)..., p] for p in 1:nchannels(co)]
end

function reshape_coefs(::Shapes.Vec, co::ConvolutionalOperator{T,D}, y::AbstractArray) where {T,D}
    ry = reshape(y, fld.(co.insize, decimations(co))..., nchannels(co))
    [ ry[fill(:,D)..., p] for p in 1:nchannels(co) ]
end
