
function save(cc::MDCDL.Cnsolt{D,S,T}, filename::String) where {D,S,T}
    dic = Dict([
        "numberOfDimensions" => D
        "decimationFactor" => cc.decimationFactor
        "numberOfChannels" => cc.nChannels
        "polyphaseOrder" => cc.polyphaseOrder
        "initMatrices" => cc.initMatrices
        "propMatrices" => cc.propMatrices
        "paramAngles" => cc.paramAngles
        "symetry" => cc.symmetry
    ])
    JSON.json(dic)
end



txt = save(cnsolt,"dummy")
println(txt)
