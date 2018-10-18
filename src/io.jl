using JSON
using InteractiveUtils: subtypes

function savefb(filename::AbstractString, nsolt::AbstractNsolt, mode::AbstractString="w")
    if match(r".*\.json$", filename) === nothing
        filename = string(filename, ".json")
    end
    open(io->savefb(io, nsolt), filename, mode)
end

savefb(io::IOStream, nsolt::AbstractNsolt) = print(io, serialize(nsolt))

function serialize(cc::AbstractNsolt{T,D}; format::AbstractString="JSON") where {T,D,S}
    configs = JSON.json(cc)
    fbname = if isa(cc, Cnsolt)
        "CNSOLT"
    elseif isa(cc, Rnsolt)
        "RNSOLT"
    end

    string("{\"Name\":\"", fbname ,"\",\"DataType\":\"", string(T), "\",\"Dimensions\":", D, ",\"Configurations\":", configs ,"}")
end

loadfb(filename::AbstractString) = open(io->MDCDL.loadfb(io), filename)

function loadfb(io::IOStream)
    dic = JSON.parse(read(io, String))

    deserialize(dic)
end

deserialize(dic::Dict; format::AbstractString="JSON") = deserialize(Val(Symbol(dic["Name"])), dic)

function deserialize(::Val{:CNSOLT}, dic::Dict)
    dtSet = Dict([ string(slf) => slf for slf in subtypes(AbstractFloat) ])

    T = dtSet[dic["DataType"]]
    D = dic["Dimensions"]
    cfgs = dic["Configurations"]

    dfa = Vector{Int}(undef, D)
    nch = Int
    ppoa = Vector{Int}(undef, D)
    foreach(keys(cfgs)) do key
        if key == "decimationFactor"
            dfa = cfgs[key]
        elseif key == "nChannels"
            nch = cfgs[key]
        elseif key == "polyphaseOrder"
            ppoa = cfgs[key]
        end
    end
    df = tuple(dfa...)
    ppo = tuple(ppoa...)

    nsolt = Cnsolt(T, df, ppo, nch)

    foreach(keys(cfgs)) do key
        if key == "initMatrices"
            for idx = 1:1
                nsolt.initMatrices[idx] .= Matrix{T}(hcat(cfgs[key][idx]...))
            end
        elseif key == "propMatrices"
            for d = 1:D, od = 1:2*ppo[d]
                nsolt.propMatrices[d][od] .= Matrix{T}(hcat(cfgs[key][d][od]...))
            end
        elseif key == "paramAngles"
            for d = 1:D, od = 1:ppo[d]
                nsolt.paramAngles[d][od] .= vcat(cfgs[key][d][od]...)
            end
        elseif key == "symmetry"
            cplxVal = map((u)-> map((x)-> x["re"]+1im*x["im"], u), cfgs[key])
            nsolt.symmetry .= Diagonal(hcat(cplxVal...))
        elseif key == "matrixF"
            cplxVal = map((u)-> map((x)-> x["re"]+1im*x["im"], u), cfgs[key])
            nsolt.matrixF .= Matrix{Complex{T}}(hcat(cplxVal...))
        end
    end

    nsolt
end

function deserialize(::Val{:RNSOLT}, dic::Dict)
    dtSet = Dict([ string(slf) => slf for slf in subtypes(AbstractFloat) ])

    T = dtSet[dic["DataType"]]
    D = dic["Dimensions"]
    cfgs = dic["Configurations"]

    dfa = Vector{Int}(undef, D)
    ncha = Vector{Int}(undef, 2)
    ppoa = Vector{Int}(undef, D)
    foreach(keys(cfgs)) do key
        if key == "decimationFactor"
            dfa = cfgs[key]
        elseif key == "nChannels"
            ncha = cfgs[key]
        elseif key == "polyphaseOrder"
            ppoa = cfgs[key]
        end
    end
    df = tuple(dfa...)
    nch = tuple(ncha...)
    ppo = tuple(ppoa...)

    nsolt = Rnsolt(T, df, ppo, nch)

    foreach(keys(cfgs)) do key
        if key == "initMatrices"
            for idx = 1:2
                nsolt.initMatrices[idx] .= Matrix{T}(hcat(cfgs[key][idx]...))
            end
        elseif key == "propMatrices"
            for d = 1:D, od = 1:ppo[d]
                nsolt.propMatrices[d][od] .= Matrix{T}(hcat(cfgs[key][d][od]...))
            end
        elseif key == "matrixC"
            nsolt.matrixC .= Matrix{T}(hcat(cfgs[key]...))
        end
    end

    nsolt
end
