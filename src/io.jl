using JSON

function save(filename::AbstractString, cc::MDCDL.Cnsolt{T,D,S}) where {T,D,S}
    configs = JSON.json(cc)
    outstr = string("{\"CNSOLT\":{\"DataType\":\"", string(T), "\",\"Dimensions\":", D, ",\"Configurations\":", configs ,"}}")

    open(io->println(io,outstr), filename, "w")
end

function save(filename::AbstractString, cc::MDCDL.Rnsolt{T,D,S}) where {T,D,S}
    configs = JSON.json(cc)
    outstr = string("{\"RNSOLT\":{\"DataType\":\"", string(T), "\",\"Dimensions\":", D, ",\"Configurations\":", configs ,"}}")

    open(io->println(io,outstr), filename, "w")
end

function load(filename::AbstractString)
    str = open(readstring, filename)
    dic = JSON.parse(str)

    fbs = map(keys(dic)) do key
        if key == "CNSOLT"
            loadCnsolt(dic[key])
        elseif key == "RNSOLT"
            loadRnsolt(dic[key])
        else
            throw(KeyError(key))
        end
    end
    fbs[1]
end

function loadCnsolt(dic::Dict)
    const dtSet = Dict([ string(slf) => slf for slf in subtypes(AbstractFloat) ])

    T = dtSet[dic["DataType"]]
    D = dic["Dimensions"]
    cfgs = dic["Configurations"]

    dfa = Vector{Int}(D)
    nch = Int
    ppoa = Vector{Int}(D)
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

    nsolt = Cnsolt(T, df, nch, ppo)

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

function loadRnsolt(dic::Dict)
    const dtSet = Dict([ string(slf) => slf for slf in subtypes(AbstractFloat) ])

    T = dtSet[dic["DataType"]]
    D = dic["Dimensions"]
    cfgs = dic["Configurations"]

    dfa = Vector{Int}(D)
    ncha = Vector{Int}(2)
    ppoa = Vector{Int}(D)
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

    nsolt = Rnsolt(T, df, nch, ppo)

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
