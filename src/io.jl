using JSON
using InteractiveUtils: subtypes

function savefb(filename::AbstractString, nsolt::AbstractNsolt, mode::AbstractString="w")
    if match(r".*\.json$", filename) === nothing
        filename = string(filename, ".json")
    end
    open(io->savefb(io, nsolt), filename, mode)
end

savefb(io::IOStream, nsolt::AbstractNsolt) = print(io, serialize(nsolt))

function serialize(nsolt::NS) where {NS<:AbstractNsolt}
    # convert to Dict
    dict_nsolt = nsolt |> JSON.json |> JSON.parse

    nsolt_configs, nsolt_params = separate_nsolt_configs_params(NS, dict_nsolt)

    delete!(nsolt_configs, "nStages")
    push!(nsolt_configs, "polyphaseOrder" => collect(orders(nsolt)))

    if NS <: Cnsolt
        nsolt_params["Φ"] = map(diag(nsolt.Φ)) do p
            Dict( "re" => real(p), "im" => imag(p))
        end
    end

    outdict = Dict(
        "NsoltType" => string(nameof(NS)),
        "ElementType" => string(eltype(NS)),
        "Configs" => nsolt_configs,
        "Params" => nsolt_params,
    )
    JSON.json(outdict)
end

function separate_nsolt_configs_params(::Type{NS}, dict::Dict) where {NS<:AbstractNsolt}
    keynames = [ "decimationFactor", "polyphaseOrder", "nChannels", "perm" ]
    configs = filter(dict) do d
        any(d.first .== keynames)
    end
    params = filter(dict) do d
        !any(d.first .== keynames)
    end

    (configs, params)
end

loadfb(filename::AbstractString) = open(io->MDCDL.loadfb(io), filename)

function loadfb(io::IOStream)
    dic = JSON.parse(read(io, String))

    deserialize(dic)
end

function deserialize(dict::Dict;
        nsolttypeset=[ subtypes(Rnsolt)..., subtypes(Cnsolt)... ],
        eltypeset=subtypes(AbstractFloat)
    )

    strNS = dict["NsoltType"]
    strT = dict["ElementType"]
    dic_nsolt_configs = dict["Configs"]
    dic_nsolt_params = dict["Params"]

    NS = Dict( string.(nsolttypeset) .=> nsolttypeset )[strNS]
    T = Dict( string.(eltypeset) .=> eltypeset )[strT]

    nsolt = create_nsolt_by_dict(NS, T, dic_nsolt_configs)

    return set_nsolt_params_by_dict!(nsolt, dic_nsolt_params)
end

function create_nsolt_by_dict(::Type{NS}, T::Type, dict::Dict) where {NS<:AbstractNsolt}
    df = get_nsolt_config(NS, dict, "decimationFactor")
    ord = get_nsolt_config(NS, dict, "polyphaseOrder")
    nch = get_nsolt_config(NS, dict, "nChannels")
    perm = if haskey(dict, "perm")
        get_nsolt_config(NS, dict, "perm")
    else
        (collect(1:length(df))...,)
    end

    NS(T, df, ord, nch, perm=perm)
end

function set_nsolt_params_by_dict!(nsolt::NS, dict::Dict) where {NS<:RnsoltTypeI}
    initpm = [ nsolt.W0 => "W0", nsolt.U0 => "U0", nsolt.CJ => "CJ"]
    proppm = [ nsolt.Udks => "Udks" ]
    set_nsolt_params_by_dict!(NS, dict, initpm, proppm)

    return nsolt
end

function set_nsolt_params_by_dict!(nsolt::NS, dict::Dict) where {NS<:RnsoltTypeII}
    initpm = [ nsolt.W0 => "W0", nsolt.U0 => "U0", nsolt.CJ => "CJ"]
    proppm = [ nsolt.Wdks => "Wdks", nsolt.Udks => "Udks" ]
    set_nsolt_params_by_dict!(NS, dict, initpm, proppm)

    return nsolt
end

function set_nsolt_params_by_dict!(nsolt::NS, dict::Dict) where {NS<:CnsoltTypeI}
    initpm = [ nsolt.V0 => "V0", nsolt.FJ => "FJ"]
    proppm = [ nsolt.Wdks => "Wdks", nsolt.Udks => "Udks", nsolt.θdks => "θdks" ]
    set_nsolt_params_by_dict!(NS, dict, initpm, proppm)

    return nsolt
end

function set_nsolt_params_by_dict!(nsolt::NS, dict::Dict) where {NS<:CnsoltTypeII}
    initpm = [ nsolt.V0 => "V0", nsolt.FJ => "FJ"]
    proppm = [ nsolt.Wdks => "Wdks", nsolt.Udks => "Udks", nsolt.θ1dks => "θ1dks", nsolt.Ŵdks => "Ŵdks", nsolt.Ûdks => "Ûdks", nsolt.θ2dks => "θ2dks" ]
    set_nsolt_params_by_dict!(NS, dict, initpm, proppm)

    return nsolt
end

function set_nsolt_params_by_dict!(NS::Type, dict::Dict, initpm::Array, proppm::Array)
    foreach(initpm) do pm
        pm.first .= get_nsolt_config(NS, dict, pm.second)
    end

    foreach(proppm) do pm
        a = get_nsolt_config(NS, dict, pm.second)
        setparams_dk!(pm.first, a)
    end
end

function setparams_dk!(dst, src)
    foreach(dst, src) do dstk, srck
        foreach(dstk, srck) do dstkd, srckd
            dstkd .= srckd
        end
    end
end

get_nsolt_config(::Type{NS}, dict::Dict, akey::AbstractString) where {NS} = get_nsolt_config(NS, Val(Symbol(akey)), dict[akey])

function get_nsolt_config(::Type{NS}, ::Val{:decimationFactor}, data) where {NS<:AbstractNsolt}
    (Int.(data)...,)
end

function get_nsolt_config(::Type{NS}, ::Val{:polyphaseOrder}, data) where {NS<:AbstractNsolt}
    (Int.(data)...,)
end

function get_nsolt_config(::Type{NS}, ::Val{:nChannels}, data) where {NS<:Rnsolt}
    (Int.(data)...,)
end

function get_nsolt_config(::Type{NS}, ::Val{:perm}, data) where {NS<:AbstractNsolt}
    (Int.(data)...,)
end

function get_nsolt_config(::Type{NS}, ::Val{:nChannels}, data) where {NS<:Cnsolt}
    Int(sum(data))
end

function get_nsolt_config(::Type{NS}, ::Val{:CJ}, data) where {T,NS<:Rnsolt{T}}
    T.(hcat(data...))
end

function get_nsolt_config(::Type{NS}, ::Val{:FJ}, data) where {T,NS<:Cnsolt{T}}
    map(hcat(data...)) do FJpm
        Complex{T}(FJpm["re"] + FJpm["im"]*im)
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:W0}, data) where {T,NS<:Rnsolt{T}}
    T.(hcat(data...))
end

function get_nsolt_config(::Type{NS}, ::Val{:U0}, data) where {T,NS<:Rnsolt{T}}
    T.(hcat(data...))
end

function get_nsolt_config(::Type{NS}, ::Val{:V0}, data) where {T,NS<:Cnsolt{T}}
    T.(hcat(data...))
end

function get_nsolt_config(::Type{NS}, ::Val{:Udks}, data) where {T,NS<:AbstractNsolt{T}}
    map(data) do Ud
        map(Ud) do Udk
            T.(hcat(Udk...))
        end
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:Wdks}, data) where {T,NS<:RnsoltTypeII{T}}
    map(data) do Wd
        map(Wd) do Wdk
            T.(hcat(Wdk...))
        end
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:Wdks}, data) where {T,NS<:Cnsolt{T}}
    map(data) do Wd
        map(Wd) do Wdk
            T.(hcat(Wdk...))
        end
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:θdks}, data) where {T,NS<:CnsoltTypeI{T}}
    map(data) do θd
        map(θd) do θdk
            T.(θdk)
        end
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:θ1dks}, data) where {T,NS<:CnsoltTypeII{T}}
    map(data) do θd
        map(θd) do θdk
            T.(θdk)
        end
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:Ŵdks}, data) where {T,NS<:CnsoltTypeII{T}}
    map(data) do Wd
        map(Wd) do Wdk
            T.(hcat(Wdk...))
        end
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:Ûdks}, data) where {T,NS<:CnsoltTypeII{T}}
    map(data) do Wd
        map(Wd) do Wdk
            T.(hcat(Wdk...))
        end
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:θ2dks}, data) where {T,NS<:CnsoltTypeII{T}}
    map(data) do θd
        map(θd) do θdk
            T.(θdk)
        end
    end
end

function get_nsolt_config(::Type{NS}, ::Val{:Φ}, data) where {T,NS<:Cnsolt{T}}
    Diagonal(map(p->Complex{T}(p["re"] + p["im"]*im), data))
end
