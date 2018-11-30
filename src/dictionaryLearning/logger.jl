savesettings(::Nothing, args...; kwargs...) = nothing
function savesettings(dirname::AbstractString, nsolt::AbstractNsolt{T,D}, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), filename="settings", kwargs...) where {T,D}
    filepath = joinpath(dirname, filename)

    txt_general = """
        Settings of $(namestring(nsolt)) Dictionary Learning:
        Element type := $T
        Decimation factor := $(decimations(nsolt))
        Polyphase order := $(orders(nsolt))
        Number of channels := $(nchannels(nsolt))

        Number of training data := $(length(trainingset))
        Epochs := $epochs

        User-defined options for Sparse Coding Stage := $sc_options
        User-defined options for Dictionary Update Stage := $du_options

        """
    txt_others = [ "Keyword $(arg.first) := $(arg.second)\n" for arg in kwargs]

    open(filepath, write=true) do io
        println(io, txt_general, txt_others...)
    end
    vlevel >= 2 && println("Settings was written in $filename.")
    nothing
end


function savesettings(dirname::AbstractString, targets::Multiscale, trainingset::AbstractArray; vlevel=0, epochs, sc_options=(), du_options=(), filename="settings", kwargs...) where {T,D}
    for idx = 1:length(targets)
        savesettings(dirname, targets.filterbanks[idx], trainingset; vlevel=vlevel, epochs=epochs, sc_options=sc_options, du_options=du_options, filename=string(filename,"_",idx))
    end
end

savelogs(::Nothing, args...; kwargs...) = nothing
function savelogs(dirname::AbstractString, nsolt::AbstractNsolt, epoch::Integer; params...)
    filename_logs = joinpath(dirname, "log")
    filename_nsolt = joinpath(dirname, "nsolt.json")

    strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
    strlogs = string("epoch $epoch:", strparams...)
    open(filename_logs, append=true) do io
        println(io, strlogs)
    end
    savefb(filename_nsolt, nsolt)
    nothing
end

function savelogs(dirname::AbstractString, targets::Multiscale, epoch::Integer; params...) # where {N,CB<:AbstractNsolt}
    for idx = 1:length(targets)
        filename_nsolt = joinpath(dirname, string("nsolt_", idx, ".json"))
        savefb(filename_nsolt, targets.filterbanks[idx])
    end
    return nothing
    # filename_logs = joinpath(dirname, "log")
    # filename_nsolt = joinpath(dirname, "nsolt.json")
    #
    # strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
    # strlogs = string("epoch $epoch:", strparams...)
    # open(filename_logs, append=true) do io
    #     println(io, strlogs)
    # end
    # save(nsolt, filename_nsolt)
end

namestring(nsolt::Rnsolt) = namestring(nsolt, "Real NSOLT")
namestring(nsolt::Cnsolt) = namestring(nsolt, "Complex NSOLT")
namestring(nsolt::AbstractNsolt{T,D}, strnsolt::AbstractString) where {T,D} = "$D-dimensional $(ifelse(istype1(nsolt), "Type-I", "Type-II")) $strnsolt"
