using ForwardDiff
using NLopt
using DataFrames
using FileIO
using Dates: now

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

initializelogs(::Nothing, args...; kwargs...) = ()

function initializelogs(dirname::AbstractString, x::AbstractArray; params...)
    csvheader = [ :epoch, :timestamp, [ Symbol("data_no$idx") for idx = 1:length(x) ]...]

    cfg_sp = (   filename = "loss_sparse_coding.csv",
                        csvheader = copy(csvheader),)
    cfg_du = (   filename = "loss_dictionary_update.csv",
                        csvheader = copy(csvheader),)

    empty_dataset = fill([], length(csvheader))
    save(joinpath(dirname, cfg_sp.filename), DataFrame(empty_dataset, cfg_sp.csvheader))
    save(joinpath(dirname, cfg_du.filename), DataFrame(empty_dataset, cfg_du.csvheader))

    (cfg_sp, cfg_du)
end

savelogs(::Nothing, args...; kwargs...) = nothing

function savelogs(  dirname::AbstractString, target, epoch::Integer,
                    cfg_sp::NamedTuple, cfg_du::NamedTuple;
                    timestamp=string(now()), loss_sparse_coding,
                    loss_dictionary_update, vlevel)
    savelogs_dictionary(dirname, target)
    # savefb(joinpath(dirname, filename_nsolt), nsolt)

    open(joinpath(dirname, cfg_sp.filename), append=true) do io
        dat = DataFrame([ epoch timestamp loss_sparse_coding...], cfg_sp.csvheader)
        save(Stream(format"CSV", io), dat, header = false)
    end

    open(joinpath(dirname, cfg_du.filename), append=true) do io
        dat = DataFrame([ epoch timestamp loss_dictionary_update...], cfg_du.csvheader)
        save(Stream(format"CSV", io), dat, header = false)
    end
end

function savelogs_dictionary(dirname::AbstractString, nsolt::AbstractNsolt)
    filename_nsolt = "nsolt.json"
    savefb(joinpath(dirname, filename_nsolt), nsolt)
end

function savelogs_dictionary(dirname::AbstractString, targets::Multiscale) # where {N,CB<:AbstractNsolt}
    for idx = 1:length(targets)
        filename_nsolt = joinpath(dirname, string("nsolt_$idx.json"))
        savefb(filename_nsolt, targets.filterbanks[idx])
    end
end

# savelogs(::Nothing, args...; kwargs...) = nothing
#
# function savelogs(dirname::AbstractString, nsolt::AbstractNsolt, epoch::Integer; params...)
#     filename_logs = joinpath(dirname, "log")
#     filename_nsolt = joinpath(dirname, "nsolt.json")
#
#     strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
#     strlogs = string("epoch $epoch:", strparams...)
#     open(filename_logs, append=true) do io
#         println(io, strlogs)
#     end
#     savefb(filename_nsolt, nsolt)
#     nothing
# end
#
# function savelogs(dirname::AbstractString, targets::Multiscale, epoch::Integer; params...) # where {N,CB<:AbstractNsolt}
#     for idx = 1:length(targets)
#         filename_nsolt = joinpath(dirname, string("nsolt_", idx, ".json"))
#         savefb(filename_nsolt, targets.filterbanks[idx])
#     end
#     return nothing
#     # filename_logs = joinpath(dirname, "log")
#     # filename_nsolt = joinpath(dirname, "nsolt.json")
#     #
#     # strparams = [ " $(prm.first) = $(prm.second)," for prm in params ]
#     # strlogs = string("epoch $epoch:", strparams...)
#     # open(filename_logs, append=true) do io
#     #     println(io, strlogs)
#     # end
#     # save(nsolt, filename_nsolt)
# end

namestring(nsolt::Rnsolt) = namestring(nsolt, "Real NSOLT")
namestring(nsolt::Cnsolt) = namestring(nsolt, "Complex NSOLT")
namestring(nsolt::AbstractNsolt{T,D}, strnsolt::AbstractString) where {T,D} = "$D-dimensional $(ifelse(istype1(nsolt), "Type-I", "Type-II")) $strnsolt"
