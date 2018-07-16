using MDCDL
@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

# write your own tests here
# @test 1 == 2
include("parallelFB.jl")
include("cnsolt.jl")
include("rnsolt.jl")

include("multiscale.jl")
