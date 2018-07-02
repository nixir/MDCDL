using Base.Test
using MDCDL

@testset "RNSOLT" begin
    @testset "Reconstruction" begin
        include("randomInit.jl")

        maxDims = 3
        # maxdfs = tuple(fill(4,D)...)
        maxdfs = [ 4, 4, 3 ]
        maxchs = [ 9, 9, 5 ]
        maxpos = [ 4, 4, 2 ]
        # maxpos = tuple(fill(4,D)...)
        for d in 1:maxDims, crdf in CartesianRange(tuple(fill(maxdfs[d],d)...)), crnch in CartesianRange(tuple(fill(maxchs[d],2)...)), crord in CartesianRange(tuple(fill(maxpos[d] .+ 1,d)...))
            df = crdf.I
            ord = crord.I .- 1
            nch = crnch.I

            szx = df .* (ord .+ 1)

            if prod(df) > sum(nch)
                @test_throws ArgumentError Rnsolt(df, nch, ord)
                continue
            end

            if nch[1] != nch[2] && any(ord .% 2 .!= 0)
                # @test_throws ArgumentError Rnsolt(df, nch, ord)
                continue
            end

            if !(cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) || !(fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
                @test_throws ArgumentError Rnsolt(df, nch, ord)
                continue
            end

            println("df = $df, nch = $nch, ord = $ord")

            nsolt = Rnsolt(df, nch, ord)


            x = rand(Float64, szx...)
            y, sc = analyze(nsolt, x)
            rx = synthesize(nsolt, y, sc)

            @test size(x) == size(rx)
            @test rx ≈ x
        end
    end

end


nothing
