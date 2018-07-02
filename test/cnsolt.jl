using Base.Test
using MDCDL

@testset "CNSOLT" begin
    @testset "Reconstruction" begin
        include("randomInit.jl")

        maxDims = 3
        # maxdfs = tuple(fill(4,D)...)
        maxdfs = [ 4, 4, 3 ]
        maxchs = [ 18, 18, 10 ]
        maxpos = [ 4, 4, 2 ]
        # maxpos = tuple(fill(4,D)...)
        for d in 1:maxDims, crdf in CartesianRange(tuple(fill(maxdfs[d],d)...)), nch in 2:maxchs[d], crord in CartesianRange(tuple(fill(maxpos[d] .+ 1,d)...))
            df = crdf.I
            ord = crord.I .- 1
            szx = df .* (ord .+ 1)

            if prod(df) > nch
                @test_throws ArgumentError Cnsolt(df, nch, ord)
                continue
            end

            if nch % 2 != 0 && any(ord .% 2 .!= 0)
                @test_throws ArgumentError Cnsolt(df, nch, ord)
                continue
            end

            println("df = $df, nch = $nch, ord = $ord")

            nsolt = Cnsolt(df, nch, ord)


            x = rand(Complex{Float64}, szx...)
            y, sc = analyze(nsolt, x)
            rx = synthesize(nsolt, y, sc)

            @test size(x) == size(rx)
            @test rx ≈ x
        end
    end

end


nothing
