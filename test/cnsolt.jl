using Base.Test
using MDCDL

@testset "CNSOLT" begin
    include("testsetGenerator.jl")
    include("randomInit.jl")

    ccsd1D = cnsoltConfigSet1D()
    ccsd2D = cnsoltConfigSet2D()
    ccsd3D = cnsoltConfigSet3D()

    ccsd = [ ccsd1D, ccsd2D, ccsd3D ]

    srand(392875929)

    # @testset "TypeSystem" begin
    #     dataTypeSet = [Float16, Float32, Float64]
    #     dimSet = collect(1:10)
    #
    #     for D in dimSet
    #         # @test Cnsolt{D,1,Float16} <: Cnsolt{D,1,Float32}
    #
    #         for T in dataTypeSet
    #             @test Cnsolt{D,1,T} <: PolyphaseFB{Complex{T},D}
    #             @test Cnsolt{D,2,T} <: PolyphaseFB{Complex{T},D}
    #         end
    #     end
    # end

    @testset "Constructor" begin

        maxDims = 3
        maxdfs = [ 3, 3, 3 ]
        maxchs = [ 10, 10, 10 ]
        maxpos = [ 2, 2, 2 ]
        defaultType = Float64

        for d in 1:maxDims, crdf in CartesianRange(tuple(fill(maxdfs[d],d)...)), nch in 2:maxchs[d], crord in CartesianRange(tuple(fill(maxpos[d] .+ 1,d)...))
            df = crdf.I
            ord = crord.I .- 1
            szx = df .* (ord .+ 1)

            if prod(df) > sum(nch)
                @test_throws ArgumentError Cnsolt(df, nch, ord)
                continue
            end

            if isodd(sum(nch)) && any(isodd.(ord))
                @test_throws ArgumentError Cnsolt(df, nch, ord)
                continue
            end

            nsolt = Cnsolt(df, nch, ord)

            if iseven(sum(nch))
                @test isa(nsolt, Cnsolt{d,1,defaultType})
            else
                @test isa(nsolt, Cnsolt{d,2,defaultType})
            end

        end
    end

    # @testset "DefaultValues"
    #     for d in 1:length(ccsd), (df, nch, ord) in ccsd[d]
    #         nsolt = Cnsolt(df, nch, ord)
    #
    #         df = 0
    #     end
    # end

    @testset "FilterSymmetry" begin
        for d in 1:length(ccsd), (df, nch, ord) in ccsd[d]
            nsolt = Cnsolt(df, nch, ord)
            randomInit!(nsolt)

            afb = MDCDL.getAnalysisBank(nsolt)
            hsafb = nsolt.symmetry' * afb

            @test hsafb ≈ conj(flipdim(hsafb,2))
        end
    end

    @testset "AnalysisSynthesis" begin
        for d in 1:length(ccsd), (df, nch, ord) in ccsd[d], lv in 1:3
            szx = (df.^lv) .* (ord .+ 1)

            nsolt = Cnsolt(df, nch, ord)
            randomInit!(nsolt)

            x = rand(Complex{Float64}, szx...)

            y = analyze(nsolt, x, lv; outputMode = :polyphase)
            rx = synthesize(nsolt, y, lv)

            @test size(x) == size(rx)
            @test rx ≈ x

            y = analyze(nsolt, x, lv; outputMode = :reshaped)
            rx = synthesize(nsolt, y, lv)

            @test size(x) == size(rx)
            @test rx ≈ x
        end
    end

    @testset "MultiscaleAnalyzer" begin
        for d in 1:length(ccsd), (df, nch, ord) in ccsd[d], lv in 1:3
            szx = (df.^lv) .* (ord .+ 1)
            x = rand(Complex{Float64}, szx)

            nsolt = Cnsolt(df, nch, ord)
            randomInit!(nsolt)

            ya = analyze(nsolt, x, lv; outputMode = :reshaped)

            afs = getAnalysisFilters(nsolt)
            myfilter = (A, h) -> MDCDL.mdfilter(A, h; boundary=:circular, operation=:conv)
            offset = df .- 1
            subx = x
            for idx = 1:lv-1
                sys = [ downsample(myfilter(subx, af), df, offset) for af in afs]
                if idx < lv
                    @test all(ya[idx] .≈ sys[2:end])
                    subx = sys[1]
                else
                    @test all(ya[idx] .≈ sys)
                end
            end

        end
    end

    @testset "MultiscaleSynthesizer" begin
        for d in 1:length(ccsd), (df, nch, ord) in ccsd[d], lv in 1:3

            nsolt = Cnsolt(df, nch, ord)
            randomInit!(nsolt)

            y = [ [ rand(Complex{Float64},((ord.+1) .* df.^(lv-l))...) for p in 1:(l==lv ? sum(nch) : sum(nch)-1) ] for l in 1:lv]

            x = synthesize(nsolt, y, lv)

            sfs = getSynthesisFilters(nsolt)
            myfilter = (A, h) -> MDCDL.mdfilter(A, h; boundary=:circular, operation=:conv)
            offset = df .- 1
            suby = y[lv]
            for idx = lv:-1:1
                subrxs = sum([ myfilter( MDCDL.upsample(suby[p], df), sfs[p] ) for p in 1:sum(nch) ])
                subrx = circshift(subrxs, -1 .* df .* ord)
                if idx > 1
                    suby = [subrx, y[idx-1]... ]
                else
                    @test size(x) == size(subrx)
                    @test subrx ≈ x
                end
            end

        end
    end
end


nothing
