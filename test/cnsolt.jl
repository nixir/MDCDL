using Base.Test
using MDCDL

@testset "CNSOLT" begin
    include("testsetGenerator.jl")
    include("randomInit.jl")

    ccsd1D = cnsoltValidConfigSet1D()
    ccsd2D = cnsoltValidConfigSet2D()
    ccsd3D = cnsoltValidConfigSet3D()

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
        defaultType = Float64

        # cfgset = [ (crdf.I, nch, crord.I .- 1) for crdf in CartesianRange(tuple(fill(4,d)...)), nch in 2:20, crord in CartesianRange(tuple(fill(6,d)...)) ]

        for d in 1:maxDims
            allcfgset = [ (crdf.I, nch, crord.I .- 1) for crdf in CartesianRange(tuple(fill(4,d)...)), nch in 2:20, crord in CartesianRange(tuple(fill(6+1,d)...)) ]
            cfgset = randsubseq(vec(allcfgset), 100 / length(allcfgset))

            for (df, nch, ord) in cfgset
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
                    @test isa(nsolt, Cnsolt{defaultType,d,:TypeI})
                else
                    @test isa(nsolt, Cnsolt{defaultType,d,:TypeII})
                end
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
        # output mode options for analyzer
        oms = [ :polyphase, :reshaped, :augumented ]
        for d in 1:length(ccsd), (df, nch, ord) in ccsd[d]
            szx = df .* (ord .+ 1)
            nsolt = Cnsolt(df, nch, ord)
            randomInit!(nsolt)

            x = rand(Complex{Float64}, szx...)

            y = analyze(nsolt, x)
            rx = synthesize(nsolt, y)

            @test size(x) == size(rx)
            @test rx ≈ x

            foreach(oms) do om
                y = analyze(nsolt, x; outputMode = om)
                rx = synthesize(nsolt, y)

                @test size(x) == size(rx)
                @test rx ≈ x
            end
        end
    end

    @testset "AnalysisSynthesisMultiscale" begin
        # output mode options for analyzer
        oms = [ :polyphase, :reshaped, :augumented ]
        for d in 1:length(ccsd), (df, nch, ord) in ccsd[d], lv in 1:3
            szx = (df.^lv) .* (ord .+ 1)
            nsolt = Cnsolt(df, nch, ord)
            randomInit!(nsolt)

            x = rand(Complex{Float64}, szx...)

            y = analyze(nsolt, x, lv)
            rx = synthesize(nsolt, y, lv)

            @test size(x) == size(rx)
            @test rx ≈ x

            foreach(oms) do om
                y = analyze(nsolt, x, lv; outputMode = om)
                rx = synthesize(nsolt, y, lv)

                @test size(x) == size(rx)
                @test rx ≈ x
            end
        end
    end

    @testset "AnalyzerKernel" begin
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
                sys = [ circshift(downsample(myfilter(subx, af), df, offset), (-1 .* fld.(ord,2))) for af in afs]
                if idx < lv
                    @test all(ya[idx] .≈ sys[2:end])
                    subx = sys[1]
                else
                    @test all(ya[idx] .≈ sys)
                end
            end

        end
    end

    @testset "SynthesizerKernel" begin
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
                subrx = circshift(subrxs, -1 .* df .* cld.(ord,2))
                if idx > 1
                    suby = [subrx, y[idx-1]... ]
                else
                    @test size(x) == size(subrx)
                    @test subrx ≈ x
                end
            end

        end
    end

    @testset "Factorization" begin
        for d in 1:length(ccsd), (df, nch, ord) in ccsd[d]
            src = Cnsolt(df, nch, ord)
            dst = Cnsolt(df, nch, ord)
            randomInit!(src)

            (angs, mus) = getAngleParameters(src)
            setAngleParameters!(dst, angs, mus)

            @test all(src.initMatrices .≈ dst.initMatrices)
            foreach(src.propMatrices, dst.propMatrices) do propSrc, propDst
                @test all(propSrc .≈ propDst)
            end
            foreach(src.paramAngles, dst.paramAngles) do angsSrc, angsDst
                @test all(angsSrc .≈ angsDst)
            end
        end
    end
end

nothing
