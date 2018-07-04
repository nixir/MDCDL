using Base.Test
using MDCDL

@testset "RNSOLT" begin
    include("testsetGenerator.jl")
    include("randomInit.jl")

    rcsd1D = rnsoltValidConfigSet1D()
    rcsd2D = rnsoltValidConfigSet2D()
    rcsd3D = rnsoltValidConfigSet3D()

    rcsd = [ rcsd1D, rcsd2D, rcsd3D ]

    srand(9387509284)

    @testset "Constructor" begin

        maxDims = 3
        defaultType = Float64

        for d in 1:maxDims
            allcfgset = [ (crdf.I, crnch.I, crord.I .- 1) for crdf in CartesianRange(tuple(fill(4,d)...)), crnch in CartesianRange(tuple(fill(10,2)...)), crord in CartesianRange(tuple(fill(6+1,d)...)) ]
            cfgsetTypeI  = filter(c -> c[2][1] == c[2][2], allcfgset)
            cfgsetTypeII = filter(c -> c[2][1] != c[2][2], allcfgset)
            cfgset = vcat(randsubseq(cfgsetTypeI, 50 / length(cfgsetTypeI))..., randsubseq(cfgsetTypeII, 50 / length(cfgsetTypeII))...)

            for (df, nch, ord) in cfgset
                if prod(df) > sum(nch)
                    @test_throws ArgumentError Rnsolt(df, nch, ord)
                    continue
                end

                if !(cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) || !(fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
                    @test_throws ArgumentError Rnsolt(df, nch, ord)
                    continue
                end

                if nch[1] != nch[2] && any(isodd.(ord))
                    @test_throws ArgumentError Rnsolt(df, nch, ord)
                    continue
                end

                nsolt = Rnsolt(df, nch, ord)

                if nch[1] == nch[2]
                    @test isa(nsolt, Rnsolt{defaultType,d,:TypeI})
                else
                    @test isa(nsolt, Rnsolt{defaultType,d,:TypeII})
                end
            end
        end
    end

    # @testset "DefaultValues"
    #     for d in 1:length(rcsd), (df, nch, ord) in rcsd[d]
    #         nsolt = Rnsolt(df, nch, ord)
    #
    #         df = 0
    #     end
    # end

    @testset "FilterSymmetry" begin
        for d in 1:length(rcsd), (df, nch, ord) in rcsd[d]
            nsolt = Rnsolt(df, nch, ord)
            randomInit!(nsolt)

            afb = MDCDL.getAnalysisBank(nsolt)
            Γ = Diagonal(vcat(fill(1,nch[1]), fill(-1,nch[2])))

            @test afb ≈ Γ * flipdim(afb,2)
        end
    end

    @testset "AnalysisSynthesis" begin
        for d in 1:length(rcsd), (df, nch, ord) in rcsd[d], lv in 1:3
            szx = (df.^lv) .* (ord .+ 1)

            nsolt = Rnsolt(df, nch, ord)
            randomInit!(nsolt)

            x = rand(Float64, szx...)

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
        for d in 1:length(rcsd), (df, nch, ord) in rcsd[d], lv in 1:3
            szx = (df.^lv) .* (ord .+ 1)
            x = rand(Float64, szx)

            nsolt = Rnsolt(df, nch, ord)
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
        for d in 1:length(rcsd), (df, nch, ord) in rcsd[d], lv in 1:3

            nsolt = Rnsolt(df, nch, ord)
            randomInit!(nsolt)

            y = [ [ rand(Float64,((ord.+1) .* df.^(lv-l))...) for p in 1:(l==lv ? sum(nch) : sum(nch)-1) ] for l in 1:lv]

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

    @testset "Factorization" begin
        for d in 1:length(rcsd), (df, nch, ord) in rcsd[d]
            src = Rnsolt(df, nch, ord)
            dst = Rnsolt(df, nch, ord)
            randomInit!(src)

            if isa(src, Rnsolt{Float64,d,:TypeII})
                continue
            end
            (angs, mus) = getAngleParameters(src)
            setAngleParameters!(dst, angs, mus)

            @test all(src.initMatrices .≈ dst.initMatrices)
            foreach(src.propMatrices, dst.propMatrices) do propSrc, propDst
                @test all(propSrc .≈ propDst)
            end
        end
    end

end


nothing
