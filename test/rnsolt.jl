using Base.Test
using MDCDL

@testset "RNSOLT" begin
    include("testsetGenerator.jl")
    include("randomInit.jl")

    rcsd1D = rnsoltConfigSet1D()
    rcsd2D = rnsoltConfigSet2D()
    rcsd3D = rnsoltConfigSet3D()

    rcsd = [ rcsd1D, rcsd2D, rcsd3D ]

    srand(9387509284)

    @testset "Constructor" begin

        maxDims = 3
        maxdfs = [ 4, 4, 4 ]
        maxchs = [ 5, 5, 5 ]
        maxpos = [ 2, 2, 2 ]
        defaultType = Float64

        for d in 1:maxDims, crdf in CartesianRange(tuple(fill(maxdfs[d],d)...)), crnch in CartesianRange(tuple(fill(maxchs[d],2)...)), crord in CartesianRange(tuple(fill(maxpos[d] .+ 1,d)...))
            df = crdf.I
            nch = crnch.I
            ord = crord.I .- 1
            szx = df .* (ord .+ 1)

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
                @test isa(nsolt, Rnsolt{d,1,defaultType})
            else
                @test isa(nsolt, Rnsolt{d,2,defaultType})
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

            y, sc = analyze(nsolt, x, lv)
            rx = synthesize(nsolt, y, sc, lv)

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

            ya, sc = analyze(nsolt, x, lv)

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
            sc = [ (ord .+ 1) .* (df .^ l) for l in lv:-1:0]

            x = synthesize(nsolt, y, sc, lv)

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
