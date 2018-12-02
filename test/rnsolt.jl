using Test
using MDCDL
using FFTW
using Random
using LinearAlgebra

@testset "RNSOLT" begin
    include("testsetGenerator.jl")

    rcsd1D = rnsoltValidConfigSet1D()
    rcsd2D = rnsoltValidConfigSet2D()
    rcsd3D = rnsoltValidConfigSet3D()

    rcsd = [ rcsd1D, rcsd2D, rcsd3D ]

    Random.seed!(9387509284)

    @testset "Constructor" begin

        maxDims = 3
        defaultType = Float64

        for d in 1:maxDims
            allcfgset = [ (crdf.I, crord.I .- 1, crnch.I) for crdf in CartesianIndices(tuple(fill(4,d)...)), crord in CartesianIndices(tuple(fill(6+1,d)...)), crnch in CartesianIndices(tuple(fill(10,2)...)) ]
            cfgsetTypeI  = filter(c -> c[3][1] == c[3][2], allcfgset)
            cfgsetTypeII = filter(c -> c[3][1] != c[3][2], allcfgset)
            cfgset = vcat(randsubseq(cfgsetTypeI, 50 / length(cfgsetTypeI))..., randsubseq(cfgsetTypeII, 50 / length(cfgsetTypeII))...)

            for (df, ord, nch) in cfgset
                if prod(df) > sum(nch)
                    @test_throws AssertionError Rnsolt(df, ord, nch)
                    continue
                end

                if !(cld(prod(df),2) <= nch[1] <= sum(nch) - fld(prod(df),2)) || !(fld(prod(df),2) <= nch[2] <= sum(nch) - cld(prod(df),2))
                    @test_throws AssertionError Rnsolt(df, ord, nch)
                    continue
                end

                if nch[1] != nch[2] && any(isodd.(ord))
                    @test_throws AssertionError Rnsolt(df, ord, nch)
                    continue
                end

                nsolt = Rnsolt(df, ord, nch)

                if nch[1] == nch[2]
                    @test nsolt isa RnsoltTypeI
                    @test istype1(nsolt) == true && istype2(nsolt) == false
                else
                    @test nsolt isa RnsoltTypeII
                    @test istype1(nsolt) == false && istype2(nsolt) == true
                end
            end
        end
    end

    @testset "DefaultValues" begin
        @testset "maximally-sampled" begin
            for d in 1:3
                for n = 1:20
                    df = (rand(1:4, d)...,)
                    ord = (fill(0, d)...,)
                    nch = prod(df)

                    x = rand(df...)
                    A = MDCDL.permdctmtx(df...)
                    expctd = A*vec(x)

                    nsolt = Rnsolt(df, ord, nch)
                    pvx = reshape(x, prod(df), :)
                    y = analyze(nsolt, pvx, (fill(1,d)...,))
                    actual = reshape(y, prod(df))

                    @test actual ≈ expctd
                end
            end
        end
    end

    @testset "FilterSymmetry" begin
        for d in 1:length(rcsd), (df, ord, nch) in rcsd[d]
            nsolt = Rnsolt(df, ord, nch)
            rand!(nsolt)

            afb = MDCDL.analysisbank(nsolt)
            Γ = Diagonal(vcat(fill(1,nch[1]), fill(-1,nch[2])))

            @test afb ≈ Γ * reverse(afb; dims=2)
        end
    end

    @testset "AnalysisSynthesis" begin
        # output mode options for analyzer
        oms = [ Shapes.Separated, Shapes.Combined, Shapes.Vec ]
        @testset "NormalOrder" begin
            for d in 1:length(rcsd), (df, ord, nch) in rcsd[d]
                szx = df .* (ord .+ 1)
                nsolt = Rnsolt(df, ord, nch)
                rand!(nsolt)

                x = rand(Float64, szx...)

                foreach(oms) do om
                    nsop = createTransform(nsolt, om(size(x)))

                    y = analyze(nsop, x)
                    rx = synthesize(nsop, y)

                    @test size(x) == size(rx)
                    @test rx ≈ x
                end
            end
        end
        @testset "DimensionPermutation" begin
            for d in 1:length(rcsd), (df, ord, nch) in rcsd[d]
                szx = df .* (ord .+ 1)
                nsolt = Rnsolt(df, ord, nch, perm=(randperm(d)...,))
                rand!(nsolt)

                x = rand(Float64, szx...)

                foreach(oms) do om
                    nsop = createTransform(nsolt, om(size(x)))

                    y = analyze(nsop, x)
                    rx = synthesize(nsop, y)

                    @test size(x) == size(rx)
                    @test rx ≈ x
                end
            end
        end
    end

    @testset "AnalyzerKernel" begin
        for d in 1:length(rcsd), (df, ord, nch) in rcsd[d]
            szx = df .* (ord .+ 1)
            x = rand(Float64, szx)

            nsolt = Rnsolt(df, ord, nch)
            rand!(nsolt)

            ana = createTransform(nsolt, Shapes.Separated())
            ya = analyze(ana, x)

            afs = analysiskernels(nsolt)
            myfilter(A, h) = begin
                ha = zero(A)
                ha[UnitRange.(1, size(h))...] = h
                real(ifft(fft(A).*fft(ha)))
            end
            offset = df .- 1

            sys = map(afs) do af
                fx = myfilter(x, af)
                dwfx = downsample(fx, df, offset)
                circshift(dwfx, (-1 .* fld.(ord,2)))
            end
            @test all(ya .≈ sys)
        end
    end

    @testset "SynthesizerKernel" begin
        for d in 1:length(rcsd), (df, ord, nch) in rcsd[d]

            nsolt = Rnsolt(df, ord, nch)
            rand!(nsolt)

            y = [ rand(Float64,((ord.+1) .* df)...) for p in 1:sum(nch) ]

            syn = createTransform(nsolt, Shapes.Separated())
            x = synthesize(syn, y)

            sfs = synthesiskernels(nsolt)
            myfilter(A, h) = begin
                ha = zero(A)
                ha[UnitRange.(1, size(h))...] = h
                real(ifft(fft(A).*fft(ha)))
            end
            offset = df .- 1

            subrxs = sum(map((yp,fp)->myfilter( MDCDL.upsample(yp, df), fp ), y, sfs))
            subrx = circshift(subrxs, -1 .* df .* cld.(ord,2))
            @test size(x) == size(subrx)
            @test subrx ≈ x
        end
    end

    @testset "Factorization" begin
        for d in 1:length(rcsd), (df, ord, nch) in rcsd[d]
            src = Rnsolt(df, ord, nch)
            dst = Rnsolt(df, ord, nch)
            rand!(src)

            (angs, mus) = getrotations(src)
            setrotations!(dst, angs, mus)

            @test src.W0 ≈ dst.W0 && src.U0 ≈ dst.U0

            foreach(src.Udks, dst.Udks) do srcUs, dstUs
                @test all(srcUs .≈ dstUs)
            end

            if istype2(src)
                foreach(src.Wdks, dst.Wdks) do srcWs, dstWs
                    @test all(srcWs .≈ dstWs)
                end
            end
        end
    end

end


nothing
