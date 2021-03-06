using Test
using MDCDL
using FFTW
using Random

@testset "CNSOLT" begin
    include("testsetGenerator.jl")

    ccsd1D = cnsoltValidConfigSet1D()
    ccsd2D = cnsoltValidConfigSet2D()
    ccsd3D = cnsoltValidConfigSet3D()

    ccsd = [ ccsd1D, ccsd2D, ccsd3D ]

    Random.seed!(392875929)

    @testset "Constructor" begin

        maxDims = 3
        defaultType = Float64

        # cfgset = [ (crdf.I, nch, crord.I .- 1) for crdf in CartesianIndices(tuple(fill(4,d)...)), nch in 2:20, crord in CartesianIndices(tuple(fill(6,d)...)) ]

        for d in 1:maxDims
            allcfgset = [ (crdf.I, crord.I .- 1, nch) for crdf in CartesianIndices(tuple(fill(4,d)...)), crord in CartesianIndices(tuple(fill(6+1,d)...)), nch in 2:20 ]
            cfgset = randsubseq(vec(allcfgset), 100 / length(allcfgset))

            for (df, ord, nch) in cfgset
                if prod(df) > sum(nch)
                    @test_throws AssertionError Cnsolt(df, ord, nch)
                    continue
                end

                if isodd(sum(nch)) && any(isodd.(ord))
                    @test_throws AssertionError Cnsolt(df, ord, nch)
                    continue
                end

                nsolt = Cnsolt(df, ord, nch)

                if iseven(sum(nch))
                    @test nsolt isa CnsoltTypeI
                    @test istype1(nsolt) == true && istype2(nsolt) == false
                else
                    @test nsolt isa CnsoltTypeII
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
                    A = MDCDL.cdftmtx(df...)
                    expctd = A*vec(x)

                    nsolt = Cnsolt(df, ord, nch)
                    pvx = reshape(x, prod(df), :)
                    y = analyze(nsolt, pvx, (fill(1,d)...,))
                    actual = reshape(y, prod(df))

                    @test actual ??? expctd
                end
            end
        end

        @testset "over-sampled" begin
            for d in 1:3
                for n = 1:20
                    df = (rand(1:4, d)...,)
                    ord = (fill(0, d)...,)
                    nch = prod(df) + rand(0:6)

                    x = rand(df...)
                    A = MDCDL.cdftmtx(df...)
                    expctd = A*vec(x)

                    nsolt = Cnsolt(df, ord, nch)
                    pvx = reshape(x, prod(df), :)
                    y = analyze(nsolt, pvx, (fill(1,d)...,))
                    actual = reshape(y[1:prod(df),:], prod(df))

                    @test actual ??? expctd
                end
            end
        end
    end

    @testset "FilterSymmetry" begin
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
            nsolt = Cnsolt(df, ord, nch)
            rand!(nsolt)

            afb = analysisbank(nsolt)
            hsafb = nsolt.??' * afb

            @test hsafb ??? conj(reverse(hsafb; dims=2))
        end
    end

    @testset "AnalysisSynthesis" begin
        # output mode options for analyzer
        oms = [ Shapes.Separated, Shapes.Combined, Shapes.Vec ]
        @testset "NormalOrder" begin
            for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
                szx = df .* (ord .+ 1)
                nsolt = Cnsolt(df, ord, nch)
                rand!(nsolt)

                x = rand(Complex{Float64}, szx...)

                foreach(oms) do om
                    nsop = createTransform(nsolt, om(size(x)))

                    y = analyze(nsop, x)
                    rx = synthesize(nsop, y)

                    @test size(x) == size(rx)
                    @test rx ??? x
                end
            end
        end
        @testset "DimensionPermutation" begin
            for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
                szx = df .* (ord .+ 1)
                nsolt = Cnsolt(df, ord, nch, perm=(randperm(d)...,))
                rand!(nsolt)

                x = rand(Complex{Float64}, szx...)

                foreach(oms) do om
                    nsop = createTransform(nsolt, om(size(x)))

                    y = analyze(nsop, x)
                    rx = synthesize(nsop, y)

                    @test size(x) == size(rx)
                    @test rx ??? x
                end
            end
        end
    end

    @testset "AnalyzerKernel" begin
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
            szx = df .* (ord .+ 1)
            x = rand(Complex{Float64}, szx)

            nsolt = Cnsolt(df, ord, nch)
            rand!(nsolt)

            ana = createTransform(nsolt, Shapes.Separated())
            ya = analyze(ana, x)

            afs = analysiskernels(nsolt)
            myfilter(A, h) = begin
                ha = zero(A)
                ha[UnitRange.(1, size(h))...] = h
                ifft(fft(A).*fft(ha))
            end
            offset = df .- 1

            sys = map(afs) do af
                fx = myfilter(x, af)
                dwfx = downsample(fx, df, offset)
                circshift(dwfx, (-1 .* fld.(ord,2)))
            end
            @test all(ya .??? sys)
        end
    end

    @testset "SynthesizerKernel" begin
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]

            nsolt = Cnsolt(df, ord, nch)
            rand!(nsolt)

            y = [ rand(Complex{Float64},((ord.+1) .* df)...) for p in 1:sum(nch) ]

            syn = createTransform(nsolt, Shapes.Separated())
            x = synthesize(syn, y)

            sfs = synthesiskernels(nsolt)
            myfilter(A, h) = begin
                ha = zero(A)
                ha[UnitRange.(1, size(h))...] = h
                ifft(fft(A).*fft(ha))
            end
            offset = df .- 1

            subrxs = sum(map((yp,fp)->myfilter( MDCDL.upsample(yp, df), fp ), y, sfs))
            subrx = circshift(subrxs, -1 .* df .* cld.(ord,2))
            @test size(x) == size(subrx)
            @test subrx ??? x
        end
    end

    @testset "Factorization" begin
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
            src = Cnsolt(df, ord, nch)
            dst = Cnsolt(df, ord, nch)
            rand!(src)

            (angs, mus) = getrotations(src)
            setrotations!(dst, angs, mus)

            @test src.V0 ??? dst.V0
            foreach(src.Wdks, dst.Wdks) do srcWs, dstWs
                @test all(srcWs .??? dstWs)
            end
            foreach(src.Udks, dst.Udks) do srcUs, dstUs
                @test all(srcUs .??? dstUs)
            end

            if istype1(src)
                foreach(src.??dks, dst.??dks) do src??s, dst??s
                    @test all(src??s .??? dst??s)
                end
            else
                foreach(src.??1dks, dst.??1dks) do src??1s, dst??1s
                    @test all(src??1s .??? dst??1s)
                end
                foreach(src.W??dks, dst.W??dks) do srcW??s, dstW??s
                    @test all(srcW??s .??? dstW??s)
                end
                foreach(src.U??dks, dst.U??dks) do srcU??s, dstU??s
                    @test all(srcU??s .??? dstU??s)
                end
                foreach(src.??2dks, dst.??2dks) do src??2s, dst??2s
                    @test all(src??2s .??? dst??2s)
                end
            end
        end
    end
end

nothing
