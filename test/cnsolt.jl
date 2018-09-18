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

        # cfgset = [ (crdf.I, nch, crord.I .- 1) for crdf in CartesianIndices(tuple(fill(4,d)...)), nch in 2:20, crord in CartesianIndices(tuple(fill(6,d)...)) ]

        for d in 1:maxDims
            allcfgset = [ (crdf.I, crord.I .- 1, nch) for crdf in CartesianIndices(tuple(fill(4,d)...)), crord in CartesianIndices(tuple(fill(6+1,d)...)), nch in 2:20 ]
            cfgset = randsubseq(vec(allcfgset), 100 / length(allcfgset))

            for (df, ord, nch) in cfgset
                if prod(df) > sum(nch)
                    @test_throws ArgumentError Cnsolt(df, ord, nch)
                    continue
                end

                if isodd(sum(nch)) && any(isodd.(ord))
                    @test_throws ArgumentError Cnsolt(df, ord, nch)
                    continue
                end

                nsolt = Cnsolt(df, ord, nch)

                # if iseven(sum(nch))
                #     @test isa(nsolt, Cnsolt{defaultType,d,:TypeI})
                # else
                #     @test isa(nsolt, Cnsolt{defaultType,d,:TypeII})
                # end
                if iseven(sum(nch))
                    @test istype1(nsolt) == true && istype2(nsolt) == false
                else
                    @test istype1(nsolt) == false && istype2(nsolt) == true
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
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
            nsolt = Cnsolt(df, ord, nch)
            rand!(nsolt)

            afb = MDCDL.analysisbank(nsolt)
            hsafb = nsolt.symmetry' * afb

            @test hsafb ≈ conj(reverse(hsafb; dims=2))
        end
    end

    @testset "AnalysisSynthesis" begin
        # output mode options for analyzer
        oms = [ Shapes.Default(), Shapes.Augumented(), Shapes.Vec() ]
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
            szx = df .* (ord .+ 1)
            nsolt = Cnsolt(df, ord, nch)
            rand!(nsolt)

            x = rand(Complex{Float64}, szx...)

            foreach(oms) do om
                nsop = createOperator(nsolt, x; shape=om)

                y = analyze(nsop, x)
                rx = synthesize(nsop, y)

                @test size(x) == size(rx)
                @test rx ≈ x
            end
        end
    end

    @testset "AnalyzerKernel" begin
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
            szx = df .* (ord .+ 1)
            x = rand(Complex{Float64}, szx)

            nsolt = Cnsolt(df, ord, nch)
            rand!(nsolt)

            ana = createOperator(nsolt, x; shape=Shapes.Default())
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
            @test all(ya .≈ sys)
        end
    end

    @testset "SynthesizerKernel" begin
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]

            nsolt = Cnsolt(df, ord, nch)
            rand!(nsolt)

            y = [ rand(Complex{Float64},((ord.+1) .* df)...) for p in 1:sum(nch) ]

            syn = createOperator(nsolt, ord.+1; shape=Shapes.Default())
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
            @test subrx ≈ x
        end
    end

    @testset "Factorization" begin
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
            src = Cnsolt(df, ord, nch)
            dst = Cnsolt(df, ord, nch)
            rand!(src)

            (angs, mus) = getrotations(src)
            setrotations!(dst, angs, mus)

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
