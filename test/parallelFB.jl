using Base.Test
using MDCDL

@testset "ParallelFB" begin
    include("testsetGenerator.jl")

    srand(3923528829)

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

    # @testset "Constructor" begin
    #
    #     maxDims = 3
    #     maxdfs = [ 3, 3, 3 ]
    #     maxchs = [ 10, 10, 10 ]
    #     maxpos = [ 2, 2, 2 ]
    #     defaultType = Float64
    #
    #     for d in 1:maxDims, crdf in CartesianRange(tuple(fill(maxdfs[d],d)...)), nch in 2:maxchs[d], crord in CartesianRange(tuple(fill(maxpos[d] .+ 1,d)...))
    #         df = crdf.I
    #         ord = crord.I .- 1
    #         szx = df .* (ord .+ 1)
    #
    #         if prod(df) > sum(nch)
    #             @test_throws ArgumentError Cnsolt(df, nch, ord)
    #             continue
    #         end
    #
    #         if isodd(sum(nch)) && any(isodd.(ord))
    #             @test_throws ArgumentError Cnsolt(df, nch, ord)
    #             continue
    #         end
    #
    #         nsolt = Cnsolt(df, nch, ord)
    #
    #         if iseven(sum(nch))
    #             @test isa(nsolt, Cnsolt{d,1,defaultType})
    #         else
    #             @test isa(nsolt, Cnsolt{d,2,defaultType})
    #         end
    #
    #     end
    # end

    @testset "MultiscaleAnalysisSynthesis" begin
        maxDims = 2
        for d in 1:maxDims, dt in [ Float64, Complex{Float64}]
            cfgs = vec([ (crdf.I, crord.I .- 1, nch, lv) for crdf in CartesianRange(tuple(fill(4,d)...)), crord in CartesianRange(tuple(fill(2+1,d)...)), nch in 2:10, lv in 1:3 ])

            subcfgs = randsubseq(cfgs, 30 / length(cfgs))

            for (df, ord, nch, lv) in subcfgs
                pfb = ParallelFB(dt, df, ord, nch)

                szFilter = df .* (ord .+ 1)
                afs = [ rand(dt, szFilter) for p in 1:sum(nch) ]
                sfs = [ rand(dt, szFilter) for p in 1:sum(nch) ]
                pfb.analysisFilters .= afs
                pfb.synthesisFilters .= sfs

                szx = rand(1:4) .* (df.^lv) .* (ord .+ 1)
                x = rand(dt, szx)

                y = analyze(pfb, x, lv)

                myfilter = (A, h) -> begin
                    ha = zeros(A)
                    ha[colon.(1,size(h))...] = h
                    if dt <: Real
                        real(ifft(fft(A).*fft(ha)))
                    else
                        ifft(fft(A).*fft(ha))
                    end
                end
                offset = df .- 1

                # check analyzer
                subx = x
                for idx = 1:lv-1
                    sys = map(afs) do af
                        fx = myfilter(subx, af)
                        dwfx = downsample(fx, df, offset)
                        circshift(dwfx, (-1 .* fld.(ord,2)))
                    end
                    if idx < lv
                        @test all(y[idx] .≈ sys[2:end])
                        subx = sys[1]
                    else
                        @test all(y[idx] .≈ sys)
                    end
                end

                rx = synthesize(pfb, y, lv)

                # check synthesizer
                suby = y[lv]
                for idx = lv:-1:1
                    fy = map(suby, sfs) do  yp, fp
                        upyp = MDCDL.upsample(yp, df)
                        myfilter( upyp, fp )
                    end
                    subrxs = sum(fy)
                    subrx = circshift(subrxs, -1 .* df .* cld.(ord,2))
                    if idx > 1
                        suby = [subrx, y[idx-1]... ]
                    else
                        @test size(rx) == size(subrx)
                        @test subrx ≈ rx
                    end
                end
            end
        end
    end
end


nothing
