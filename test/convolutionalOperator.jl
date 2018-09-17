using Test
using MDCDL
using FFTW
using Random

@testset "convolutionalOperator" begin
    include("testsetGenerator.jl")

    Random.seed!(3923528829)

    @testset "Analysis" begin
        maxDims = 2
        for d in 1:maxDims, dt in [ Float64, Complex{Float64} ]
            cfgs = vec([ (crdf.I, crord.I .- 1, nch) for crdf in CartesianIndices(tuple(fill(4,d)...)), crord in CartesianIndices(tuple(fill(2+1,d)...)), nch in 2:10 ])

            subcfgs = randsubseq(cfgs, 30 / length(cfgs))

            for (df, ord, nch) in subcfgs
                szFilter = df .* (ord .+ 1)
                afs = [ rand(dt, szFilter) for p in 1:sum(nch) ]
                szx = rand(1:4) .* df .* (ord .+ 1)
                ca = ConvolutionalOperator(ParallelFilters(df, ord, nch, afs), szx)

                x = rand(dt, szx)
                y = analyze(ca, x)

                myfilter(A, h) = begin
                    ha = zero(A)
                    ha[UnitRange.(1, size(h))...] = h
                    if dt <: Real
                        real(ifft(fft(A).*fft(ha)))
                    else
                        ifft(fft(A).*fft(ha))
                    end
                end
                offset = df .- 1

                sys = map(afs) do af
                    fx = myfilter(x, af)
                    dwfx = downsample(fx, df, offset)
                    circshift(dwfx, (-1 .* fld.(ord,2)))
                end
                @test all(y .≈ sys)
            end
        end
    end

    @testset "Synthesis" begin
        maxDims = 2
        for d in 1:maxDims, dt in [ Float64, Complex{Float64} ]
            cfgs = vec([ (crdf.I, crord.I .- 1, nch) for crdf in CartesianIndices(tuple(fill(4,d)...)), crord in CartesianIndices(tuple(fill(2+1,d)...)), nch in 2:10 ])

            subcfgs = randsubseq(cfgs, 30 / length(cfgs))

            for (df, ord, nch) in subcfgs
                szFilter = df .* (ord .+ 1)
                sfs = [ rand(dt, szFilter) for p in 1:sum(nch) ]
                szy = rand(1:4) .* (ord .+ 1)
                # cs = ConvolutionalOperator(sfs, szy, df, ord, nch)
                cs = ConvolutionalOperator(ParallelFilters(df, ord, nch, sfs), szy)

                y = [ rand(dt, szy) for p in 1:sum(nch) ]
                rx = synthesize(cs, y)

                myfilter(A, h) = begin
                    ha = zero(A)
                    ha[UnitRange.(1, size(h))...] = h
                    if dt <: Real
                        real(ifft(fft(A).*fft(ha)))
                    else
                        ifft(fft(A).*fft(ha))
                    end
                end
                # check synthesizer
                fy = map(y, sfs) do  yp, fp
                    upyp = MDCDL.upsample(yp, df)
                    myfilter( upyp, fp )
                end
                subrxs = sum(fy)
                subrx = circshift(subrxs, -1 .* df .* cld.(ord,2))

                @test subrx ≈ rx
            end
        end
    end
end


nothing
