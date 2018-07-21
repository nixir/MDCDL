using Base.Test
using MDCDL

@testset "Multiscale" begin
    include("testsetGenerator.jl")
    include("randomInit.jl")

    ccsd1D = cnsoltValidConfigSet1D()
    ccsd2D = cnsoltValidConfigSet2D()
    ccsd3D = cnsoltValidConfigSet3D()

    ccsd = [ ccsd1D, ccsd2D, ccsd3D ]

    rcsd1D = rnsoltValidConfigSet1D()
    rcsd2D = rnsoltValidConfigSet2D()
    rcsd3D = rnsoltValidConfigSet3D()

    rcsd = [ rcsd1D, rcsd2D, rcsd3D ]

    srand(3923528829)

    @testset "ParallelFB" begin
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

                mspfb = Multiscale(pfb, lv)

                szx = rand(1:4) .* (df.^lv) .* (ord .+ 1)
                x = rand(dt, szx)

                y = analyze(mspfb, x)

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

                rx = synthesize(mspfb, y)

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

    @testset "CNSOLT" begin
        # output mode options for analyzer
        oms = [ :reshaped, :augumented ]
        for d in 1:length(ccsd), (df, ord, nch) in ccsd[d], lv in 1:3
            szx = (df.^lv) .* (ord .+ 1)
            nsolt = Cnsolt(df, ord, nch)
            randomInit!(nsolt)
            msnsolt = Multiscale(nsolt, lv)

            x = rand(Complex{Float64}, szx...)

            y = analyze(msnsolt, x)
            rx = synthesize(msnsolt, y)

            @test rx ≈ x

            foreach(oms) do om
                y = analyze(nsolt, x; outputMode = om)
                rx = synthesize(nsolt, y)

                @test size(x) == size(rx)
                @test rx ≈ x
            end
        end
    end
    @testset "RNSOLT" begin
        # output mode options for analyzer
        oms = [ :reshaped, :augumented ]
        for d in 1:length(ccsd), (df, ord, nch) in rcsd[d], lv in 1:3
            szx = (df.^lv) .* (ord .+ 1)
            nsolt = Rnsolt(df, ord, nch)
            randomInit!(nsolt)
            msnsolt = Multiscale(nsolt, lv)

            x = rand(Float64, szx...)

            y = analyze(msnsolt, x)
            rx = synthesize(msnsolt, y)

            @test rx ≈ x

            foreach(oms) do om
                y = analyze(nsolt, x; outputMode = om)
                rx = synthesize(nsolt, y)

                @test size(x) == size(rx)
                @test rx ≈ x
            end
        end
    end
end


nothing
