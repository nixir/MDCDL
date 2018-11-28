using Test
using MDCDL
using FFTW
using ImageFiltering
using Random

@testset "Multidimensionals" begin
    Random.seed!(39273529)

    @testset "representation of linear operator" begin
        @testset "FFT" begin
            for d = 1:5
                for n = 1:10
                    szx = rand(1:8,d)
                    x = rand(szx...)

                    A = MDCDL.representationmatrix(fft, szx...)
                    @test A*vec(x) ≈ vec(fft(x))
                end
            end
        end

        @testset "DCT" begin
            for d = 1:5
                for n = 1:10
                    szx = rand(1:8,d)
                    x = rand(szx...)

                    A = MDCDL.representationmatrix(dct, szx...)
                    @test A*vec(x) ≈ vec(dct(x))
                end
            end
        end

        @testset "imfilter" begin
            for d = 1:3
                for n = 1:10
                    szx = rand(1:8,d)
                    x = rand(szx...)
                    szker = rand.(UnitRange.(1, szx))
                    ker = centered(rand(szker...))

                    myfcn(t) = imfilter(t, ker, Pad(:circular))

                    A = MDCDL.representationmatrix(myfcn, szx...)
                    @test A*vec(x) ≈ vec(myfcn(x))
                end
            end
        end
    end

#     @testset "Constructor" begin
#
#         maxDims = 3
#         defaultType = Float64
#
#         # cfgset = [ (crdf.I, nch, crord.I .- 1) for crdf in CartesianIndices(tuple(fill(4,d)...)), nch in 2:20, crord in CartesianIndices(tuple(fill(6,d)...)) ]
#
#         for d in 1:maxDims
#             allcfgset = [ (crdf.I, crord.I .- 1, nch) for crdf in CartesianIndices(tuple(fill(4,d)...)), crord in CartesianIndices(tuple(fill(6+1,d)...)), nch in 2:20 ]
#             cfgset = randsubseq(vec(allcfgset), 100 / length(allcfgset))
#
#             for (df, ord, nch) in cfgset
#                 if prod(df) > sum(nch)
#                     @test_throws AssertionError Cnsolt(df, ord, nch)
#                     continue
#                 end
#
#                 if isodd(sum(nch)) && any(isodd.(ord))
#                     @test_throws AssertionError Cnsolt(df, ord, nch)
#                     continue
#                 end
#
#                 nsolt = Cnsolt(df, ord, nch)
#
#                 if iseven(sum(nch))
#                     @test istype1(nsolt) == true && istype2(nsolt) == false
#                 else
#                     @test istype1(nsolt) == false && istype2(nsolt) == true
#                 end
#             end
#         end
#     end
#
#     # @testset "DefaultValues"
#     #     for d in 1:length(ccsd), (df, nch, ord) in ccsd[d]
#     #         nsolt = Cnsolt(df, nch, ord)
#     #
#     #         df = 0
#     #     end
#     # end
#
#     @testset "FilterSymmetry" begin
#         for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
#             nsolt = Cnsolt(df, ord, nch)
#             rand!(nsolt)
#
#             afb = analysisbank(nsolt)
#             hsafb = nsolt.Φ' * afb
#
#             @test hsafb ≈ conj(reverse(hsafb; dims=2))
#         end
#     end
#
#     @testset "AnalysisSynthesis" begin
#         # output mode options for analyzer
#         oms = [ Shapes.Separated, Shapes.Combined, Shapes.Vec ]
#         for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
#             szx = df .* (ord .+ 1)
#             nsolt = Cnsolt(df, ord, nch)
#             rand!(nsolt)
#
#             x = rand(Complex{Float64}, szx...)
#
#             foreach(oms) do om
#                 nsop = createTransform(nsolt, om(size(x)))
#
#                 y = analyze(nsop, x)
#                 rx = synthesize(nsop, y)
#
#                 @test size(x) == size(rx)
#                 @test rx ≈ x
#             end
#         end
#     end
#
#     @testset "AnalyzerKernel" begin
#         for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
#             szx = df .* (ord .+ 1)
#             x = rand(Complex{Float64}, szx)
#
#             nsolt = Cnsolt(df, ord, nch)
#             rand!(nsolt)
#
#             ana = createTransform(nsolt, Shapes.Separated())
#             ya = analyze(ana, x)
#
#             afs = analysiskernels(nsolt)
#             myfilter(A, h) = begin
#                 ha = zero(A)
#                 ha[UnitRange.(1, size(h))...] = h
#                 ifft(fft(A).*fft(ha))
#             end
#             offset = df .- 1
#
#             sys = map(afs) do af
#                 fx = myfilter(x, af)
#                 dwfx = downsample(fx, df, offset)
#                 circshift(dwfx, (-1 .* fld.(ord,2)))
#             end
#             @test all(ya .≈ sys)
#         end
#     end
#
#     @testset "SynthesizerKernel" begin
#         for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
#
#             nsolt = Cnsolt(df, ord, nch)
#             rand!(nsolt)
#
#             y = [ rand(Complex{Float64},((ord.+1) .* df)...) for p in 1:sum(nch) ]
#
#             syn = createTransform(nsolt, Shapes.Separated())
#             x = synthesize(syn, y)
#
#             sfs = synthesiskernels(nsolt)
#             myfilter(A, h) = begin
#                 ha = zero(A)
#                 ha[UnitRange.(1, size(h))...] = h
#                 ifft(fft(A).*fft(ha))
#             end
#             offset = df .- 1
#
#             subrxs = sum(map((yp,fp)->myfilter( MDCDL.upsample(yp, df), fp ), y, sfs))
#             subrx = circshift(subrxs, -1 .* df .* cld.(ord,2))
#             @test size(x) == size(subrx)
#             @test subrx ≈ x
#         end
#     end
#
#     @testset "Factorization" begin
#         for d in 1:length(ccsd), (df, ord, nch) in ccsd[d]
#             src = Cnsolt(df, ord, nch)
#             dst = Cnsolt(df, ord, nch)
#             rand!(src)
#
#             (angs, mus) = getrotations(src)
#             setrotations!(dst, angs, mus)
#
#             @test src.V0 ≈ dst.V0
#             foreach(src.Wdks, dst.Wdks) do srcWs, dstWs
#                 @test all(srcWs .≈ dstWs)
#             end
#             foreach(src.Udks, dst.Udks) do srcUs, dstUs
#                 @test all(srcUs .≈ dstUs)
#             end
#
#             if istype1(src)
#                 foreach(src.θdks, dst.θdks) do srcθs, dstθs
#                     @test all(srcθs .≈ dstθs)
#                 end
#             else
#                 foreach(src.θ1dks, dst.θ1dks) do srcθ1s, dstθ1s
#                     @test all(srcθ1s .≈ dstθ1s)
#                 end
#                 foreach(src.Ŵdks, dst.Ŵdks) do srcŴs, dstŴs
#                     @test all(srcŴs .≈ dstŴs)
#                 end
#                 foreach(src.Ûdks, dst.Ûdks) do srcÛs, dstÛs
#                     @test all(srcÛs .≈ dstÛs)
#                 end
#                 foreach(src.θ2dks, dst.θ2dks) do srcθ2s, dstθ2s
#                     @test all(srcθ2s .≈ dstθ2s)
#                 end
#             end
#         end
#     end
end

nothing
