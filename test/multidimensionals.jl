using Test
using MDCDL
using FFTW
using ImageFiltering
using Random
using LinearAlgebra

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

    @testset "permutated DCT" begin
        @testset "unitarity" begin
            for d = 1:5
                for n = 1:10
                    szx = rand(1:8, d)
                    x = rand(szx...)

                    A = MDCDL.permdctmtx(szx...)

                    @test A' * A ≈ I
                end
            end
        end
        @testset "symmetry" begin
            for d = 1:5
                for n = 1:10
                    szx = rand(1:8, d)
                    x = rand(szx...)

                    A = MDCDL.permdctmtx(szx...)

                    nch = size(A, 1)
                    Γ = diagm(0 => [ ones(cld(nch, 2)); -ones(fld(nch, 2)) ])
                    @test Γ * reverse(A, dims=2) ≈ A
                end
            end
        end
        # for d = 1:5
        #     for n = 1:10
        #         szx = rand(1:8,d)
        #         x = rand(szx...)
        #
        #         A = MDCDL.permdctmtx(szx...)
        #         @test sort(A*vec(x)) ≈ sort(vec(dct(x)/sqrt(prod(szx))))
        #     end
        # end
    end

    @testset "centered DFT" begin
        @testset "unitarity" begin
            for d = 1:5
                for n = 1:10
                    szx = rand(1:8, d)
                    x = rand(szx...)

                    A = MDCDL.cdftmtx(szx...)

                    @test A' * A ≈ I
                end
            end
        end
        @testset "symmetry" begin
            for d = 1:5
                for n = 1:10
                    szx = rand(1:8, d)
                    x = rand(szx...)

                    A = MDCDL.cdftmtx(szx...)

                    @test conj(reverse(A, dims=2)) ≈ A
                end
            end
        end
        # for d = 1:5
        #     for n = 1:10
        #         szx = rand(1:8,d)
        #         x = rand(szx...)
        #
        #         A = MDCDL.permdctmtx(szx...)
        #         @test sort(A*vec(x)) ≈ sort(vec(dct(x)/sqrt(prod(szx))))
        #     end
        # end
    end

    @testset "polyphase matrix" begin
        for d = 1:5
            for n = 1:10
                df = (rand(1:4, d)...,)
                szx = ((rand(1:3, d) .* df)...,)
                x = rand(szx...)

                pvx = MDCDL.mdarray2polyphase(x, df)

                @test size(pvx.data, 1) == prod(df)
                @test size(pvx.data, 2) == fld(length(x), prod(df))

                rx = MDCDL.polyphase2mdarray(pvx, df)
                @test rx == x
            end
        end
    end
end

nothing
