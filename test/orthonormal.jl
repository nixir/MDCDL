using Test
using MDCDL
using Random
using LinearAlgebra

@testset "Orthonormal matrices" begin
    Random.seed!(883290234387)

    @testset "factorization" begin
        for p = 2:32
            for n = 1:5
                # generate random orthonormal matrix
                A = qr(rand(p, p)).Q |> Matrix

                θ, μ = MDCDL.mat2rotations(A)

                @test length(θ) == fld(p * (p-1), 2)
                @test length(μ) == p
                @test isreal(θ)
                @test isreal(μ)
                @test all(abs.(μ) .≈ 1)
            end
        end
    end

    @testset "generation" begin
        for p = 2:32
            for n = 1:5
                θ = pi * (rand(fld(p * (p-1), 2)) .- 0.5)
                μ = rand([-1.0, 1.0], p)

                A = MDCDL.rotations2mat(θ, μ)

                @test all(size(A) .== (p, p))
                @test isreal(A)
                @test A' * A ≈ I
            end
        end
    end

    @testset "reconstruction" begin
        for p = 2:32
            for n = 1:5
                A = qr(rand(p, p)).Q |> Matrix

                θ, μ = MDCDL.mat2rotations(A)
                rA = MDCDL.rotations2mat(θ, μ)

                @test rA ≈ A
            end
        end
    end

    @testset "non-destructivity" begin
        for p = 2:32
            for n = 1:5
                A = qr(rand(p, p)).Q |> Matrix

                expctdA = copy(A)
                θ, μ = MDCDL.mat2rotations(A)
                @test A == expctdA

                expctdθ = copy(θ)
                expctdμ = copy(μ)
                rA = MDCDL.rotations2mat(θ, μ)
                @test θ == expctdθ
                @test μ == expctdμ
            end
        end
    end
end
