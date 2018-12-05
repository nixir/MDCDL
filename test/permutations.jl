using Test
using MDCDL
using Random

@testset "Permutation" begin
    Random.seed!(39328582)
    @testset "inversion" begin
        for d = 1:5
            for n = 1:20
                df = (rand(1:4, d)...,)
                x = rand((4 .* df)...)

                pvx = MDCDL.mdarray2polyphase(x, df)

                nBlocks = pvx.nBlocks
                expctd = copy(pvx.data)
                actual = MDCDL.irotatedimspv(MDCDL.rotatedimspv(pvx.data, nBlocks[1]), nBlocks[1])
                @test actual == expctd
            end
        end
    end

    @testset "cyclic group" begin
        for d = 1:5
            for n = 1:20
                df = (rand(1:4, d)...,)
                x = rand((4 .* df)...)

                expctd = copy(x)

                pvx = MDCDL.mdarray2polyphase(x, df)
                pvdata = pvx.data
                nBlocks = pvx.nBlocks

                for blk in nBlocks
                    pvdata = MDCDL.rotatedimspv(pvdata, blk)
                end

                rpvx = MDCDL.PolyphaseVector(pvdata, nBlocks)
                actual = MDCDL.polyphase2mdarray(rpvx, df)
                @test actual == expctd
            end
        end
    end


    @testset "pv vs md" begin
        for d = 1:5
            for n = 1:20
                df = (fill(1,d)...,)
                szx = rand(1:16, d)
                x = rand(szx...)

                expctd = permutedims(x, [collect(2:d)..., 1])

                pvx = MDCDL.mdarray2polyphase(x, df)
                nBlocks = pvx.nBlocks

                permdata = MDCDL.rotatedimspv(pvx.data, nBlocks[1])
                permpvx = PolyphaseVector(permdata, ([nBlocks[2:d]..., nBlocks[1]]...,))

                actual = MDCDL.polyphase2mdarray(permpvx, df)
                @test actual == expctd
            end
        end
    end

    @testset "PolyphaseVector Type" begin
        for d = 1:5
            for n = 1:20
                df = (fill(1,d)...,)
                szx = rand(1:16, d)
                x = rand(szx...)
                nrot = rand(1:d)

                expctd = permutedims(x, circshift(collect(1:d), -nrot))

                pvx = MDCDL.mdarray2polyphase(x, df)
                permpvx = foldl(1:nrot, init = pvx) do prevpvx, k
                    MDCDL.rotatedimspv(prevpvx)
                end
                actual = MDCDL.polyphase2mdarray(permpvx, df)
                @test actual == expctd
            end
        end
    end
end
