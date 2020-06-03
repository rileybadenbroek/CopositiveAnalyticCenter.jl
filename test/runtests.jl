using CopositiveAnalyticCenter, Test, DelimitedFiles

@testset "6 × 6 random matrices" begin
    for i in 1:10
        A = readdlm("randmat_6x6_v$i.txt")
        @test !is_completely_positive(A)
    end
end
