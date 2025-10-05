using Test

@testset "SelfConsistentHartreeFock.jl" begin
    @testset "displacement" include("displacement.jl")
    @testset "Hartree-Fock_approximation" include("hartree-fock.jl")
end
