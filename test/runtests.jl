using Test

@testset "SelfConsistentHartreeFock.jl" begin
    @testset "displacement" include("displacement.jl")
    @testset "Hartree-Fock_approximation" include("hartree-fock.jl")
    @testset "symbolic utils" include("symbolic_utils.jl")
    @testset "lyapunov solver" include("lyapunov_equation.jl")
end
