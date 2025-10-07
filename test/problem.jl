module test_iterative_problem

using SelfConsistentHartreeFock, SecondQuantizedAlgebra, Symbolics
using MatrixEquations, UnPack, LinearAlgebra
import SecondQuantizedAlgebra as SQA
using Test

@testset "duffing" begin
    h = FockSpace(:cavity)

    @qnumbers a::Destroy(h)
    @variables F::Real Δ::Real K::Real κ::Real

    H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a)

    sys = HartreeFockSystem(H, [a], [κ])
    A, Fs = SelfConsistentHartreeFock.construct_iterative_eom(sys)
    vars = vcat(sys.operators, SQA.adjoint.(sys.operators)) .|> SQA.average

    @test isequal(expand.(A*vars) + Fs, sys.mean_field_eom)
end
end
