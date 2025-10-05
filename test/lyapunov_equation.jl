module test_lyapunov

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

    p = Dict(F => 1.0, Δ => 0.0, K => 1.0, κ => 1.0)
    problem = IterativeProblem(sys, p)

    α = ComplexF64[1.0 + 0.0im, 0.0, 0.0]
    @unpack M, D = problem.dynamical_matrix
    _M = M(α)
    _D = D(α)

    @test _M == [-0.5-4.0im   0.0-2.0im; 0.0+2.0im -0.5+4.0im]
    @test _D == [1.0 0.0; 0.0 0]

    @test sum(eigvals(_M)) ≈ -1.0 + 0.0im

    _C1 = lyap(_M, _D)
    _C2 = lyapc(_M, _D)
    @test _C1 ≈ _C2

    # Hermiticity
    @test _C2[1, 2] ≈ conj(_C2[2, 1])

    # Bosonic commutation relations
    @test _C2[1, 1] - _C2[2, 2] ≈ 1.0

    n = real(_C2[2, 2])
    m = _C2[1, 2]
    @test n * (n + 1) > abs(m)^2 # uncertainty relation
end
end
