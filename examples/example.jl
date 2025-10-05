using SelfConsistentHartreeFock, SecondQuantizedAlgebra
using Symbolics, SymbolicUtils
using UnPack

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables F::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a) / 2

sys = HartreeFockSystem(H, [a], [κ])
collect_dict(sys.H)
@unpack A, B = sys.dynamical_matrix

p = Dict(Δ => -0.01, K => 0.001, F => 0.01, κ => 0.001)
problem = IterativeProblem(sys, p)

using MatrixEquations, UnPack, LinearAlgebra

α = ComplexF64[1.0 + 1im, 0.0, 0.0]

@unpack M, D = problem.dynamical_matrix
M(α)
D(α)
eigvals(M(α))
C = lyapc(M(α), D(α)) # Solves M X + X M^† + D = 0
sum(diag(C)) # 0.9999999999999996 + 0.0im
