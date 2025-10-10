using SelfConsistentHartreeFock, SecondQuantizedAlgebra
using SelfConsistentHartreeFock.MatrixEquations
using SelfConsistentHartreeFock: @unpack
using LinearAlgebra
import SecondQuantizedAlgebra as SQA
using Symbolics

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables G::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + G * (a' * a' + a * a)

sys = HartreeFockSystem(H, [a], [κ])
collect_dict(sys.H)

p = Dict(G => 0.01, Δ => 0.0, K => 0.001, κ => 0.005)
problem = IterativeProblem(sys, p)
