using SecondQuantizedAlgebra, Symbolics, Plots
import HarmonicSteadyState as HSS
import QuantumCumulants as QC
import SelfConsistentHartreeFock as SCHF
import SecondQuantizedAlgebra as SQA
using MatrixEquations

Δsweep = range(-0.01, 0.03, 101)
Kval = 0.01
Fval = 0.002
κval = 0.001

h = QC.FockSpace(:cavity)

QC.@qnumbers b::Destroy(h)

@variables Δ::Real K::Real F::Real κ::Real
param = [Δ, K, F, κ]

H = -Δ * b' * b + K * (b'^2 * b^2) + F * (b' + b)
ops = [b, b']

eqs = QC.complete(QC.meanfield(ops, H, [b]; rates=[κ], order=1))

fixed = (K => Kval, F => Fval, κ => κval)
varied = (Δ => Δsweep)
problem = HSS.HomotopyContinuationProblem(eqs, param, varied, fixed)

result = get_steady_states(problem, HSS.TotalDegree())
branches_real = HSS.get_branches(result, "bᵣ"; class=["stable"], not_class=[])
branches_imag = HSS.get_branches(result, "bᵢ"; class=["stable"], not_class=[])
meanfield_sol = @. branches_real[1] + im * branches_imag[1]

sys = SCHF.HartreeFockSystem(H, [b], [κ])

sys.mean_field_eom
sys.dynamical_matrix.A
sys.dynamical_matrix.B

function KO!(du, u, p)
    Δ, K, F, κ = p
    α, n, m = u

    du[1] =
        0.5 * F - conj(α) * Δ +
        2 * K * conj(m) * α +
        4 * K * n * conj(α) +
        2 * K * (conj(α)^2) * α

    A = -Δ + 4 * K * n + 4 * K * abs(α)^2
    B = 2(K * m + K * (α^2))
    M = [
        (-im * A-κ / 2) (-im*B)
        (im*conj.(B))   (im * A-κ / 2)
    ]
    D = [κ 0.0; 0.0 0.0]
    C = lyapc(M, D)
    du[2] = C[2, 2]
    du[3] = C[1, 2]

    return nothing
end

using NonlinearSolve

α0 = ComplexF64[meanfield_sol[1], 0, 0]
p = [Δsweep[1], Kval, Fval, κval]
prob = NonlinearProblem(KO!, α0, p)

sol = solve(prob, NewtonRaphson(), p)
