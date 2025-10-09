using SelfConsistentHartreeFock, SecondQuantizedAlgebra
using SelfConsistentHartreeFock.MatrixEquations
using SelfConsistentHartreeFock: @unpack
using LinearAlgebra
import SecondQuantizedAlgebra as SQA
using Symbolics

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables G::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + G * (a'*a' + a*a)

sys = HartreeFockSystem(H, [a], [κ])

collect_dict(sys.H)
sys.operators |> display
sys.correlators |> display

sys.dynamical_matrix.A |> display
sys.dynamical_matrix.B |> display

p = Dict(G => 0.01, Δ => 0.0, K => 0.001, κ => 0.005)
problem = IterativeProblem(sys, p)
α = ComplexF64[1.0+1im, 0.0, 0.0]
@unpack M, D = problem.dynamical_matrix;
_M = M(α)
_D = D(α)
_M |> display
_D
eigvals(_M) |> sum
Dampening = 1.0
MaxIter = 1_000
ConvergenceMetricThreshold = 1e-3
alg_kwargs = (;Algorithm = :Anderson, Dampening, MaxIter, ConvergenceMetricThreshold)

α0 = ComplexF64[rand(ComplexF64), 0.0, 0.0]
fixed_point(problem, α0)
Δsweep = range(-0.01, 0.03, 101)


results_up = parameter_sweep(problem, Δ, Δsweep, α0; alg_kwargs...);
results_down = parameter_sweep(problem, Δ, reverse(Δsweep), α0; alg_kwargs...);

results_up[1]

amplitude_up = map(results_up) do result
    result[SQA.average(a)] |> norm
end
amplitude_down = map(results_down) do result
    result[SQA.average(a)] |> norm
end

fluctuation_up = map(results_up) do result
    result[SQA.average(a'*a)] |> norm
end
fluctuation_down = map(results_down) do result
    result[SQA.average(a'*a)] |> norm
end
anomalous_up = map(results_up) do result
    result[SQA.average(a*a)] |> norm
end
anomalous_down = map(results_down) do result
    result[SQA.average(a*a)] |> norm
end;

amplitude_up
using Plots
l = length(amplitude_down)
plt1 = plot(Δsweep, amplitude_up; xlabel = "Detuning Δ", ylabel = "Amplitude", legend = false)
plot!(reverse(Δsweep)[1:l], amplitude_down)

plt2 = plot(Δsweep, fluctuation_up; xlabel = "Detuning Δ", ylabel = "Fluctuation", legend = false)
plot!(reverse(Δsweep)[1:l], fluctuation_down)

plt3 = plot(Δsweep, anomalous_up; xlabel = "Detuning Δ", ylabel = "Anomalous", legend = false)
plot!(reverse(Δsweep)[1:l], anomalous_down)

plot(plt1, plt2, plt3; layout = (3, 1), size=(500, 600))
