using SelfConsistentHartreeFock, SecondQuantizedAlgebra, Plots
using MatrixEquations, UnPack, LinearAlgebra
import SecondQuantizedAlgebra as SQA
using Symbolics

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables F::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a) / 2

sys = HartreeFockSystem(H, [a], [κ])
sys.mean_field_eom
A, Fs = SelfConsistentHartreeFock.construct_iterative_eom(sys)

p = Dict(F => 0.01, Δ => -0.01, K => 0.001, κ => 0.001)
problem = IterativeProblem(sys, p)

α0 = ComplexF64[rand(ComplexF64), 0.0, 0.0]
fixed_point(problem, α0)

Δsweep = range(-0.01, 0.03, 101)
results_up = parameter_sweep(problem, Δ, Δsweep, α0)
results_down = parameter_sweep(problem, Δ, reverse(Δsweep), α0)

amplitude_up = map(results_up) do result
    result.αs[1] |> norm
end
amplitude_down = map(results_down) do result
    result.αs[1] |> norm
end

fluctuation_up = map(results_up) do result
    result.ns[1]
end
fluctuation_down = map(results_down) do result
    result.ns[1]
end

anomalous_up = map(results_up) do result
    result.ms[1] |> norm
end
anomalous_down = map(results_down) do result
    result.ms[1] |> norm
end

plt1 = plot(Δsweep, amplitude_up; xlabel = "Detuning Δ", ylabel = "Amplitude", legend = false)
plot!(reverse(Δsweep)[1:55], amplitude_down)

plt2 = plot(Δsweep, fluctuation_up; xlabel = "Detuning Δ", ylabel = "Fluctuation", legend = false)
plot!(reverse(Δsweep)[1:55], fluctuation_down)

plt3 = plot(Δsweep, anomalous_up; xlabel = "Detuning Δ", ylabel = "Anomalous", legend = false)
plot!(reverse(Δsweep)[1:55], anomalous_down)

plot(plt1, plt2, plt3; layout = (3, 1), size=(500, 600))
