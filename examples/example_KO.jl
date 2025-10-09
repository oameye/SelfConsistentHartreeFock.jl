using SelfConsistentHartreeFock, SecondQuantizedAlgebra, Plots
import SelfConsistentHartreeFock as SCHF
using MatrixEquations, UnPack, LinearAlgebra
import SecondQuantizedAlgebra as SQA
using Symbolics

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables F::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a) / 2

sys = HartreeFockSystem(H, [a], [κ])

p = Dict(F => 0.01, Δ => -0.01, K => 0.001, κ => 0.001)
problem = IterativeProblem(sys, p)

α0 = ComplexF64[rand(ComplexF64), 0.0, 0.0]
fixed_point(problem, α0)

alg_kwargs = (;Algorithm = :Anderson,
                ConvergenceMetricThreshold= 1e-10, MaxIter = 1000, MaxM = 10, ExtrapolationPeriod = 7, Dampening = 1.0)
Δsweep = range(-0.01, 0.03, 101)
results_up = parameter_sweep(problem, Δ, Δsweep, α0)
results_down = parameter_sweep(problem, Δ, reverse(Δsweep), α0; Dampening = 0.5)

amplitude_up = map(results_up) do result
    v = result.FixedPoint_
    if v === missing
        return NaN
    end
    SCHF.as_complex_view(v)[1] |> norm
end
amplitude_down = map(results_down) do result
    v = result.FixedPoint_
    if ismissing(v)
        return NaN
    end
    SCHF.as_complex_view(v)[1] |> norm
end

fluctuation_up = map(results_up) do result
    v = result.FixedPoint_
    if v === missing
        return NaN
    end
    SCHF.as_complex_view(v)[2] |> norm
end
fluctuation_down = map(results_down) do result
    v = result.FixedPoint_
    if v === missing
        return NaN
    end
    SCHF.as_complex_view(v)[2] |> norm
end
anomalous_up = map(results_up) do result
    v = result.FixedPoint_
    if v === missing
        return NaN
    end
    SCHF.as_complex_view(v)[3] |> norm
end
anomalous_down = map(results_down) do result
    v = result.FixedPoint_
    if v === missing
        return NaN
    end
    SCHF.as_complex_view(v)[3] |> norm
end;

plt1 = plot(Δsweep, amplitude_up; xlabel = "Detuning Δ", ylabel = "Amplitude", legend = false)
plot!(reverse(Δsweep), amplitude_down)

plt2 = plot(Δsweep, fluctuation_up; xlabel = "Detuning Δ", ylabel = "Fluctuation", legend = false)
plot!(reverse(Δsweep), fluctuation_down)

plt3 = plot(Δsweep, anomalous_up; xlabel = "Detuning Δ", ylabel = "Anomalous", legend = false)
plot!(reverse(Δsweep), anomalous_down)

plot(plt1, plt2, plt3; layout = (3, 1), size=(500, 600))
