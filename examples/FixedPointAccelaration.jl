using SelfConsistentHartreeFock, SecondQuantizedAlgebra, Plots
import SecondQuantizedAlgebra as SQA
using Symbolics, LinearAlgebra

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables F::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a) / 2

sys = HartreeFockSystem(H, [a], [κ])
sys.mean_field_eom
A, Fs = SelfConsistentHartreeFock.construct_iterative_eom(sys)

p = Dict(F => 0.01, Δ => -0.01, K => 0.001, κ => 0.01)
problem = IterativeProblem(sys, p)

α0 = ComplexF64[rand(ComplexF64), 0.0, 0.0]
picard_sol = fixed_point(
    problem, α0; solver=:picard, maxiters=2000, damping=0.5, rtol=1e-10
)
Dampening = 0.2
MaxIter = 10000
ConvergenceMetricThreshold = 1e-12
fpa_sol = fixed_point(problem, α0; Dampening, MaxIter, ConvergenceMetricThreshold)

Δsweep = range(-0.01, 0.03, 501)
results_up_picard = parameter_sweep(
    problem, Δ, Δsweep, α0; solver=:picard, maxiters=2000, damping=0.5, rtol=1e-10
)
results_up_fpa = parameter_sweep(problem, Δ, Δsweep, α0; Dampening, MaxIter, ConvergenceMetricThreshold)
results_down_picard = parameter_sweep(
    problem, Δ, reverse(Δsweep), α0; solver=:picard, maxiters=2000, damping=0.5, rtol=1e-10
)
results_down_fpa = parameter_sweep(problem, Δ, reverse(Δsweep), α0; Dampening, MaxIter, ConvergenceMetricThreshold)

function summarize(results)
    amplitude = map(results) do result
        norm(result.αs[1])
    end
    fluctuation = map(results) do result
        result.ns[1]
    end
    anomalous = map(results) do result
        norm(result.ms[1])
    end
    for collection in (amplitude, fluctuation, anomalous)
        for idx in eachindex(Δsweep)
            a = get(collection, idx, missing)
            if ismissing(a)
                push!(collection, NaN)
            end
        end
    end
    return amplitude, fluctuation, anomalous
end

amp_up_picard, fluct_up_picard, anom_up_picard = summarize(results_up_picard)
amp_up_fpa, fluct_up_fpa, anom_up_fpa = summarize(results_up_fpa)

amp_down_picard, fluct_down_picard, anom_down_picard = summarize(results_down_picard)
amp_down_fpa, fluct_down_fpa, anom_down_fpa = summarize(results_down_fpa)

plt1 = plot(Δsweep, amp_up_fpa; xlabel="Detuning Δ", ylabel="Amplitude")
plot!(plt1, Δsweep, amp_up_picard; label="Picard (up)")
plot!(plt1, reverse(Δsweep), amp_down_fpa; label="FPA (down)")
plot!(plt1, reverse(Δsweep), amp_down_picard; label="Picard (down)")

plt2 = plot(Δsweep, fluct_up_fpa; xlabel="Detuning Δ", ylabel="Fluctuation")
plot!(plt2, Δsweep, fluct_up_picard; label="Picard (up)")
plot!(plt2, reverse(Δsweep), fluct_down_fpa; label="FPA (down)")
plot!(plt2, reverse(Δsweep), fluct_down_picard; label="Picard (down)")

plt3 = plot(Δsweep, anom_up_fpa; xlabel="Detuning Δ", ylabel="Anomalous")
plot!(plt3, Δsweep, anom_up_picard; label="Picard (up)")
plot!(plt3, reverse(Δsweep), anom_down_fpa; label="FPA (down)")
plot!(plt3, reverse(Δsweep), anom_down_picard; label="Picard (down)")

plot(plt1, plt2, plt3; layout=(3, 1), size=(500, 600))
