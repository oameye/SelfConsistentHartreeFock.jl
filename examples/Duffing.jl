using SelfConsistentHartreeFock, SecondQuantizedAlgebra, Plots
using FixedPointAccelerationNext
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
sol = fixed_point(
    problem, α0; solver=:picard, maxiters=10_000, damping=0.5, rtol=1e-10
)

Δsweep = range(-0.01, 0.03, 501)
results_up = parameter_sweep(
    problem, Δ, Δsweep, α0; solver=:picard, maxiters=10_000, damping=0.5, rtol=1e-10
)
results_down = parameter_sweep(
    problem, Δ, reverse(Δsweep), α0; solver=:picard, maxiters=10_000, damping=0.5, rtol=1e-10
)

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

amp_up, fluct_up, anom_up = summarize(results_up)
amp_down, fluct_down, anom_down = summarize(results_down)

plt1 = plot(;xlabel="Detuning Δ", ylabel="Amplitude")
plot!(plt1, Δsweep, amp_up; label="Picard (up)")
plot!(plt1, reverse(Δsweep), amp_down; label="Picard (down)")

plt2 = plot(; xlabel="Detuning Δ", ylabel="Fluctuation")
plot!(plt2, Δsweep, fluct_up; label="Picard (up)")
plot!(plt2, reverse(Δsweep), fluct_down; label="Picard (down)")

plt3 = plot(; xlabel="Detuning Δ", ylabel="Anomalous")
plot!(plt3, Δsweep, anom_up; label="Picard (up)")
plot!(plt3, reverse(Δsweep), anom_down; label="Picard (down)")

plot(plt1, plt2, plt3; layout=(3, 1), size=(500, 600))
