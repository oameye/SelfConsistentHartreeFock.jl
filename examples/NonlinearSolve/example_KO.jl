using SecondQuantizedAlgebra, Plots
import SelfConsistentHartreeFock as SCHF
import SecondQuantizedAlgebra as SQA
using SelfConsistentHartreeFock: @unpack, Diagonal
using QuantumCumulants, ModelingToolkit, Symbolics

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
ps = @parameters F::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a) / 2

# eqs = QuantumCumulants.complete(QuantumCumulants.meanfield([a], H, [a]; rates=[κ], order=1))
# @named sys = System(eqs)
# ModelingToolkit.unknowns(sys)
# sys

sys = SCHF.HartreeFockSystem(H, [a], [κ])
propertynames(sys)

# me = QuantumCumulants.meanfield([a], sys.H, [a]; rates=[κ], order=1)
# eqs = QuantumCumulants.complete(me)

@named MTKsystem = ModelingToolkit.System(sys, ps)

@unpack operators, correlators = sys
unknowns = vcat(operators, correlators)
averaged_unknowns = SQA.average.(unknowns)

meanfield = sys.mean_field_eom
lyapunov = SCHF.compute_lyapunov(sys)

eqs = Symbolics.unwrap.(vcat(meanfield, lyapunov))
simplified_eqs = SCHF.replace_conj(eqs, averaged_unknowns)

varmap = SCHF.make_varmap(averaged_unknowns)
prepared_eqs = Num[Symbolics.substitute(eq, varmap) for eq in simplified_eqs]

vs_mtk = collect(values(varmap))
D = ModelingToolkit.Differential(ModelingToolkit.t_nounits)
MTKeqs = map(eachindex(vs_mtk)) do i
    Symbolics.Equation(D(vs_mtk[i]), prepared_eqs[i])
end

# mtkcompile(MTKsystem)
# ModelingToolkit.NonlinearFunction(MTKsystem; check_compatibility=false)

p = Dict(F => 0.01, Δ => -0.01, K => 0.001, κ => 0.001)
α0 = ComplexF64[rand(ComplexF64), 0.0, 0.0]
init = merge(Dict(ModelingToolkit.unknowns(MTKsystem) .=> α0), p)

ModelingToolkit.equations(MTKsystem)[1]
tspan = (0.0, 650.0)
prob = ODEProblem(MTKsystem, init, tspan; jac=true)

using OrdinaryDiffEq
sol = solve(prob, Tsit5())
trange = range(tspan..., 1000)
plot(abs.(getindex.(sol(trange).u,1)))

using NonlinearSolve
prob = NonlinearProblem(prob)
solve(prob, NewtonRaphson())
# eqs = vcat(meanfield, lyapunov) .~ 0
# α0 = ComplexF64[rand(ComplexF64), 0.0, 0.0]
# fixed_point(problem, α0)

# alg_kwargs = (;Algorithm = :Anderson,
#                 ConvergenceMetricThreshold= 1e-10, MaxIter = 1000, MaxM = 10, ExtrapolationPeriod = 7, Dampening = 1.0)
# Δsweep = range(-0.01, 0.03, 101)
# results_up = parameter_sweep(problem, Δ, Δsweep, α0)
# results_down = parameter_sweep(problem, Δ, reverse(Δsweep), α0; Dampening = 0.5)

# amplitude_up = map(results_up) do result
#     v = result.FixedPoint_
#     if v === missing
#         return NaN
#     end
#     SCHF.as_complex_view(v)[1] |> norm
# end
# amplitude_down = map(results_down) do result
#     v = result.FixedPoint_
#     if ismissing(v)
#         return NaN
#     end
#     SCHF.as_complex_view(v)[1] |> norm
# end

# fluctuation_up = map(results_up) do result
#     v = result.FixedPoint_
#     if v === missing
#         return NaN
#     end
#     SCHF.as_complex_view(v)[2] |> norm
# end
# fluctuation_down = map(results_down) do result
#     v = result.FixedPoint_
#     if v === missing
#         return NaN
#     end
#     SCHF.as_complex_view(v)[2] |> norm
# end
# anomalous_up = map(results_up) do result
#     v = result.FixedPoint_
#     if v === missing
#         return NaN
#     end
#     SCHF.as_complex_view(v)[3] |> norm
# end
# anomalous_down = map(results_down) do result
#     v = result.FixedPoint_
#     if v === missing
#         return NaN
#     end
#     SCHF.as_complex_view(v)[3] |> norm
# end;

# plt1 = plot(Δsweep, amplitude_up; xlabel = "Detuning Δ", ylabel = "Amplitude", legend = false)
# plot!(reverse(Δsweep), amplitude_down)

# plt2 = plot(Δsweep, fluctuation_up; xlabel = "Detuning Δ", ylabel = "Fluctuation", legend = false)
# plot!(reverse(Δsweep), fluctuation_down)

# plt3 = plot(Δsweep, anomalous_up; xlabel = "Detuning Δ", ylabel = "Anomalous", legend = false)
# plot!(reverse(Δsweep), anomalous_down)

# plot(plt1, plt2, plt3; layout = (3, 1), size=(500, 600))
