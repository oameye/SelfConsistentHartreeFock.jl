# struct Result
#     αs::Vector{ComplexF64}
#     ns::Vector{Float64}
#     ms::Vector{ComplexF64}

#     function Result(v::Vector{Float64})
#         return new([v[1] + im * v[2]], [v[3]], [v[4] + im * v[5]])
#     end
#     function Result(::Missing)
#         return new([], [], [])
#     end
# end
function lyapunov_fixed_point(α, problem)
    @unpack M, D = problem.dynamical_matrix
    correlation_matrix = MatrixEquations.lyapc(M(α), D(α))
    return get_correlation_vector(correlation_matrix)
end

function meanfield_fixed_point(α, problem)
    @unpack I, F = problem.meanfield
    # v = I(α)-J(α) \ F(α) # Newton
    v = I(α) - F(α) # Newton
    return v[1:(end ÷ 2)]
end

function self_consistent_fixed_point(α, problem)
    v = as_complex_view(α)
    α′ = meanfield_fixed_point(v, problem)
    cs = lyapunov_fixed_point(v, problem)
    return collect(as_interleaved_view(vcat(α′, cs)))
end

function fixed_point(problem::IterativeProblem, α0; kwargs...)
    α0 = collect(as_interleaved_view(α0))
    f(α) = self_consistent_fixed_point(α, problem)
    FPresult = FixedPointAcceleration.fixed_point(f, α0; kwargs...)
    return  Dict(problem.unknowns .=> as_complex_view(FPresult.FixedPoint_))
end

function parameter_sweep(problem::IterativeProblem, param, range, α0; kwargs...)
    results = []
    p = problem.p
    α = collect(as_interleaved_view(α0))
    for val in range
        problem = remake(problem, merge(p, Dict(param => val)))
        f(α) = self_consistent_fixed_point(α, problem)
        FP = FixedPointAcceleration.fixed_point(f, α; kwargs...)
        if ismissing(FP.FixedPoint_)
            # push!(results, Dict(problem.unknowns .=> fill(NaN, length(α0))))
            push!(results, FP)
            # α = FP.FixedPoint_
        else
            push!(results, FP)
            α = FP.FixedPoint_
        end
    end
    return results
end
