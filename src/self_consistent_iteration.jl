struct Result
    αs::Vector{ComplexF64}
    ns::Vector{Float64}
    ms::Vector{ComplexF64}

    function Result(v::AbstractVector{ComplexF64})
        @assert length(v) ≥ 3 "Expected three complex entries (α, n, m)."
        α, n, m = v[1:3]
        return new([ComplexF64(α)], [Float64(real(n))], [ComplexF64(m)])
    end

    function Result(::Missing)
        return new([], [], [])
    end
end
function lyapunov_fixed_point(α, problem)
    @unpack M, D = problem.dynamical_matrix
    # correlation_matrix = MatrixEquations.lyapc(M(α), D(α))
    correlation_matrix = lyap(M(α), D(α))
    Nm = correlation_matrix[(end ÷ 2 + 1):end, (end ÷ 2 + 1):end]
    Mm = correlation_matrix[1:(end ÷ 2), (end ÷ 2 + 1):end]
    return vcat(vec(Nm), vec(Mm))
end

function meanfield_fixed_point(α, problem)
    @unpack A, Fs = problem.meanfield
    v = -inv(A(α)) * Fs(α)
    return v[1:(end ÷ 2)]
end

function self_consistent_fixed_point(α::AbstractVector{ComplexF64}, problem)
    α′ = meanfield_fixed_point(α, problem)
    n, m = lyapunov_fixed_point(α, problem)
    return ComplexF64[α′[1], n, m]
end

function fixed_point_accelerated(problem::IterativeProblem, α0; kwargs...)
    f(α) = self_consistent_fixed_point(α, problem)
    return FixedPointAcceleration.fixed_point(f, α0; kwargs...)
end

function fixed_point(problem::IterativeProblem, α0; solver::Symbol=:fpa, kwargs...)
    if solver === :fpa
        FPresult = fixed_point_accelerated(problem, α0; kwargs...)
        return Result(FPresult.FixedPoint_)
    elseif solver === :picard
        picard = fixed_point_picard(problem, α0; kwargs...)
        return picard.converged ? Result(picard.value) : Result(missing)
    else
        throw(
            ArgumentError("Unknown solver $solver. Supported solvers are :fpa and :picard.")
        )
    end
end

function parameter_sweep(
    problem::IterativeProblem, param, range, α0; solver::Symbol=:fpa, kwargs...
)
    results = Result[]
    p = problem.p
    if solver === :fpa
        α = α0
        for val in range
            local_problem = remake(problem, merge(p, Dict(param => val)))
            f(α) = self_consistent_fixed_point(α, local_problem)
            FP = FixedPointAcceleration.fixed_point(
                f, α; kwargs...
            )
            if ismissing(FP.FixedPoint_)
                break
            end
            push!(results, Result(FP.FixedPoint_))
            α = FP.FixedPoint_
        end
    elseif solver === :picard
        α = α0
        for val in range
            local_problem = remake(problem, merge(p, Dict(param => val)))
            picard = fixed_point_picard(local_problem, α; kwargs...)
            if !picard.converged
                break
            end
            push!(results, Result(picard.value))
            α = picard.value
        end
    else
        throw(
            ArgumentError("Unknown solver $solver. Supported solvers are :fpa and :picard.")
        )
    end
    return results
end

function parameter_sweep_compare(
    problem::IterativeProblem,
    param,
    range,
    α0;
    fpa_kwargs::NamedTuple=(;),
    picard_kwargs::NamedTuple=(;),
)
    fpa_results = parameter_sweep(problem, param, range, α0; solver=:fpa, fpa_kwargs...)
    picard_results = parameter_sweep(
        problem, param, range, α0; solver=:picard, picard_kwargs...
    )
    matched = min(length(fpa_results), length(picard_results))
    return (fpa=fpa_results, picard=picard_results, matched=matched)
end
