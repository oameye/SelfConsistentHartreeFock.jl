struct Result
    αs::Vector{ComplexF64}
    ns::Vector{Float64}
    ms::Vector{ComplexF64}

    function Result(v::Vector{Float64})
        new([v[1]+im*v[2]], [v[3]], [v[4]+im*v[5]])
    end
        function Result(::Missing)
        new([], [], [])
    end
end
function lyapunov_fixed_point(α, problem)
    @unpack M, D = problem.dynamical_matrix
    correlation_matrix = MatrixEquations.lyapc(M(α), D(α))
    Nm = correlation_matrix[end÷2+1:end, end÷2+1:end]
    Mm = correlation_matrix[1:end÷2, end÷2+1:end]
    vcat(vec(Nm), vec(Mm))
end

function meanfield_fixed_point(α, problem)
    @unpack A, Fs = problem.meanfield
    v = - inv(A(α))*Fs(α)
    v[1:end÷2]
end

function self_consistent_fixed_point(α, problem)
    v = [α[1]+im*α[2], α[3], α[4]+im*α[5]]
    α′ = meanfield_fixed_point(v, problem)
    n, m = lyapunov_fixed_point(v, problem)
    [real(α′[1]), imag(α′[1]), real(n), real(m), imag(m)]
end

function fixed_point(problem::IterativeProblem, α0; kwargs...)
    α0 = [real(α0[1]), imag(α0[1]), real(α0[2]), real(α0[3]), imag(α0[3])]
    f(α) = self_consistent_fixed_point(α, problem)
    FPresult = FixedPointAcceleration.fixed_point(f, α0; kwargs...)
    Result(FPresult.FixedPoint_)
end

function parameter_sweep(problem::IterativeProblem, param, range, α0; kwargs...);
    results = Result[]
    p = problem.p
    α = [real(α0[1]), imag(α0[1]), real(α0[2]), real(α0[3]), imag(α0[3])]
    for val in range
        problem = remake(problem, merge(p, Dict(param => val)))
        f(α) = self_consistent_fixed_point(α, problem)
        FP = FixedPointAcceleration.fixed_point(f, α; Algorithm = :Anderson, kwargs...)
        if ismissing(FP.FixedPoint_)
            break
        end
        push!(results, Result(FP.FixedPoint_))
        α = FP.FixedPoint_
    end
    results
end
