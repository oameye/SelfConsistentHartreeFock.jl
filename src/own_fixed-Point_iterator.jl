
function _damped_fixed_point(f, α0;
    maxiters::Integer=500,
    rtol::Real=1e-8,
    atol::Real=1e-12,
    damping::Real=0.5,
    min_damping::Real=1e-4,
    damping_growth::Real=1.2,
    verbose::Bool=false,
    normfun=norm)

    damping > 0 || throw(ArgumentError("Damping must be positive, got $damping"))
    min_damping > 0 || throw(ArgumentError("min_damping must be positive, got $min_damping"))

    α = copy(α0)
    g = f(α)
    F = α .- g
    residual = normfun(F)
    scale = max(normfun(g), normfun(α), 1.0)
    tol = max(atol, rtol * scale)
    if residual ≤ tol
        return (value=α, converged=true, iterations=0, residual=residual)
    end

    n = length(α)
    T = eltype(α)
    H = Matrix{T}(I, n, n)

    reset_identity!(H) = begin
        H .= zero(T)
        @inbounds for i in 1:n
            H[i, i] = one(T)
        end
    end

    damping_current = damping

    for iter in 1:maxiters
        damping_local = damping_current
        accepted = false
        α_next = similar(α)
        g_next = similar(g)
        F_next = similar(F)
        residual_next = residual
        scale_next = scale
        tol_next = tol
        attempts = 0

        while !accepted
            step = -(H * F)
            if normfun(step) == 0
                step .= -F
            end
            α_candidate = α .+ damping_local .* step
            if !all(isfinite, α_candidate)
                damping_local *= 0.5
                if damping_local < min_damping
                    return (value=α_candidate, converged=false, iterations=iter, residual=NaN)
                end
                continue
            end

            g_candidate = f(α_candidate)
            F_candidate = α_candidate .- g_candidate
            residual_candidate = normfun(F_candidate)
            scale_candidate = max(normfun(g_candidate), normfun(α_candidate), 1.0)
            tol_candidate = max(atol, rtol * scale_candidate)

            if residual_candidate ≤ residual || damping_local ≤ min_damping
                accepted = true
                α_next = α_candidate
                g_next = g_candidate
                F_next = F_candidate
                residual_next = residual_candidate
                scale_next = scale_candidate
                tol_next = tol_candidate
            else
                damping_local *= 0.5
                attempts += 1
                if damping_local < min_damping || attempts > 20
                    accepted = true
                    α_next = α_candidate
                    g_next = g_candidate
                    F_next = F_candidate
                    residual_next = residual_candidate
                    scale_next = scale_candidate
                    tol_next = tol_candidate
                end
            end
        end

        verbose && println("[fixed-point] iter=$(iter) residual=$(residual_next) (damping=$(damping_local))")

        if residual_next ≤ tol_next
            return (value=α_next, converged=true, iterations=iter, residual=residual_next)
        end

        s = α_next .- α
        y = F_next .- F
        denom = dot(s, y)
        if abs(denom) > 1e-12
            correction = s .- H * y
            H += (correction * adjoint(s)) / denom
        else
            reset_identity!(H)
        end

        if damping_local < damping_current
            damping_current = max(damping_local, min_damping)
        else
            damping_current = min(1.0, damping_current * damping_growth)
        end

        α = α_next
        g = g_next
        F = F_next
        residual = residual_next
        scale = scale_next
        tol = tol_next
    end

    (value=α, converged=false, iterations=maxiters, residual=residual)
end


function fixed_point_picard(problem::IterativeProblem, α0; kwargs...)
    f(α) = self_consistent_fixed_point(α, problem)
    _damped_fixed_point(f, α0; kwargs...)
end

# function _damped_fixed_point(
#     f,
#     α0;
#     maxiters::Integer=500,
#     rtol::Real=1e-8,
#     atol::Real=1e-12,
#     damping::Real=0.5,
#     verbose::Bool=false,
#     normfun=norm,
# )
#     damping > 0 || throw(ArgumentError("Damping must be positive, got $damping"))
#     α = copy(α0)
#     residual = Inf
#     for iter in 1:maxiters
#         fx = f(α)
#         step = fx .- α
#         residual = normfun(step)
#         αnext = α .+ damping .* step
#         verbose && println("[fixed-point] iter=$(iter) residual=$(residual)")
#         if !all(isfinite, αnext)
#             return (value=αnext, converged=false, iterations=iter, residual=NaN)
#         end
#         scale = max(normfun(fx), normfun(α), 1.0)
#         if residual ≤ max(atol, rtol * scale)
#             return (value=αnext, converged=true, iterations=iter, residual=residual)
#         end
#         α = αnext
#     end
#     fx = f(α)
#     residual = normfun(fx .- α)
#     return (value=α, converged=false, iterations=maxiters, residual=residual)
# end
