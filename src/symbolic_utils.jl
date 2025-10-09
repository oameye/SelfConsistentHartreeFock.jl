Base.zero(s::BasicSymbolic{SQA.CNumber}) = 0

function hasvar(x, y)
    SymbolicUtils.istree(x) || return false
    for arg in SymbolicUtils.arguments(x)
        isequal(arg, y) && return true
    end
    return false
end

# Recursive occurrence test (handle leaf Syms safely)
occurs(x::SymbolicUtils.BasicSymbolic, v) =
    isequal(x, v) || (SymbolicUtils.istree(x) && any(occurs(arg, v) for arg in SymbolicUtils.arguments(x)))
occurs(x, v) = isequal(x, v)

# True if none of the args depend (even nonlinearly) on any variable in vars
function hasnotvars(args, vars)
    all(!occurs(arg, v) for arg in args for v in vars)
end

# Node-level independence
independent_of_vars(x, vars) = all(!occurs(x, v) for v in vars)

function get_prefactor(x::BasicSymbolic, y)
    @compactified x::BasicSymbolic begin
        Add  => sum([get_prefactor(arg, y) for arg in SymbolicUtils.arguments(x)])
        Mul  => hasvar(x, y) ? prod(filter(x -> !isequal(x, y), SymbolicUtils.arguments(x))) : 0
        Term => isequal(x, y) ? 1 : 0
        Sym  => isequal(x, y) ? 1 : 0
        _    => x
    end
end
get_prefactor(x, y) = zero(x)

function get_independent(x::BasicSymbolic, vars)
    @compactified x::BasicSymbolic begin
        Add  => sum([get_independent(arg, vars) for arg in SymbolicUtils.arguments(x)])
        Mul  => hasnotvars(SymbolicUtils.arguments(x), vars) ? x : 0
        Term => all(!isequal(x, v) for v in vars) ? x : 0
        Sym  => all(!isequal(x, v) for v in vars) ? x : 0
        _    => independent_of_vars(x, vars) ? x : 0
    end
end
get_independent(x, vars) = x

# Coefficient of v in x from terms that are linear in v (and independent of other vars)
function get_linear_prefactor(x::BasicSymbolic, v, vars)
    @compactified x::BasicSymbolic begin
        Add  => sum(get_linear_prefactor(arg, v, vars) for arg in SymbolicUtils.arguments(x))
        Mul  => begin
            args = SymbolicUtils.arguments(x)
            nv = count(a -> isequal(a, v), args)  # require v as a direct multiplicative factor exactly once
            nv == 1 || return 0
            rest = filter(a -> !isequal(a, v), args)
            return hasnotvars(rest, vars) ? prod(rest) : 0
        end
        Term => isequal(x, v) ? 1 : 0
        Sym  => isequal(x, v) ? 1 : 0
        _    => 0
    end
end
get_linear_prefactor(x, v, vars) = 0

function is_linear_mul_in_v(args, v, vars)
    nv = count(a -> isequal(a, v), args)
    nv == 1 || return false
    rest = filter(a -> !isequal(a, v), args)
    return hasnotvars(rest, vars)
end

# Extract all terms that contain v but are not linear in v (w.r.t. vars)
function get_nonlinear_terms(x::BasicSymbolic, v, vars)
    @compactified x::BasicSymbolic begin
        Add  => sum(get_nonlinear_terms(arg, v, vars) for arg in SymbolicUtils.arguments(x))
        Mul  => begin
            args = SymbolicUtils.arguments(x)
            nv = count(a -> isequal(a, v), args)
            nv == 0 && return 0
            is_linear_mul_in_v(args, v, vars) ? 0 : x
        end
        Term => isequal(x, v) ? 0 : (occurs(x, v) ? x : 0)
        Sym  => isequal(x, v) ? 0 : (occurs(x, v) ? x : 0)
        _    => occurs(x, v) ? x : 0  # e.g. powers like v^2, functions of v
    end
end
get_nonlinear_terms(x, v, vars) = 0


# Any dependency on any variable in vars
any_occurs(x, vars) = any(occurs(x, v) for v in vars)

# Extract all terms that are nonlinear w.r.t. the set `vars`
function get_nonlinear_terms(x::BasicSymbolic, vars)
    @compactified x::BasicSymbolic begin
        Add  => sum(get_nonlinear_terms(arg, vars) for arg in SymbolicUtils.arguments(x))
        Mul  => begin
            args = SymbolicUtils.arguments(x)
            # Identify direct factors that are exactly one of `vars`
            is_direct = [any(isequal(a, v) for v in vars) for a in args]
            n_direct = count(identity, is_direct)
            rest = [a for (a, d) in zip(args, is_direct) if !d]

            # If any remaining factor depends on vars (nested), it's nonlinear
            any_nested = any(any(occurs(r, v) for v in vars) for r in rest)
            if any_nested
                return x
            end

            # Otherwise classification depends on number of direct factors
            if n_direct == 0
                return 0             # independent of vars
            elseif n_direct == 1
                return 0             # linear in exactly one var
            else
                return x             # nonlinear: >= 2 direct occurrences
            end
        end
        Term => (any(isequal(x, v) for v in vars) ? 0 : (any_occurs(x, vars) ? x : 0))
        Sym  => (any(isequal(x, v) for v in vars) ? 0 : (any_occurs(x, vars) ? x : 0))
        _    => any_occurs(x, vars) ? x : 0  # e.g. powers v^2, functions of vars
    end
end
get_nonlinear_terms(x, vars) = 0
