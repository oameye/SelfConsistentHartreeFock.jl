Base.zero(s::BasicSymbolic{SQA.CNumber}) = 0

function hasvar(x, y)
    for arg in SymbolicUtils.arguments(x)
        isequal(arg, y) && return true
    end
    return false
end
function hasnotvars(args, vars)
    all(!isequal(x, v) for v in vars, x in args) || return false
end

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
        _    => x
    end
end
get_independent(x, vars) = x
