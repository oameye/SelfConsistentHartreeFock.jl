function MTK.System(sys::HartreeFockSystem, ps; kwargs...)
    @unpack operators, correlators = sys
    unknowns = vcat(operators, correlators)
    averaged_unknowns = SQA.average.(unknowns)

    meanfield = sys.mean_field_eom
    lyapunov = compute_lyapunov(sys)

    eqs = Symbolics.unwrap.(vcat(meanfield, lyapunov))
    simplified_eqs = replace_conj(eqs, averaged_unknowns)
    varmap = make_varmap(averaged_unknowns)
    prepared_eqs = [Symbolics.substitute(eq, varmap) for eq in simplified_eqs]

    vs_mtk = collect(values(varmap))
    D = MTK.Differential(MTK.t_nounits)
    MTKeqs = map(eachindex(vs_mtk)) do i
        Symbolics.Equation(D(vs_mtk[i]), prepared_eqs[i])
    end

    MTKsys = MTK.System(MTKeqs, MTK.t_nounits, vs_mtk, ps; kwargs...)
    return MTK.complete(MTKsys)
end

function compute_lyapunov(sys::HartreeFockSystem)
    @unpack operators, dynamical_matrix, rates = sys
    @unpack A, B = dynamical_matrix
    A = unwrap.(A)
    B = unwrap.(B) # This is needed to make conj work properly on B

    # dynamical matrix
    M = [
        (-im * A-Diagonal(rates) / 2)   (-im*B)
        (im*conj.(B))   (im * transpose(A)-Diagonal(rates) / 2)
    ]

    # vacuum input noise
    D = [Diagonal(rates) zero(A); zero(A) zero(A)]

    C = SQA.average.(get_correlation_matrix(operators))
    lyapunov_matrix = Symbolics.wrap(M * C + C * adjoint(M) + D)
    return vec(lyapunov_matrix[(end ÷ 2 + 1):end, :])
end

function SQA._conj(v::SQA.Average)
    arg = v.arguments[1]
    adj_arg = adjoint(arg)
    return SQA._average(adj_arg)
end

function replace_conj(eqs, vs)
    # subs = Dict()
    # for op in unknowns
    #     if !isequal(adjoint(op), op)
    #         subs[SQA.average(op')] = SQA._average(op)'
    #     end
    # end
    # simplified_eqs = map(eqs) do eq
    #     Symbolics.substitute(eq, subs)
    # end
    vhash = map(hash, vs)

    vs′ = map(SQA._conj, vs)
    vs′hash = map(hash, vs′)
    i = 1
    while i <= length(vs′)
        if vs′hash[i] ∈ vhash
            deleteat!(vs′, i)
            deleteat!(vs′hash, i)
        else
            i += 1
        end
    end
    simplified_eqs = map(eqs) do eq
        substitute_conj(eq, vs′, vs′hash)
    end
    return simplified_eqs
end
function substitute_conj(term::T, vs′, vs′hash) where {T}
    if SymbolicUtils.iscall(term)
        if term isa SQA.Average
            if hash(term) ∈ vs′hash
                term′ = SQA._conj(term)
                return conj(term′)
            else
                return term
            end
        else
            _f = x -> substitute_conj(x, vs′, vs′hash)
            args = map(_f, SymbolicUtils.arguments(term))
            operation = SymbolicUtils.operation(term)
            metadata = TermInterface.metadata(term)
            return TermInterface.maketerm(T, operation, args, metadata)
        end
    else
        return term
    end
end

# Adding MTK variables
function add_vars!(varmap, vs)
    keys = getindex.(varmap, 1)
    vals = getindex.(varmap, 2)
    hashkeys = map(hash, keys)
    hashvals = map(hash, vals)
    hashvs = map(hash, vs)
    for i in 1:length(vs)
        if !(hashvs[i] ∈ hashkeys)
            var = make_var(vs[i])
            !(hash(var) ∈ hashvals) || @warn string(
                "Two different averages have the exact same name. ",
                "This may lead to unexpected behavior when trying to access the solution for $(vals[i])",
            )
            push!(keys, vs[i])
            push!(vals, var)
            push!(hashkeys, hashvs[i])
        end
    end
    for i in (length(varmap) + 1):length(keys)
        push!(varmap, keys[i] => vals[i])
    end
    return varmap
end

function make_var_term(v)
    sym = Symbol(string(v))
    d = source_metadata(:make_var, sym)
    var_f = SymbolicUtils.Sym{SymbolicUtils.FnType{Tuple{Any},Complex}}(sym; metadata=d)
    return SymbolicUtils.Term{Complex}(var_f, [MTK.t_nounits]; metadata=d)
end

function make_var_sym(v)
    sym = Symbol(string(v))
    d = source_metadata(:make_var, sym)
    return var_f = SymbolicUtils.Sym{SymbolicUtils.FnType{Tuple{Any},Complex}}(
        sym; metadata=d
    )
    # return SymbolicUtils.Term{Complex}(var_f, [MTK.t_nounits]; metadata = d)
end

function source_metadata(source, name)
    return Base.ImmutableDict{DataType,Any}(Symbolics.VariableSource, (source, name))
end

function make_varmap(vs)
    varmap = Pair{Any,Any}[]
    add_vars!(varmap, vs)
    return Dict(varmap)
end
