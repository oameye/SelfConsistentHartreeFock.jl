function remove_constants!(expr::QTerm)
    filter!(isQMul, SQA.arguments(expr))
    return expr
end
isQMul(x) = x isa QMul

function push_or_add!(dict, key, value)
    if haskey(dict, key)
        dict[key] += value
    else
        dict[key] = value
    end
end
function collect_dict(expr::QAdd)
    dict = Dict{Any,Any}()
    for term in expr.arguments
        if term isa QMul
            qmul = QMul(1.0, term.args_nc)
            push_or_add!(dict, qmul, term.arg_c)
        else
            push_or_add!(dict, 1.0, term)
        end
    end
    return dict
end

function find_conjugate_pairs(ops)
    pairs = Tuple{Int,Int}[]
    for i in 1:length(ops)
        for j in i:length(ops)
            if isequal(ops[i], ops[j]')
                push!(pairs, (i, j))
            end
        end
    end
    return pairs
end

function triu_vec(A::AbstractMatrix)
    m, n = size(A)
    p = min(m, n)
    # number of elements in the upper triangle of an m×n matrix
    len = p * (n + 1) - div(p * (p + 1), 2)

    v = Vector{eltype(A)}()
    sizehint!(v, len)

    @inbounds for j in 1:n
        for i in 1:min(j, m)
            push!(v, A[i, j])
        end
    end

    return v
end

# [re1, im1, re2, im2, ...]  ->  Vector{Complex{T}}   (no copy)
as_complex_view(x::Vector{T}) where {T<:Real} = begin
    @assert iseven(length(x)) "length must be even (pairs of re,im)."
    reinterpret(Complex{T}, x)
end
# Back to interleaved real/imag view (no copy)
as_interleaved_view(z::Vector{Complex{T}}) where {T<:Real} =
    reinterpret(T, z)
