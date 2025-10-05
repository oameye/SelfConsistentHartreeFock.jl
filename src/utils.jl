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
