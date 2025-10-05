hartree_fock_approx(op) = op
hartree_fock_approx(op::SQA.QAdd) = mapreduce(hartree_fock_approx, +, op.arguments)
function hartree_fock_approx(op::SQA.QMul)
    factors = collect(op.args_nc)
    length(factors) <= 2 && return op
    coeff = op.arg_c
    terms = Any[]
    for (i, j) in combinations(1:length(factors), 2)
        pair_expr = factors[i] * factors[j]
        pair_expectation = SQA.average(pair_expr)
        remaining = [factors[k] for k in 1:length(factors) if k != i && k != j]
        remaining_expr = isempty(remaining) ? 1 : hartree_fock_approx(prod(remaining))
        push!(terms, coeff * pair_expectation * remaining_expr)
    end
    _sum = sum(terms)
    remove_constants!(_sum)
    return _sum
end
