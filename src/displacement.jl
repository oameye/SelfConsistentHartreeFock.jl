function displacement(op::QMul)
    return op.arg_c * prod(displacement, op.args_nc)
end
function displacement(op::QAdd)
    return sum(displacement, op.arguments)
end
displacement(a::Create) = SQA.average(a) + a
displacement(a::Destroy) = SQA.average(a) + a
