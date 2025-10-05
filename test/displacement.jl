module test_displacement

using SelfConsistentHartreeFock, SecondQuantizedAlgebra
import SecondQuantizedAlgebra as SQA
using Symbolics
using Test

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables F::Real Δ::Real K::Real κ::Real

testcase1 = a' * a
testresult = displacement(testcase1)
result =
    a' * a + SQA.average(a) * a' + SQA.average(a') * a + SQA.average(a') * SQA.average(a)
@test isequal(result, testresult)

testcase2 = K * (a'^2 * a^2)
result = displacement(testcase2)
collect_dict(result)

testcase3 = F * (a' + a)
testresult = displacement(testcase3)
result = F * (a' + a) + F * SQA.average(a') + F * SQA.average(a)
@test isequal(hash(sort(hash.(result.arguments))), hash(sort(hash.(testresult.arguments))))

end
