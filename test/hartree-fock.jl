module test_hartree_fock

using SelfConsistentHartreeFock, SecondQuantizedAlgebra
import SecondQuantizedAlgebra as SQA
using Symbolics
using Test

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables F::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a)

testcase1 = (a'^2 * a^2)
testresult = hartree_fock_approx(testcase1)
result = SQA.average(a' * a') * a * a + SQA.average(a' * a) * a'* a + SQA.average(a' * a) * a'* a + SQA.average(a' * a) * a'* a + SQA.average(a' * a) * a'* a + SQA.average(a * a) * a'* a'
@test isequal(result, testresult)


testcase2 = (a'^2 * a)
testresult = hartree_fock_approx(testcase2)
result = SQA.average(a' * a') * a + SQA.average(a' * a) * a' + SQA.average(a' * a) * a'
@test isequal(result, testresult)

testcase3 = (a' * a^2)
testresult = hartree_fock_approx(testcase3)
result = SQA.average(a' * a) * a + SQA.average(a' * a) * a + SQA.average(a * a) * a'
@test isequal(result, testresult)

end
