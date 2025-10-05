module test_hartree_fock

using SelfConsistentHartreeFock, SecondQuantizedAlgebra
using Test

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@cnumbers F Δ K κ

H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a) / 2

testcase1 = K * (a'^2 * a^2)
result = hartree_fock_approx(testcase1)
collect_dict(result)

testcase2 = (a'^2 * a)
result = hartree_fock_approx(testcase2)
collect_dict(result)

testcase3 = (a' * a^2)
result = hartree_fock_approx(testcase3)
collect_dict(result)

end
