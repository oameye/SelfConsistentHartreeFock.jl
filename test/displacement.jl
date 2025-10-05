module test_displacement

using SelfConsistentHartreeFock, SecondQuantizedAlgebra
using Test

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@qnumbers d::Destroy(h)
@cnumbers F Δ K κ

testcase1 = -Δ * a' * a
result = displacement(testcase1, d)
collect_dict(result)

testcase2 = K * (a'^2 * a^2)
result = displacement(testcase2, d)
collect_dict(result)

testcase3 = F * (a' + a) / 2
result = displacement(testcase3, d)
collect_dict(result)

end
