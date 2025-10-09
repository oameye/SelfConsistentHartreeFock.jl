module test_symbolic_utils
using SelfConsistentHartreeFock, SecondQuantizedAlgebra, Test
using Symbolics, SymbolicUtils
using Symbolics: unwrap
import SecondQuantizedAlgebra as SQA
using UnPack

using SelfConsistentHartreeFock: get_prefactor, get_independent

@testset "duffing" begin
    h = FockSpace(:cavity)
    @qnumbers a::Destroy(h)
    @variables F::Real Δ::Real K::Real κ::Real

    H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a)

    sys = HartreeFockSystem(H, [a], [κ])

    @unpack mean_field_eom = sys
    eom = unwrap.(mean_field_eom)
    vars = vcat(sys.operators, SQA.adjoint.(sys.operators))
    averaged_vars = SQA.average.(vars)

    indepdent = get_independent(eom[1], averaged_vars)
    @test isequal(indepdent, F)
    result = 2 * K * (SQA.average(a' * a')) + 2 * K * (SQA.average(a'))^2
    @test isequal(get_prefactor(eom[1], averaged_vars[1]), result)
end

@testset "symbolic_utils basic" begin
    @variables x::Real y::Real z::Real

    # Build expressions
    expr1 = 3x + 2x * y + y + 5
    expr2 = 7y + 9
    expr3 = x * y + z + 4

    # Unwrap expressions and variables
    ux = unwrap(x)
    uy = unwrap(y)
    uz = unwrap(z)
    uexpr1 = unwrap(expr1)
    uexpr2 = unwrap(expr2)
    uexpr3 = unwrap(expr3)

    # Expected prefactors (unwrapped)
    expected_pref_x_expr1 = unwrap(3 + 2y)      # 3 from 3x, 2y from 2x*y
    expected_pref_y_expr1 = unwrap(2x + 1)      # 2x from 2x*y, 1 from lone y
    expected_pref_x_expr2 = unwrap(0)

    # Test get_prefactor
    @test isequal(get_prefactor(uexpr1, ux), expected_pref_x_expr1)
    @test isequal(get_prefactor(uexpr1, uy), expected_pref_y_expr1)
    @test isequal(get_prefactor(uexpr2, ux), expected_pref_x_expr2)
    @test isequal(get_prefactor(uexpr3, uz), unwrap(1))

    # Expected independent parts
    expected_indep_expr1_wrt_x = unwrap(y + 5)
    expected_indep_expr1_wrt_y = unwrap(3x + 5)
    expected_indep_expr3_wrt_xy = unwrap(z + 4)

    @test isequal(get_independent(uexpr1, [ux]), expected_indep_expr1_wrt_x)
    @test isequal(get_independent(uexpr1, [uy]), expected_indep_expr1_wrt_y)
    @test isequal(get_independent(uexpr3, [ux, uy]), expected_indep_expr3_wrt_xy)

    # Decomposition check: expr1 ≈ x * pref_x + independent_x
    recon_expr1 = unwrap(x * get_prefactor(uexpr1, ux) + get_independent(uexpr1, [ux]))
    @test isequal(uexpr1, expand(recon_expr1))

    # Variable absent case for get_independent (all independent)
    @test isequal(get_independent(uexpr2, [ux]), uexpr2)

    # Zero prefactor when variable not present
    @test isequal(get_prefactor(uexpr1, uz), unwrap(0))

    # Multiple variables independence consistency:
    # Split expr3 relative to x then remove y from remainder; final independent wrt x,y should match direct
    pref_x_expr3 = get_prefactor(uexpr3, ux)
    remainder_expr3 = unwrap(uexpr3 - unwrap(x * pref_x_expr3))
    independent_via_two_step = get_independent(remainder_expr3, [uy])
    @test isequal(independent_via_two_step, get_independent(uexpr3, [ux, uy]))
end

@testset "get_linear" begin
    using SelfConsistentHartreeFock: get_linear_prefactor, get_nonlinear_terms

    h = FockSpace(:cavity)

    @qnumbers a::Destroy(h)
    @variables G::Real Δ::Real K::Real κ::Real

    H = -Δ * a' * a + K * (a'^2 * a^2) + G * (a' * a' + a * a)
    sys = HartreeFockSystem(H, [a], [κ])
    exprs1 = unwrap(sys.mean_field_eom[1])
    exprs2 = unwrap(sys.mean_field_eom[2])

    result1 = get_linear_prefactor(exprs1, SQA.average(a), [SQA.average(a), SQA.average(a')])
    result2 = get_linear_prefactor(
        exprs1, SQA.average(a'), [SQA.average(a), SQA.average(a')]
    )
    @test isequal(result1, unwrap(2 * G + 2 * K * SQA.average(a' * a')))
    @test isequal(result2, unwrap(-Δ + 4 * K * SQA.average(a' * a)))

    result = get_nonlinear_terms(exprs1, SQA.average(a), [SQA.average(a), SQA.average(a')])
    @test isequal(result, unwrap(2 * K * SQA.average(a')^2 * SQA.average(a)))

    result = get_nonlinear_terms(exprs2, SQA.average(a'), [SQA.average(a), SQA.average(a')])
    @test isequal(result, unwrap(2 * K * SQA.average(a)^2 * SQA.average(a')))

    result2 = get_nonlinear_terms(exprs2, [SQA.average(a), SQA.average(a')])
    result1 = get_nonlinear_terms(exprs1, [SQA.average(a), SQA.average(a')])
    @test isequal(result2, unwrap(2 * K * SQA.average(a)^2 * SQA.average(a')))
    @test isequal(result1, unwrap(2 * K * SQA.average(a')^2 * SQA.average(a)))
end
end # module
