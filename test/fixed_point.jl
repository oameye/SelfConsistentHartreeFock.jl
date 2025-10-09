module test_fixed_point

using SelfConsistentHartreeFock
using SecondQuantizedAlgebra
using Symbolics
using Test

h = FockSpace(:cavity)

@qnumbers a::Destroy(h)
@variables F::Real Δ::Real K::Real κ::Real

H = -Δ * a' * a + K * (a'^2 * a^2) + F * (a' + a)

sys = HartreeFockSystem(H, [a], [κ])
p = Dict(F => 0.01, Δ => -0.01, K => 0.001, κ => 0.001)
problem = IterativeProblem(sys, p)

α0 = ComplexF64[0.05 + 0.0im, 0.0 + 0.0im, 0.0 + 0.0im]

@testset "picard solver matches accelerated solver" begin
    reference = fixed_point(problem, α0)
    picard = fixed_point(problem, α0; solver = :picard, maxiters = 2000, damping = 0.5, rtol = 1e-10)

    @test !isempty(reference.αs)
    @test !isempty(picard.αs)
    @test isapprox(reference.αs[1], picard.αs[1]; atol = 1e-7)
    @test isapprox(reference.ns[1], picard.ns[1]; atol = 1e-7)
    @test isapprox(reference.ms[1], picard.ms[1]; atol = 1e-7)

    failure = fixed_point(problem, α0; solver = :picard, maxiters = 1, damping = 0.1)
    @test isempty(failure.αs)
end

@testset "parameter sweep comparison" begin
    Δsweep = range(-0.01, stop = -0.005, length = 5)
    sweeps = parameter_sweep_compare(
        problem,
        Δ,
        Δsweep,
        α0;
        picard_kwargs = (; maxiters = 2000, damping = 0.5, rtol = 1e-10),
    )

    @test sweeps.matched == length(Δsweep)
    @test length(sweeps.fpa) ≥ sweeps.matched
    @test length(sweeps.picard) ≥ sweeps.matched

    for idx in 1:sweeps.matched
        r_fpa = sweeps.fpa[idx]
        r_picard = sweeps.picard[idx]
        @test isapprox(r_fpa.αs[1], r_picard.αs[1]; atol = 1e-6)
        @test isapprox(r_fpa.ns[1], r_picard.ns[1]; atol = 1e-6)
        @test isapprox(r_fpa.ms[1], r_picard.ms[1]; atol = 1e-6)
    end
end

@testset "example sweep (101 points)" begin
    Δsweep_full = range(-0.01, stop = 0.03, length = 101)
    comparison = parameter_sweep_compare(
        problem,
        Δ,
        Δsweep_full,
        α0;
        picard_kwargs = (; maxiters = 2000, damping = 0.5, rtol = 1e-10),
    )

    @test comparison.matched == length(comparison.picard)
    @test length(comparison.fpa) ≥ comparison.matched
    @test comparison.matched > 0

    for idx in 1:comparison.matched
        r_fpa = comparison.fpa[idx]
        r_picard = comparison.picard[idx]
        @test isapprox(r_fpa.αs[1], r_picard.αs[1]; atol = 1e-6)
        @test isapprox(r_fpa.ns[1], r_picard.ns[1]; atol = 1e-6)
        @test isapprox(r_fpa.ms[1], r_picard.ms[1]; atol = 1e-6)
    end
end

end
