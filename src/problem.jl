struct DynamicalMatrixFunction
    M::Function
    D::Function
end

struct MeanfieldFunction
    I::Function
    F::Function
    J::Function
end

struct IterativeProblem
    sys::HartreeFockSystem
    meanfield::MeanfieldFunction
    dynamical_matrix::DynamicalMatrixFunction
    unknowns::Vector
    p::Dict

    function IterativeProblem(sys::HartreeFockSystem, p::Dict)
        @unpack operators, correlators = sys
        unknowns = vcat(operators, correlators)
        vars = (unique(vcat(unknowns, SQA.adjoint.(unknowns))))

        meanfield = compile_iterative_eom(sys, unknowns, vars, p)
        dynamical_matrix = compile_dynamical_matrix(sys, unknowns, vars, p)

        return new(sys, meanfield, dynamical_matrix, SQA.average.(unknowns), p)
    end
end
function Base.show(io::IO, ::MIME"text/plain", prob::IterativeProblem)
    return print(io, "IterativeProblem")
end

function remake(prob::IterativeProblem, p::Dict)
    return IterativeProblem(prob.sys, p)
end

function construct_iterative_eom(sys::HartreeFockSystem)
    operators = sys.operators
    vars = SQA.average.(vcat(operators, SQA.adjoint.(operators)))
    eom = sys.mean_field_eom
    jac = Symbolics.jacobian(eom, vars)
    return unwrap.((vars, eom, jac))
end

function compile_iterative_eom(
    sys::HartreeFockSystem, unknowns::Vector{QNumber}, vars::Vector{QNumber}, p::Dict
)
    I, eom, jac = construct_iterative_eom(sys)

    averaged_vars = SQA.average.(vars)

    F_oop, _ = compile(eom, averaged_vars, p)
    Ff = wrap_function_vars(F_oop, unknowns, vars)

    jac_oop, _ = compile(jac, averaged_vars, p)
    Jf = wrap_function_vars(jac_oop, unknowns, vars)

    I_oop, _ = compile(I, averaged_vars, p)
    If = wrap_function_vars(I_oop, unknowns, vars)

    return MeanfieldFunction(If, Ff, Jf)
end

function compile(ex::AbstractArray{Complex{Num}}, vars, p::Dict)
    # too avoid problems with complex numbers in Symbolics
    exim = getfield.(ex, :im)
    exre = getfield.(ex, :re)
    ex = Symbolics.substitute.(exre, Ref(p)) + im * Symbolics.substitute.(exim, Ref(p))
    return f = Symbolics.build_function(ex, vars; expression=Val(false))
end

function compile(ex, vars, p::Dict)
    ex = Symbolics.substitute.(ex, Ref(p))
    return f = Symbolics.build_function(ex, vars; expression=Val(false))
end

function compile_dynamical_matrix(sys::HartreeFockSystem, unknowns, vars, p::Dict)
    @unpack operators, dynamical_matrix, rates = sys
    @unpack A, B = dynamical_matrix
    A = unwrap.(A)
    B = unwrap.(B) # This is needed to make conj work properly on B

    # dynamical matrix
    M = [
        (-im * A-Diagonal(rates) / 2)   (-im*B)
        (im*conj.(B))   (im * transpose(A)-Diagonal(rates) / 2)
    ]

    # vacuum input noise
    D = [Diagonal(rates) zero(A); zero(A) zero(A)]

    # compile dynamical matrix
    averaged_vars = SQA.average.(vars)
    fM_oop, _ = compile(M, averaged_vars, p)
    fD_oop, _ = compile(D, averaged_vars, p)
    fM = wrap_function_vars(fM_oop, unknowns, vars)
    fD = wrap_function_vars(fD_oop, unknowns, vars)
    return DynamicalMatrixFunction(fM, fD)
end

function make_vars_function(unknowns, vars)
    pairs = find_conjugate_pairs(vars)
    function f_vars(unknowns)
        vars = zeros(ComplexF64, length(vars))
        for (i, var) in enumerate(unknowns)
            vars[i] = var
        end
        for (i, j) in pairs
            vars[j] = conj(vars[i])
        end
        return vars
    end
    return f_vars
end

function wrap_function_vars(f, unknowns, vars)
    f_vars = make_vars_function(unknowns, vars)
    function g(unknowns)
        vars = f_vars(unknowns)
        return f(vars)
    end
    return g
end

# function compile_system(sys::HartreeFockSystem, p::Dict)
#     @unpack operators, mean_field_eom, dynamical_matrix, rates, correlators = sys

#     unknowns = vcat(operators, correlators)
#     vars = unique(vcat(unknowns, SQA.adjoint.(unknowns)))
#     avaeraged_vars = SQA.average.(vars)

#     # dynamical matrix
#     dynamical_matrix_function = compile_dynamical_matrix(sys, avaeraged_vars, p)
#     # mean field eom
#     f_eom_oop, _ = compile(mean_field_eom, avaeraged_vars, p)

#     f_nonlinearsolve = make_nonlinearsolve_function(
#         dynamical_matrix_function, f_eom_oop, unknowns, vars
#     )
#     f_eom_vars = wrap_function_vars(f_eom_oop, unknowns, vars)
#     f_dyn_vars = wrap_function_vars(f_dyn, unknowns, vars)
#     return unknowns, f_nonlinearsolve, f_eom_vars, f_dyn_vars
# end

# function make_nonlinearsolve_function(dynamical_matrix_function, f_eom, unknowns, vars)
#     @unpack f_dyn = dynamical_matrix_function
#     f_vars = make_vars_function(unknowns, vars)
#     function f(unknowns)
#         vars = f_vars(unknowns)
#         return vcat(f_dyn(vars)[:], f_eom(vars))
#     end
#     return f
# end

# function get_prefactors(eom, vars)
#     dict = Dict{Any,Any}()
#     eom = deepcopy(eom)
#     for var in vars
#         prefactor = get_prefactor(eom, var)
#         dict[var] = prefactor
#         eom = Symbolics.simplify(eom - prefactor * var)
#     end
#     dict[1] = get_independent(eom, vars)
#     return dict
# end
# function construct_iterator_eom(eom, vars)
#     N = length(vars) # vars is ordered as [a1, a2, ..., a1†, a2†, ...]
#     L = zeros(Num, N, N)
#     Fs = zeros(Num, N)
#     Ns = zeros(Num, N)
#     for i in eachindex(eom)
#         eq = eom[i]
#         for j in 1:N
#             var = vars[j]
#             L[i, j] =  get_linear_prefactor(eq,var, vars)
#         end
#         Fs[i] = get_independent(eq, vars)
#         Ns[i] = get_nonlinear_terms(eq, vars)
#     end
#     return L, Ns, Fs
# end
