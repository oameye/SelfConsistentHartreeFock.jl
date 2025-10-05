struct DymamicMatrix
    A::Matrix{Num}
    B::Matrix{Num}
end
Base.show(io::IO, ::MIME"text/plain", dm::DymamicMatrix) = print(io, "DymamicMatrix")

function dynamical_matrix(dict::Dict, operators)
    N = length(operators)
    A = zeros(Num, N, N)
    B = zeros(Num, N, N)
    correlators = Vector{QMul}()
    for i in 1:N
        for j in 1:N
            op_i = operators[i]
            op_j = operators[j]
            key_A = op_i' * op_j
            key_B = op_i' * op_j' # by convention, we label B with two creation operators
            A[i, j] += haskey(dict, key_A) ? dict[key_A] : 0.0
            B[i, j] += haskey(dict, key_B) ? 2*dict[key_B] : 0.0
            push!(correlators, key_A)
            push!(correlators, key_B)
        end
    end
    return correlators, DymamicMatrix(A, B)
end
# function dynamical_matrix(HF, operators, rates=zeros(Num, length(operators)))
#     dict = collect_dict(HF)
#     return dynamical_matrix(dict, operators, rates)
# end

function steady_state_function(dict::Dict, operators)
    eom = Vector{Num}(undef, length(operators) * 2)
    for idx in eachindex(operators)
        op = operators[idx]
        eom[idx] = dict[op]
        eom[idx + length(operators)] = dict[op']
    end
    return eom
end

struct HartreeFockSystem
    H::SQA.QAdd
    mean_field_eom::Vector{Num}
    dynamical_matrix::DymamicMatrix
    operators::Vector{Destroy}
    correlators::Vector{QMul}
    rates::Vector{Num}

    function HartreeFockSystem(H, operators, rates=zeros(Num, length(operators)))
        @assert length(operators) == length(rates)
        displacedH = displacement(H)
        HF = hartree_fock_approx(displacedH)
        dict = collect_dict(HF)
        correlators, DM = dynamical_matrix(dict, operators)
        mean_field_eom = steady_state_function(dict, operators)
        return new(HF, mean_field_eom, DM, operators, correlators, rates)
    end
end
function Base.show(io::IO, ::MIME"text/plain", sys::HartreeFockSystem)
    print(io, "HartreeFockSystem of modes: ", sys.operators)
end
