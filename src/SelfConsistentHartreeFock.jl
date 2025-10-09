module SelfConsistentHartreeFock

import SecondQuantizedAlgebra as SQA
using SecondQuantizedAlgebra: Destroy, Create, QTerm, QMul, QAdd, QNumber
import Combinatorics: combinations
using SymbolicUtils: @compactified, BasicSymbolic, isadd, SymbolicUtils
using Symbolics: Num, Symbolics, unwrap
using UnPack: @unpack
using LinearAlgebra: Diagonal, norm, I, dot, lyap

using MatrixEquations: MatrixEquations
using FixedPointAccelerationNext: FixedPointAccelerationNext

export displacement,
    collect_dict,
    hartree_fock_approx,
    HartreeFockSystem,
    HartreeFockProblem,
    IterativeProblem,
    remake,
    fixed_point,
    parameter_sweep,
    parameter_sweep_compare

include("utils.jl")
include("symbolic_utils.jl")
include("displacement.jl")
include("hartree-fock_approximation.jl")
include("system.jl")
include("problem.jl")
include("self_consistent_iteration.jl")
include("own_fixed-Point_iterator.jl")

end # module SelfConsistentHartreeFock
