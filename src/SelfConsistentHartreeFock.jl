module SelfConsistentHartreeFock

import SecondQuantizedAlgebra as SQA
using SecondQuantizedAlgebra: Destroy, Create, QTerm, QMul, QAdd, QNumber
import Combinatorics: combinations

using UnPack: @unpack
using LinearAlgebra: Diagonal

using SymbolicUtils: @compactified, BasicSymbolic, isadd, SymbolicUtils
using Symbolics: Num, Symbolics, unwrap
using TermInterface: TermInterface

using MatrixEquations: MatrixEquations
using FixedPointAcceleration: FixedPointAcceleration

import ModelingToolkit as MTK

export displacement,
    collect_dict,
    hartree_fock_approx,
    HartreeFockSystem,
    HartreeFockProblem,
    IterativeProblem,
    remake,
    fixed_point,
    parameter_sweep

include("utils.jl")
include("symbolic_utils.jl")
include("displacement.jl")
include("hartree-fock_approximation.jl")
include("system.jl")
include("iterationproblem.jl")
include("NonlinearProblem.jl")
include("self_consistent_iteration.jl")

end # module SelfConsistentHartreeFock
