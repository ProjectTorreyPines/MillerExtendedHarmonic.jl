module MillerExtendedHarmonic

using RecipesBase
import LinearAlgebra: dot
using Optim
using Dierckx

const halfpi = 0.5 * π
const twopi = 2π
const inv_twopi = 1.0 / twopi

include("MXH.jl")
export MXH, MXH!, flat_coeffs, flat_coeffs!, copy_MXH!

include("metrics.jl")

include("relative.jl")

const document = Dict()
document[Symbol(@__MODULE__)] = [name for name in Base.names(@__MODULE__; all=false, imported=false) if name != Symbol(@__MODULE__)]

end
