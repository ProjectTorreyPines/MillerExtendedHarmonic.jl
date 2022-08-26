module MillerExtendedHarmonic

using RecipesBase
import LinearAlgebra: dot

include("MXH.jl")
export MXH, R_MXH, Z_MXH, flat_coeffs, flat_coeffs!, copy_MXH!

include("relative.jl")
export in_surface, nearest_angle

include("metrics.jl")
export Tr, dTr_dρ, dTr_dθ, dR_dρ, dZ_dρ, dR_dθ, dZ_dθ, Jacobian, ∇ρ, ∇ρ2, ∇θ, ∇θ2

end # module