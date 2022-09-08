module MillerExtendedHarmonic

using RecipesBase
import LinearAlgebra: dot

include("MXH.jl")
export MXH, R_MXH, Z_MXH, flat_coeffs, flat_coeffs!, copy_MXH!, MXH_coeffs!

include("metrics.jl")
export Tr, dTr_dρ, dTr_dθ, dR_dρ, dZ_dρ, dR_dθ, dZ_dθ
export Jacobian, ∇ρ, ∇ρ2, ∇θ, ∇θ2, gρρ, gρθ, gθθ

include("relative.jl")
export in_surface, nearest_angle

end # module