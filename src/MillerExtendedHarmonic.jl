module MillerExtendedHarmonic

using RecipesBase
import LinearAlgebra: dot
using Optim

const halfpi = 0.5 * π

include("MXH.jl")
export MXH, MXH!, R_MXH, Z_MXH, flat_coeffs, flat_coeffs!, copy_MXH!, MXH_coeffs!
export fit_flattened!, optimize_fit!

include("metrics.jl")
export Tr, dTr_dρ, dTr_dθ, dR_dρ, dZ_dρ, dR_dθ, dZ_dθ
export Jacobian, ∇ρ, ∇ρ2, ∇θ, ∇θ2, gρρ, gρθ, gθθ, gρρ_gρθ, gρθ_gθθ, gρρ_gρθ_gθθ

include("relative.jl")
export in_surface, nearest_angle

end # module