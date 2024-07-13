function in_surface(R::Real, Z::Real, mxh::MXH)
    R0 = mxh.R0
    Z0 = mxh.Z0
    ϵ = mxh.ϵ
    κ = mxh.κ
    c0 = mxh.c0
    c = mxh.c
    s = mxh.s
    return in_surface(R, Z, R0, Z0, ϵ, κ, c0, c, s)
end

function in_surface(R::Real, Z::Real, flat::AbstractVector{<:Real})
    return in_surface(R, Z, unflatten_view(flat)...)
end

function in_surface(R::Real, Z::Real, R0::Real, Z0::Real, ϵ::Real, κ::Real, c0::Real,
    c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    a = R0 * ϵ
    abs(R - R0) > a && return false

    b = a * κ
    abs(Z - Z0) > b && return false

    θo = asin((Z0 - Z) / b)
    Ro = R_MXH(θo, R0, ϵ, c0, c, s, a)
    R > Ro && return false

    θi = (θo < 0.0 ? -1.0 : 1.0) * π - θo
    Ri = R_MXH(θi, R0, ϵ, c0, c, s, a)
    R < Ri && return false

    return true
end

"""
Returns the angle on surface at height Z closest to R
If point is above/below maximum/minimum Z of surface, return +/- π/2
"""
function nearest_angle(R::Real, Z::Real, mxh::MXH)
    R0 = mxh.R0
    Z0 = mxh.Z0
    ϵ = mxh.ϵ
    κ = mxh.κ
    c0 = mxh.c0
    c = mxh.c
    s = mxh.s
    return nearest_angle(R, Z, R0, Z0, ϵ, κ, c0, c, s)
end

function nearest_angle(R::Real, Z::Real, flat::AbstractVector{<:Real})
    return nearest_angle(R, Z, unflatten_view(flat)...)
end

function nearest_angle(R::Real, Z::Real, R0::Real, Z0::Real, ϵ::Real, κ::Real, c0::Real,
    c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    a = R0 * ϵ

    if a == 0.0
        if Z == Z0
            R >= R0 ? (return 0.0) : (return 1.0 * π) # ensures a float gets returned
        else
            return sign(Z0 - Z) * 0.5 * π
        end
    end

    b = a * κ
    aa = (Z0 - Z) / b
    abs(aa) > 1 && return sign(aa) * 0.5 * π
    θ = asin(aa)
    signθ = θ < 0.0 ? -1.0 : 1.0
    θext = signθ * 0.5 * π
    Rext = R_MXH(θext, R0, ϵ, c0, c, s, a)
    R <= Rext && (θ = signθ * π - θ)
    return θ
end