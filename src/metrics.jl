# Functions for computing derivatives, the Jacobian, and metrics
#
# N.B., Some of these require a radial derivative of the MXH coefficient, 
#       which should come from external knowledge

function ΔTr(θ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    θr = c0
    θr += cs_sum(θ, c, s)
    return θr
end

@inline function Tr(θ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    return θ + ΔTr(θ, c0, c, s)
end

@inline function dTr_dρ(θ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    return ΔTr(θ, dc0, dc, ds)
end

function dTr_dθ(θ::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    dθr_dθ = 1.0
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        dθr_dθ += dot((-C, S), sincos(m * θ))
    end
    return dθr_dθ
end

function dR_dρ(θ::Real, R0::Real, ϵ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real},
               dR0::Real, dϵ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    θr = Tr(θ, c0, c, s)
    dθr = dTr_dρ(θ, dc0, dc, ds)
    a = R0 * ϵ
    da = R0 * dϵ + dR0 * ϵ
    dR_dρ = R0 
    dR_dρ += dot((-a * dθr, da), sincos(θr))
    return dR_dρ
end

function dZ_dρ(θ::Real, R0::Real, ϵ::Real, κ::Real, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real)
    db  = dR0 * ϵ * κ
    db += R0 * dϵ * κ 
    db += R0 *  ϵ * dκ
    return dZ0 - db * sin(θ)
end

function dR_dθ(θ::Real, R0::Real, ϵ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    θr = Tr(θ, c0, c, s)
    dθr_dθ = dTr_dθ(θ, c, s)
    return -R0 * ϵ * sin(θr) * dθr_dθ
end

function dZ_dθ(θ::Real, R0::Real, ϵ::Real, κ::Real)
    return -R0 * ϵ * κ * cos(θ)
end

function Jacobian(θ::Real, R0::Real, ϵ::Real, κ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real},
                  dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    a = ϵ * R0 
    R_ρ = dR_dρ(θ, R0, ϵ, c0,c, s, dR0, dϵ, dc0, dc, ds)
    R_θ = dR_dθ(θ, R0, ϵ, c0, c, s)
    Z_ρ = dZ_dρ(θ, R0, ϵ, κ, dR0, dZ0, dϵ, dκ)
    Z_θ = dZ_dθ(θ, R0, ϵ, κ)
    R = R_MXH(θ, R0, c0, c, s, a)

    return R * (R_θ * Z_ρ - Z_θ * R_ρ)
end

function ∇ρ(θ::Real, R0::Real, ϵ::Real, κ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real},
            dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    gr2 = ∇ρ2(θ, R0, ϵ, κ, c0, c, s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
    return sqrt(gr2)
end

function ∇ρ2(θ::Real, R0::Real, ϵ::Real, κ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real},
    dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    a = ϵ * R0 
    R = R_MXH(θ, R0, c0, c, s, a)
    J = Jacobian(θ, R0, ϵ, κ, c0, c, s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
    R_θ = dR_dθ(θ, R0, ϵ, c0, c, s)
    Z_θ = dZ_dθ(θ, R0, ϵ, κ)
    gr2 = (R / J)^2
    gr2 *= R_θ^2 + Z_θ^2
    return gr2
end

function ∇θ(θ::Real, R0::Real, ϵ::Real, κ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real},
            dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    gt2 = ∇θ2(θ, R0, ϵ, κ, c0, c, s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
    return sqrt(gt2)
end

function ∇θ2(θ::Real, R0::Real, ϵ::Real, κ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real},
            dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    a = ϵ * R0 
    R = R_MXH(θ, R0, c0, c, s, a)
    J = Jacobian(θ, R0, ϵ, κ, c0, c, s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
    R_ρ = dR_dρ(θ, R0, ϵ, c0,c, s, dR0, dϵ, dc0, dc, ds)
    Z_ρ = dZ_dρ(θ, R0, ϵ, κ, dR0, dZ0, dϵ, dκ)

    gt2 = (R / J)^2
    gt2 *= R_ρ^2 + Z_ρ^2
    return gt2
end