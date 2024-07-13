# Functions for computing derivatives, the Jacobian, and metrics
#
# N.B., Some of these require a radial derivative of the MXH coefficient,
#       which should come from external knowledge

ΔTr(θ::Real, mxh::MXH) = ΔTr(θ, mxh.c0, mxh.c, mxh.s)

function ΔTr(θ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    θr = c0
    θr += cs_sum(θ, c, s)
    return θr
end

Tr(θ::Real, mxh::MXH) = Tr(θ, mxh.c0, mxh.c, mxh.s)

@inline function Tr(θ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    return θ + ΔTr(θ, c0, c, s)
end

function Tr(θ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real}, Fsin::AbstractMatrix{<:Real}, Fcos::AbstractMatrix{<:Real})
    θr = θ + c0
    l = θindex(θ, Fsin)
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        scmt = (Fsin[m, l], Fcos[m, l])
        θr += dot((S, C), scmt)
    end
    return θr
end

dTr_dρ(θ::Real, mxh::MXH, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real}) = dTr_dρ(θ, dc0, dc, ds)

@inline function dTr_dρ(θ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    return ΔTr(θ, dc0, dc, ds)
end

dTr_dθ(θ::Real, mxh::MXH) = dTr_dθ(θ, mxh.c, mxh.s)

function dTr_dθ(θ::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    dθr_dθ = one(promote_type(typeof(θ), eltype(s), eltype(c)))
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        dθr_dθ += m * dot((-C, S), sincos(m * θ))
    end
    return dθr_dθ
end

function Tr_dTrdρ(θ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real}, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    θr = θ + c0
    dθr_dρ = dc0
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        dS = ds[m]
        dC = dc[m]
        scmt = sincos(m * θ)
        θr += dot((S, C), scmt)
        dθr_dρ += dot((dS, dC), scmt)
    end
    return θr, dθr_dρ
end

function Tr_dTrdθ(θ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    θr = θ + c0
    dθr_dθ = one(promote_type(typeof(θ), eltype(s), eltype(c)))
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        scmt = sincos(m * θ)
        θr += dot((S, C), scmt)
        dθr_dθ += m * dot((-C, S), scmt)
    end
    return θr, dθr_dθ
end

function Tr_dTrdρ_dTrdθ(θ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real}, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    θr = θ + c0
    dθr_dρ = dc0
    dθr_dθ = one(promote_type(typeof(θ), eltype(s), eltype(c))) # maintain type stability with FowardDiff
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        dS = ds[m]
        dC = dc[m]
        scmt = sincos(m * θ)
        θr += dot((S, C), scmt)
        dθr_dρ += dot((dS, dC), scmt)
        dθr_dθ += m * dot((-C, S), scmt)
    end
    return θr, dθr_dρ, dθr_dθ
end

@inline function θindex(θ::Real, F::AbstractMatrix{<:Real})
    _, L = size(F)
    return θindex(θ, L)
end

@inline θindex(θ::Real, L::Int) = Int(round(θ * L * inv_twopi)) + 1

function Tr_dTrdρ_dTrdθ(
    θ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real},
    Fsin::AbstractMatrix{<:Real},
    Fcos::AbstractMatrix{<:Real}
)
    θr = θ + c0
    dθr_dρ = dc0
    dθr_dθ = one(promote_type(typeof(θ), eltype(s), eltype(c))) # maintain type stability with FowardDiff
    l = θindex(θ, Fsin)
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        dS = ds[m]
        dC = dc[m]
        scmt = (Fsin[m, l], Fcos[m, l])
        θr += dot((S, C), scmt)
        dθr_dρ += dot((dS, dC), scmt)
        dθr_dθ += m * dot((-C, S), scmt)
    end
    return θr, dθr_dρ, dθr_dθ
end

function dR_dρ(θ::Real, mxh::MXH, dR0::Real, dϵ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    return dR_dρ(θ, mxh.R0, mxh.ϵ, mxh.c0, mxh.c, mxh.s, dR0, dϵ, dc0, dc, ds)
end

function dR_dρ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dϵ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    θr, dθr_dρ = Tr_dTrdρ(θ, c0, c, s, dc0, dc, ds)
    return dR_dρ(R0, ϵ, θr, dR0, dϵ, dθr_dρ)
end

function dR_dρ(R0::Real, ϵ::Real, θr::Real, dR0::Real, dϵ::Real, dθr_dρ::Real)
    a = R0 * ϵ
    da = R0 * dϵ + dR0 * ϵ
    dR_dρ = dR0
    dR_dρ += dot((-a * dθr_dρ, da), sincos(θr))
    return dR_dρ
end

function dZ_dρ(θ::Real, mxh::MXH, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real)
    return dZ_dρ(θ, mxh.R0, mxh.ϵ, mxh.κ, dR0, dZ0, dϵ, dκ)
end

function dZ_dρ(θ::Real, R0::Real, ϵ::Real, κ::Real, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real)
    db = (dR0 * ϵ + R0 * dϵ) * κ
    db += R0 * ϵ * dκ
    return dZ0 - db * sin(θ)
end

dR_dθ(θ::Real, mxh::MXH) = dR_dθ(θ, mxh.R0, mxh.ϵ, mxh.c0, mxh.c, mxh.s)

function dR_dθ(θ::Real, R0::Real, ϵ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    θr, dθr_dθ = Tr_dTrdθ(θ, c0, c, s)
    return dR_dθ(R0, ϵ, θr, dθr_dθ)
end

@inline dR_dθ(R0::Real, ϵ::Real, θr::Real, dθr_dθ::Real) = -R0 * ϵ * sin(θr) * dθr_dθ

dZ_dθ(θ::Real, mxh::MXH) = dZ_dθ(θ, mxh.R0, mxh.ϵ, mxh.κ)

function dZ_dθ(θ::Real, R0::Real, ϵ::Real, κ::Real)
    return -R0 * ϵ * κ * cos(θ)
end

function dRdρ_dRdθ(R0::Real, ϵ::Real, θr::Real, dR0::Real, dϵ::Real, dθr_dρ::Real, dθr_dθ::Real)
    a = R0 * ϵ
    da = R0 * dϵ + dR0 * ϵ
    sctr = sincos(θr)
    R_ρ = dR0
    R_ρ += dot((-a * dθr_dρ, da), sctr)
    R_θ = -a * sctr[1] * dθr_dθ
    return R_ρ, R_θ
end

function dZdρ_dZdθ(θ::Real, R0::Real, ϵ::Real, κ::Real, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real)
    a = R0 * ϵ
    da = R0 * dϵ + dR0 * ϵ
    db = da * κ
    db += a * dκ
    st, ct = sincos(θ)
    Z_ρ = dZ0 - db * st
    Z_θ = -a * κ * ct
    return Z_ρ, Z_θ
end

function dRdρ_dRdθ_dZdρ_dZdθ(θ::Real, R0::Real, ϵ::Real, κ::Real, θr::Real, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dθr_dρ::Real, dθr_dθ::Real)
    a = R0 * ϵ
    da = R0 * dϵ + dR0 * ϵ
    sctr = sincos(θr)
    R_ρ = dR0 + dot((-a * dθr_dρ, da), sctr)
    R_θ = -a * sctr[1] * dθr_dθ

    db = da * κ
    db += a * dκ
    st, ct = sincos(θ)
    Z_ρ = dZ0 - db * st
    Z_θ = -a * κ * ct
    return R_ρ, R_θ, Z_ρ, Z_θ
end

function dRdρ_dRdθ_dZdρ_dZdθ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    θr::Real,
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dθr_dρ::Real,
    dθr_dθ::Real,
    Fsin::AbstractMatrix{<:Real},
    Fcos::AbstractMatrix{<:Real}
)
    a = R0 * ϵ
    da = R0 * dϵ + dR0 * ϵ
    sctr = sincos(θr)
    R_ρ = dR0 + dot((-a * dθr_dρ, da), sctr)
    R_θ = -a * sctr[1] * dθr_dθ

    db = da * κ
    db += a * dκ
    l = θindex(θ, Fsin)
    st = Fsin[1, l]
    ct = Fcos[1, l]
    Z_ρ = dZ0 - db * st
    Z_θ = -a * κ * ct
    return R_ρ, R_θ, Z_ρ, Z_θ
end

function Jacobian(θ::Real, mxh::MXH, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    return Jacobian(θ, mxh.R0, mxh.ϵ, mxh.κ, mxh.c0, mxh.c, mxh.s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
end

function Jacobian(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    a = ϵ * R0
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ)
    R = R_MXH(R0, a, θr)
    return Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
end

function Jacobian(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real},
    Fsin::AbstractMatrix{<:Real},
    Fcos::AbstractMatrix{<:Real}
)
    a = ϵ * R0
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds, Fsin, Fcos)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ, Fsin, Fcos)
    R = R_MXH(R0, a, θr)
    return Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
end

@inline Jacobian(R::Real, R_ρ::Real, R_θ::Real, Z_ρ::Real, Z_θ::Real) = R * (R_θ * Z_ρ - Z_θ * R_ρ)

function JacMat(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ)

    return R_ρ, R_θ, Z_ρ, Z_θ
end

function ∇ρ(θ::Real, mxh::MXH, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    return ∇ρ(θ, mxh.R0, mxh.ϵ, mxh.κ, mxh.c0, mxh.c, mxh.s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
end

function ∇ρ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    gr2 = ∇ρ2(θ, R0, ϵ, κ, c0, c, s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
    return sqrt(gr2)
end

function ∇ρ2(θ::Real, mxh::MXH, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    return ∇ρ2(θ, mxh.R0, mxh.ϵ, mxh.κ, mxh.c0, mxh.c, mxh.s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
end

function ∇ρ2(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    a = ϵ * R0
    R_θ = dR_dθ(θ, R0, ϵ, c0, c, s)
    Z_θ = dZ_dθ(θ, R0, ϵ, κ)
    gr2 = R_θ^2 + Z_θ^2
    if gr2 != 0.0
        R = R_MXH(θ, R0, c0, c, s, a)
        R_ρ = dR_dρ(θ, R0, ϵ, c0, c, s, dR0, dϵ, dc0, dc, ds)
        Z_ρ = dZ_dρ(θ, R0, ϵ, κ, dR0, dZ0, dϵ, dκ)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        gr2 *= (R / J)^2
    end
    return gr2
end

function ∇θ(θ::Real, mxh::MXH, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    return ∇θ(θ, mxh.R0, mxh.ϵ, mxh.κ, mxh.c0, mxh.c, mxh.s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
end

function ∇θ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    gt2 = ∇θ2(θ, R0, ϵ, κ, c0, c, s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
    return sqrt(gt2)
end

function ∇θ2(θ::Real, mxh::MXH, dR0::Real, dZ0::Real, dϵ::Real, dκ::Real, dc0::Real, dc::AbstractVector{<:Real}, ds::AbstractVector{<:Real})
    return ∇θ2(θ, mxh.R0, mxh.ϵ, mxh.κ, mxh.c0, mxh.c, mxh.s, dR0, dZ0, dϵ, dκ, dc0, dc, ds)
end

function ∇θ2(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    a = ϵ * R0
    R_ρ = dR_dρ(θ, R0, ϵ, c0, c, s, dR0, dϵ, dc0, dc, ds)
    Z_ρ = dZ_dρ(θ, R0, ϵ, κ, dR0, dZ0, dϵ, dκ)
    gt2 = R_ρ^2 + Z_ρ^2
    if gt2 != 0.0
        R = R_MXH(θ, R0, c0, c, s, a)
        R_θ = dR_dθ(θ, R0, ϵ, c0, c, s)
        Z_θ = dZ_dθ(θ, R0, ϵ, κ)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        gt2 *= (R / J)^2
    end
    return gt2
end

function gρρ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ)
    grr = R_θ^2 + Z_θ^2
    if grr != 0.0
        a = R0 * ϵ
        R = R_MXH(R0, a, θr)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        grr /= J
    end
    return grr
end

function gρθ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ)
    grt = -(R_ρ * R_θ + Z_ρ * Z_θ)
    if grt != 0.0
        a = R0 * ϵ
        R = R_MXH(R0, a, θr)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        grt /= J
    end
    return grt
end

function gθθ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ)
    gtt = R_ρ^2 + Z_ρ^2
    if gtt != 0.0
        a = R0 * ϵ
        R = R_MXH(R0, a, θr)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        gtt /= J
    end
    return gtt
end

function gρρ_gρθ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ)
    grr = R_θ^2 + Z_θ^2
    grt = -(R_ρ * R_θ + Z_ρ * Z_θ)
    if grr != 0.0 || grt != 0.0
        a = R0 * ϵ
        R = R_MXH(R0, a, θr)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        grr /= J
        grt /= J
    end
    return grr, grt
end

function gρθ_gθθ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ)
    grt = -(R_ρ * R_θ + Z_ρ * Z_θ)
    gtt = R_ρ^2 + Z_ρ^2
    if grt != 0.0 || gtt != 0.0
        a = R0 * ϵ
        R = R_MXH(R0, a, θr)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        grt /= J
        gtt /= J
    end
    return grt, gtt
end

function gρρ_gρθ_gθθ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real}
)
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ)
    grr = R_θ^2 + Z_θ^2
    grt = -(R_ρ * R_θ + Z_ρ * Z_θ)
    gtt = R_ρ^2 + Z_ρ^2
    if grr != 0.0
        a = R0 * ϵ
        R = R_MXH(R0, a, θr)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        grr /= J
        grt /= J
        gtt /= J
    end
    return grr, grt, gtt
end

function gρρ_gρθ_gθθ(
    θ::Real,
    R0::Real,
    ϵ::Real,
    κ::Real,
    c0::Real,
    c::AbstractVector{<:Real},
    s::AbstractVector{<:Real},
    dR0::Real,
    dZ0::Real,
    dϵ::Real,
    dκ::Real,
    dc0::Real,
    dc::AbstractVector{<:Real},
    ds::AbstractVector{<:Real},
    Fsin::AbstractMatrix{<:Real},
    Fcos::AbstractMatrix{<:Real}
)
    θr, dθr_dρ, dθr_dθ = Tr_dTrdρ_dTrdθ(θ, c0, c, s, dc0, dc, ds, Fsin, Fcos)
    R_ρ, R_θ, Z_ρ, Z_θ = dRdρ_dRdθ_dZdρ_dZdθ(θ, R0, ϵ, κ, θr, dR0, dZ0, dϵ, dκ, dθr_dρ, dθr_dθ, Fsin, Fcos)
    grr = R_θ^2 + Z_θ^2
    grt = -(R_ρ * R_θ + Z_ρ * Z_θ)
    gtt = R_ρ^2 + Z_ρ^2
    if grr != 0.0
        a = R0 * ϵ
        R = R_MXH(R0, a, θr)
        J = Jacobian(R, R_ρ, R_θ, Z_ρ, Z_θ)
        grr /= J
        grt /= J
        gtt /= J
    end
    return grr, grt, gtt
end