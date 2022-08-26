mutable struct MXH{T <: Real, U<:AbstractVector{<:Real}}
    R0::T  # Major Radius
    Z0::T  # Elevation
    ϵ::T   # Inverse aspect ratio a/R0
    κ::T   # Elongation
    c0::T  # Tilt
    c::U   # Cosine coefficients acos.([ovality,...])
    s::U   # Sine coefficients asin.([triangularity,-squareness,...]
    function MXH{T, U}(R0::T, Z0::T, ϵ::T, κ::T, c0::T, c::U, s::U) where {T <: Real, U<:AbstractVector{<:Real}}
        return length(c) == length(s) ? new{T, U}(R0, Z0, ϵ, κ, c0, c, s) : throw(DimensionMismatch)
    end
end

function MXH(R0::T, Z0::T, ϵ::T, κ::T, c0::T, c::U, s::U) where {T <: Real, U<:AbstractVector{<:Real}}
    return MXH{T, U}(R0, Z0, ϵ, κ, c0, c, s)
end

function MXH(R0::Real, Z0::Real, ϵ::Real, κ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    return MXH(promote(R0, Z0, ϵ, κ, c0)..., promote_vectors(c, s)...)
end

function promote_vectors(c::T, s::U) where {T<:AbstractVector{<:Real}, U<:AbstractVector{<:Real}}
    T === U && return c, s
    try
        return promote(c, s)
    catch
        return promote(_promote_vectors(c,s)...)
    end
end
_promote_vectors(c::T, s::U) where {T<:AbstractVector{<:Real}, U<:AbstractRange{<:Real}} = (c, collect(s))
_promote_vectors(c::T, s::U) where {T<:AbstractRange{<:Real}, U<:AbstractVector{<:Real}} = (collect(c), s)

function MXH(R0::Real, n_coeffs::Integer)
    return MXH(R0, 0.0, 0.3, 1.0, 0.0, zeros(n_coeffs), zeros(n_coeffs))
end

function flat_coeffs(mxh::MXH)
    L = length(mxh.c)
    flat = zeros(5 + 2L)
    return flat_coeffs!(flat, mxh)
end

function flat_coeffs!(flat::AbstractVector{<:Real}, mxh::MXH)
    L = length(mxh.c)
    @assert length(flat) == 5 + 2L
    flat[1] = mxh.R0
    flat[2] = mxh.Z0
    flat[3] = mxh.ϵ
    flat[4] = mxh.κ
    flat[5] = mxh.c0
    @views flat[6:(5 + L)] .= mxh.c
    @views flat[(6 + L):(5 + 2L)] .= mxh.s
    return flat
end

function unflatten(flat::AbstractVector{<:Real})
    R0 = flat[1]
    Z0 = flat[2]
    ϵ  = flat[3]
    κ  = flat[4]
    c0 = flat[5]
    L = (length(flat) - 5) ÷ 2
    c = flat[6:(5 + L)]
    s = flat[(6 + L):(5 + 2L)]
    return R0, Z0, ϵ, κ, c0, c, s
end

function unflatten_view(flat::AbstractVector{<:Real})
    R0 = flat[1]
    Z0 = flat[2]
    ϵ  = flat[3]
    κ  = flat[4]
    c0 = flat[5]
    L = (length(flat) - 5) ÷ 2
    @views c = flat[6:(5 + L)]
    @views s = flat[(6 + L):(5 + 2L)]
    return R0, Z0, ϵ, κ, c0, c, s
end

function MXH(flat::AbstractVector{<:Real})
    #@views R0, Z0, ϵ, κ, c0 = flat[1:5]
    #L = (length(flat) - 5) ÷ 2
    #c = flat[6:(5 + L)]
    #s = flat[(6 + L):(5 + 2L)]
    return MXH(unflatten(flat)...)
end

function copy_MXH!(mxh1::MXH, mxh2::MXH)
    mxh1.R0 = mxh2.R0
    mxh1.Z0 = mxh2.Z0
    mxh1.ϵ = mxh2.ϵ
    mxh1.κ = mxh2.κ
    mxh1.c0 = mxh2.c0
    mxh1.c .= mxh2.c
    mxh1.s .= mxh2.s
    return mxh1
end

"""
    MXH_moment(f, w, d)

This does Int[f.w]/Int[w.w]
If w is a pure Fourier mode, this gives the Fourier coefficient
"""
function MXH_moment(f::AbstractVector{<:Real}, w::AbstractVector{<:Real}, d::AbstractVector{<:Real})
    # Could probably be replaced by some Julia trapz
    N = length(f)
    @assert length(w) == length(d) == N
    @inbounds s0 = sum((f[i] * w[i] + f[i+1] * w[i+1]) * d[i] for i in 1:(N-1))
    @inbounds s1 = sum((w[i]^2 + w[i+1]^2) * d[i] for i in 1:(N-1))
    res = s0 / s1
    return res
end

"""
    MXH(pr::Vector{T}, pz::Vector{T}, MXH_modes::Integer) where {T<:Real}

Compute Fourier coefficients for Miller-extended-harmonic representation:

    R(r,θ) = R(r) + a(r)*cos(θᵣ(r,θ)) where θᵣ(r,θ) = θ + C₀(r) + sum[Cᵢ(r)*cos(i*θ) + Sᵢ(r)*sin(i*θ)]
    Z(r,θ) = Z(r) - κ(r)*a(r)*sin(θ)

Where pr,pz are the flux surface coordinates and MXH_modes is the number of modes
"""
function MXH(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, MXH_modes::Integer=5)
    R0 = 0.5 * (maximum(pr) + minimum(pr))
    Z0 = 0.5 * (maximum(pz) + minimum(pz))
    a = 0.5 * (maximum(pr) - minimum(pr))
    b = 0.5 * (maximum(pz) - minimum(pz))
    return MXH(pr, pz, R0, Z0, a, b, MXH_modes)
end


function reorder_flux_surface!(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real})
    return reorder_flux_surface!(pr, pz, sum(pr)/length(pr), sum(pz)/length(pz))
end

function reorder_flux_surface!(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, R0::Real, Z0::Real)
    @assert length(pr) == length(pz)

    # flip to clockwise so θ will increase
    @views istart = argmax(pr[1:end-1])
    if pz[istart+1] > pz[istart]
        reverse!(pr)
        reverse!(pz)
        istart = length(pr) + 1 - istart
    end

    # start from low-field side point above z0 (only if flux surface closes)
    if (pr[1] == pr[end]) && (pz[1] == pz[end])
        @views istart = argmin(abs.(pz[1:end-1] .- Z0) .+ (pr[1:end-1] .< R0) .+ (pz[1:end-1] .< Z0))
        @views pr[1:end-1] .= circshift(pr[1:end-1], 1 - istart)
        @views pz[1:end-1] .= circshift(pz[1:end-1], 1 - istart)
        pr[end] = pr[1]
        pz[end] = pz[1]
    end

    return pr, pz
end

function MXH_angles!(θ::AbstractVector{<:Real}, Δθᵣ::AbstractVector{<:Real},
                     pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real},
                     R0::Real, Z0::Real, a::Real, b::Real)
    @assert length(θ) == length(Δθᵣ) == length(pr) == length(pz)
    th = 0.0
    thr = 0.0
    @inbounds for j in eachindex(θ)
        th_old = th
        thr_old = thr
        aa = (Z0 - pz[j]) / b
        aa = max(-1, min(1, aa))
        th = asin(aa)
        bb = (pr[j] - R0) / a
        bb = max(-1, min(1, bb))
        thr = acos(bb)
        if (j == 1) || ((th > th_old) && (thr > thr_old))
            @inbounds θ[j] = th
            @inbounds Δθᵣ[j] = thr
        elseif (th < th_old) && (thr > thr_old)
            @inbounds θ[j] = π - th
            @inbounds Δθᵣ[j] = thr
        elseif (th < th_old) && (thr < thr_old)
            @inbounds θ[j] = π - th
            @inbounds Δθᵣ[j] = 2π - thr
        elseif (th > th_old) && (thr < thr_old)
            @inbounds θ[j] = 2π + th
            @inbounds Δθᵣ[j] = 2π - thr
        end
        @inbounds Δθᵣ[j] -= θ[j]
    end
end

function MXH_coeffs!(sin_coeffs::AbstractVector{<:Real}, cos_coeffs::AbstractVector{<:Real},
                    θ::AbstractVector{<:Real}, Δθᵣ::AbstractVector{<:Real}, dθ::AbstractVector{<:Real};
                    Fm::Union{AbstractVector{<:Real}, Nothing}=nothing)
    @assert length(sin_coeffs) == length(cos_coeffs)
    Fm === nothing && (Fm = similar(θ))
    @inbounds for m in eachindex(sin_coeffs)
        Fm .= sin.(m .* θ)
        sin_coeffs[m] = MXH_moment(Δθᵣ, Fm, dθ)

        Fm .= cos.(m .* θ)
        cos_coeffs[m] = MXH_moment(Δθᵣ, Fm, dθ)
    end
end

function MXH(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, R0::Real, Z0::Real, a::Real, b::Real, MXH_modes::Integer)

    @assert length(pr) == length(pz)

    reorder_flux_surface!(pr, pz, R0, Z0)

    θ = similar(pr)
    Δθᵣ = similar(pr)

    # Calculate angles with proper branches
    MXH_angles!(θ, Δθᵣ, pr, pz, R0, Z0, a, b)

    dθ = similar(θ)
    @inbounds for j in eachindex(dθ)[1:end-1]
        dθ[j] = θ[j+1] - θ[j]
        dθ[j] < 0 && (dθ[j] += 2π)
    end
    dθ[end] = dθ[1]

    Fm = similar(θ)
    Fm .= 1.0  # cos(0 * θ)
    tilt = MXH_moment(Δθᵣ, Fm, dθ)

    sin_coeffs = zeros(MXH_modes)
    cos_coeffs = zeros(MXH_modes)
    MXH_coeffs!(sin_coeffs, cos_coeffs, θ, Δθᵣ, dθ; Fm)

    return MXH(R0, Z0, a / R0, b / a, tilt, cos_coeffs, sin_coeffs)
end

@recipe function plot_mxh(mxh::MXH; adaptive_grid_N=100)
    @series begin
        aspect_ratio --> :equal
        label --> ""
        mxh(adaptive_grid_N)
    end
end

function Base.show(io::IO, mxh::MXH)
    println(io, "R0: $(mxh.R0)")
    println(io, "Z0: $(mxh.Z0)")
    println(io, "ϵ: $(mxh.ϵ)")
    println(io, "κ: $(mxh.κ)")
    println(io, "c0: $(mxh.c0)")
    println(io, "c: $(mxh.c)")
    println(io, "s: $(mxh.s)")
end

function (mxh::MXH)(adaptive_grid_N::Integer=100)
    step = mxh.R0 / adaptive_grid_N
    a = mxh.ϵ * mxh.R0
    N = Int(ceil(2π * a * mxh.κ / step / 2.0)) * 2 + 1
    Θ = LinRange(0, 2π, N)
    tmp = mxh.(Θ)
    tmp[end] = tmp[1]
    return [r for (r,z) in tmp],[z for (r,z) in tmp]
end

function R_MXH(θ::Real, mxh::MXH, a=nothing)
    return R_MXH(θ, mxh.R0, mxh.ϵ, mxh.c0, mxh.c, mxh.s, a)
end

function R_MXH(θ::Real, flat::AbstractVector{<:Real}, a=nothing)
    # L = (length(flat) - 5) ÷ 2
    # R0 = flat[1]
    # ϵ  = flat[3]
    # c0 = flat[5]
    # @views r = R_MXH(θ, R0, ϵ, c0, flat[6:(5 + L)], flat[(6 + L):(5 + 2L)], a)
    # return r
    return R_MXH(θ, unflatten_view(flat)..., a) 
end

function R_MXH(θ::Real, R0::Real, Z0::Real, ϵ::Real, κ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real}, a=nothing)
    return R_MXH(θ, R0, ϵ, c0, c, s, a)
end

function R_MXH(θ::Real, R0::Real, ϵ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real}, a::Nothing=nothing)
    return R_MXH(θ, R0, ϵ, c0, c, s, ϵ * R0)
end

@inline function R_MXH(θ::Real, R0::Real, ϵ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real}, a::Real)
    return R_MXH(θ, R0, c0, c, s, ϵ * R0)
end

@inline function R_MXH(θ::Real, R0::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real}, a::Real)
    cs = cs_sum(θ, c, s)
    θr = θ + c0 + cs
    return R0 + a * cos(θr)
end

@inline function cs_sum(θ::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    tot = 0.0
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        tot += dot((S, C), sincos(m * θ))
    end
    return tot
end

Z_MXH(θ::Real, mxh::MXH, a=nothing) = Z_MXH(θ, mxh.R0, mxh.Z0, mxh.ϵ, mxh.κ, a)

function Z_MXH(θ::Real, flat::AbstractVector{<:Real}, a=nothing)
    @views z = Z_MXH(θ, flat[1:4]..., a)
    return z
end

function Z_MXH(θ::Real, R0::Real, Z0::Real, ϵ::Real, κ::Real, c0::Real, c, s, a=nothing)
    return Z_MXH(θ, R0, Z0, ϵ, κ, a)
end

function Z_MXH(θ::Real, R0::Real, Z0::Real, ϵ::Real, κ::Real, a::Nothing=nothing)
    return Z_MXH(θ, R0, Z0, ϵ, κ, ϵ * R0)
end

@inline function Z_MXH(θ::Real, R0::Real, Z0::Real, ϵ::Real, κ::Real, a::Real)
    return Z_MXH(θ, Z0, κ, a)
end

@inline function Z_MXH(θ::Real, Z0::Real, κ::Real, a::Real)
    return Z0 - κ * a * sin(θ)
end

function (mxh::MXH)(θ::Real)
    a = mxh.ϵ * mxh.R0
    r = R_MXH(θ, mxh, a)
    z = Z_MXH(θ, mxh, a)
    return r, z
end