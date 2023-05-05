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

function find_extremum(xm, x0, xp, ym, y0, yp)
    a = ((yp - y0) * (xm - x0) - (ym - y0) * (xp - x0)) / ((xp - xm) * (xp - x0) * (xm - x0))
    b = (ym - y0) / (xm - x0)
    xext = 0.5 * (x0 + xm - b / a)
    yext = (a * (xext - xm) + b ) * (xext - x0) + y0
    return xext, yext
end

function find_extrema(R, Z)

    function im_ip(i0)
        N = length(R)
        periodic = (R[1]==R[N] && Z[1]==Z[N])
        im = 0
        ip = 0
        if i0 == 1
            periodic ? im = N-1 : im = N
        else
            im = i0 - 1
        end
        if i0 == N
            periodic ? ip = 2 : ip = 1
        else
            ip = i0 + 1
        end
        return im, ip
    end

    i0 = argmin(R)
    im, ip = im_ip(i0)
    xm = Z[im]
    x0 = Z[i0]
    xp = Z[ip]
    ym = R[im]
    y0 = R[i0]
    yp = R[ip]
    z_at_rmin, rmin = find_extremum(xm, x0, xp, ym, y0, yp)

    i0 = argmax(R)
    im, ip = im_ip(i0)
    xm = Z[im]
    x0 = Z[i0]
    xp = Z[ip]
    ym = R[im]
    y0 = R[i0]
    yp = R[ip]
    z_at_rmax, rmax = find_extremum(xm, x0, xp, ym, y0, yp)

    i0 = argmin(Z)
    im, ip = im_ip(i0)
    xm = R[im]
    x0 = R[i0]
    xp = R[ip]
    ym = Z[im]
    y0 = Z[i0]
    yp = Z[ip]
    r_at_zmin, zmin = find_extremum(xm, x0, xp, ym, y0, yp)

    i0 = argmax(Z)
    im, ip = im_ip(i0)
    xm = R[im]
    x0 = R[i0]
    xp = R[ip]
    ym = Z[im]
    y0 = Z[i0]
    yp = Z[ip]
    r_at_zmax, zmax = find_extremum(xm, x0, xp, ym, y0, yp)

    return (rmax, z_at_rmax), (r_at_zmax, zmax), (rmin, z_at_rmin), (r_at_zmin, zmin)
end

"""
    MXH(pr::Vector{T}, pz::Vector{T}, MXH_modes::Integer) where {T<:Real}

Compute Fourier coefficients for Miller-extended-harmonic representation:

    R(r,θ) = R(r) + a(r)*cos(θᵣ(r,θ)) where θᵣ(r,θ) = θ + C₀(r) + sum[Cᵢ(r)*cos(i*θ) + Sᵢ(r)*sin(i*θ)]
    Z(r,θ) = Z(r) - κ(r)*a(r)*sin(θ)

Where pr,pz are the flux surface coordinates and MXH_modes is the number of modes
"""
function MXH(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, MXH_modes::Integer=5; θ=nothing, Δθᵣ=nothing, dθ=nothing, Fm=nothing, optimize_fit=false)
    sin_coeffs = zeros(MXH_modes)
    cos_coeffs = zeros(MXH_modes)
    mxh = MXH(0.0, 0.0, 0.0, 0.0, 0.0, cos_coeffs, sin_coeffs)
    return MXH!(mxh, pr, pz; θ, Δθᵣ, dθ, Fm, optimize_fit)
end

function MXH!(mxh::MXH, pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real};
              θ=nothing, Δθᵣ=nothing, dθ=nothing, Fm=nothing,
              optimize_fit=false)
    rmin = 0.0
    rmax = 0.0
    zmin = 0.0
    zmax = 0.0
    if optimize_fit
        outer, top, inner, bottom = find_extrema(pr, pz)
        rmin, _ = inner
        rmax, _ = outer
        _, zmin = bottom
        _, zmax = top
    else
        rmin = minimum(pr)
        rmax = maximum(pr)
        zmin = minimum(pz)
        zmax = maximum(pz)
    end
    R0 = 0.5 * (rmax + rmin)
    Z0 = 0.5 * (zmax + zmin)
    a  = 0.5 * (rmax - rmin)
    b  = 0.5 * (zmax - zmin)
    return MXH!(mxh, pr, pz, R0, Z0, a, b, θ, Δθᵣ, dθ, Fm, optimize_fit)
end

function clockwise!(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real})
    @assert length(pr) == length(pz)

    # flip to clockwise so θ will increase
    @views Rmax = argmax(pr[1:end-1])
    if pz[Rmax+1] > pz[Rmax]
        reverse!(pr)
        reverse!(pz)
    end

    return pr, pz
end

function reorder_flux_surface!(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real})
    return reorder_flux_surface!(pr, pz, sum(pr)/length(pr), sum(pz)/length(pz))
end

function reorder_flux_surface!(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, R0::Real, Z0::Real)
    # flip to clockwise so θ will increase
    clockwise!(pr, pz)

    # find point closest to the midplane (1st quadrant)
    @views istart = argmin(abs.(pz[1:end-1] .- Z0) .+ (pr[1:end-1] .< R0) .+ (pz[1:end-1] .< Z0))

    # sort points in flux surface so that istart is the first point
    reorder_flux_surface!(pr, pz, istart)

    return pr, pz
end

function reorder_flux_surface!(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, istart::Int)
    # flip to clockwise so θ will increase
    clockwise!(pr, pz)

    # start from low-field side point above z0 (only if flux surface closes)
    if (pr[1] == pr[end]) && (pz[1] == pz[end])
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
    jrmax = argmax(pr)
    jzmin = argmin(pz)
    jrmin = argmin(pr)
    jzmax = argmax(pz)

    branch = jrmax < jzmax ? 0 : 1
    @inbounds for j in eachindex(θ)
        aa = (Z0 - pz[j]) / b
        aa = max(-1, min(1, aa))
        th = asin(aa)
        bb = (pr[j] - R0) / a
        bb = max(-1, min(1, bb))
        thr = acos(bb)
        if branch == 0
            (j > jrmax) && (branch = 1)
        elseif branch == 1
            (j > jzmin) && (branch = 2)
        elseif branch == 2
            (j > jrmin) && (branch = 3)
        elseif branch == 3
            (j > jzmax) && (branch = 4)
        elseif branch == 4
            (j > jrmax && jrmax > jzmax) && (branch = 5)
        end

        if branch == 0
            @inbounds θ[j] = th
            @inbounds Δθᵣ[j] = -thr
        elseif branch == 1
            @inbounds θ[j] = th
            @inbounds Δθᵣ[j] = thr
        elseif branch == 2
            @inbounds θ[j] = π - th
            @inbounds Δθᵣ[j] = thr
        elseif branch == 3
            @inbounds θ[j] = π - th
            @inbounds Δθᵣ[j] = 2π - thr
        elseif branch == 4
            @inbounds θ[j] = 2π + th
            @inbounds Δθᵣ[j] = 2π - thr
        elseif branch == 5
            @inbounds θ[j] = 2π + th
            @inbounds Δθᵣ[j] = 2π + thr
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

function MXH(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, R0::Real, Z0::Real, a::Real, b::Real, MXH_modes::Integer;
             θ=nothing, Δθᵣ=nothing, dθ=nothing, Fm=nothing)

    sin_coeffs = zeros(MXH_modes)
    cos_coeffs = zeros(MXH_modes)
    mxh = MXH(0.0, 0.0, 0.0, 0.0, 0.0, cos_coeffs, sin_coeffs)
    return MXH!(mxh, pr, pz, R0, Z0, a, b, θ, Δθᵣ, dθ, Fm)
end

function MXH!(mxh::MXH, pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, R0::Real, Z0::Real, a::Real, b::Real,
              θ::Nothing=nothing, Δθᵣ::Nothing=nothing, dθ::Nothing=nothing, Fm::Nothing=nothing, optimize_fit=false)
    MXH!(mxh, pr, pz, R0, Z0, a, b, similar(pr), similar(pr), similar(pr), similar(pr), optimize_fit)
end

function MXH!(mxh::MXH, pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, R0::Real, Z0::Real, a::Real, b::Real,
              θ::AbstractVector{<:Real}, Δθᵣ::AbstractVector{<:Real}, dθ::AbstractVector{<:Real}, Fm::AbstractVector{<:Real}, optimize_fit=false)

    @assert length(pr) == length(pz)

    mxh.R0 = R0
    mxh.Z0 = Z0
    mxh.ϵ = a / R0
    mxh.κ = b / a

    reorder_flux_surface!(pr, pz, R0, Z0)

    # Calculate angles with proper branches
    MXH_angles!(θ, Δθᵣ, pr, pz, R0, Z0, a, b)

    @inbounds @views for j in eachindex(dθ)[1:end-1]
        dθ[j] = θ[j+1] - θ[j]
        dθ[j] < 0 && (dθ[j] += 2π)
    end
    dθ[end] = dθ[1]

    Fm .= 1.0  # cos(0 * θ)
    mxh.c0 = MXH_moment(Δθᵣ, Fm, dθ)

    MXH_coeffs!(mxh.s, mxh.c, θ, Δθᵣ, dθ; Fm)

    if optimize_fit
        flat = flat_coeffs(mxh)
        optimize_fit!(flat, pr, pz; debug=true)
        R0, Z0, ϵ, κ, c0, c, s = unflatten_view(flat)
        mxh.R0 = R0
        mxh.Z0 = Z0
        mxh.ϵ = ϵ
        mxh.κ = κ
        mxh.c0 = c0
        mxh.c .= c
        mxh.s .= s
    end

    return mxh
end

function fit_residual(x, pr, pz)

    Rmin = x[1]
    Rmax = x[2]
    Zmin = x[3]
    Zmax = x[4]
    c0 = x[5]
    L = (length(x) - 5) ÷ 2
    @views c = x[6:(5 + L)]
    @views s = x[(6 + L):(5 + 2L)]
    R0 = 0.5 * (Rmax + Rmin)
    a  = 0.5 * (Rmax - Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    b  = 0.5 * (Zmax - Zmin)
    κ = b / a

    R_at_Zmin = R_MXH(0.5 * π, R0, c0, c, s, a)
    R_at_Zmax = R_MXH(-0.5 * π, R0, c0, c, s, a)

    res = 0.0

    #j_zmax = argmax(pz)
    #j_zmin = argmin(pz)
    @inbounds for j in eachindex(pr)
        #(j == j_zmax || j == j_zmin) && continue
        aa = (Z0 - pz[j]) / b
        aa = max(-1, min(1, aa))
        th = asin(aa)
        if pz[j] <= Z0
            pr[j] < R_at_Zmin && (th = π - th)
        else
            pr[j] <= R_at_Zmax ? (th = π - th) : (th += 2π)
        end
        fac = 1.0
        #D_cs_sum(th, c, s) < -1.0 && (fac = 1e4)
        res += fac * ((R_MXH(th, R0, c0, c, s, a) - pr[j]) ^ 2 + (Z_MXH(th, Z0, κ, a) - pz[j]) ^ 2)
    end

    return res
end

function optimize_fit!(flat::AbstractVector{<:Real}, pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}; inner_optimizer=Optim.LBFGS(), debug=false)

    f = x -> fit_residual(x, pr, pz)

    R0 = flat[1]
    Z0 = flat[2]
    a = R0 * flat[3]
    b = a * flat[4]
    δa = 0.1 * a
    δb = 0.1 * b
    Rmin = R0 - a
    Rmax = R0 + a
    Zmin = Z0 - b
    Zmax = Z0 + b

    lower = zero(flat)
    upper = zero(flat)

    lower[1] = Rmin - δa
    flat[1]  = Rmin
    upper[1] = Rmin + δa
    lower[2] = Rmax - δa
    flat[2]  = Rmax
    upper[2] = Rmax + δa
    lower[3] = Zmin - δb
    flat[3]  = Zmin
    upper[3] = Zmin + δb
    lower[4] = Zmax - δb
    flat[4]  = Zmax
    upper[4] = Zmax + δb
    lower[5:end] .= -0.5 * π
    upper[5:end] .=  0.5 * π
#    debug && println("Original: ", f(flat))
    M0 = 5
    M = (length(flat) - 5) ÷ 2
    if M > M0
        flat[(6+M0):(5 + M)] .= 0
        flat[(6 + M + M0):(5 + 2M)] .= 0
    end

    algo = Optim.Fminbox(inner_optimizer)
    options = Optim.Options()#store_trace=true, show_trace=true)
    res = Optim.optimize(f, lower, upper, flat, algo, options; autodiff = :forward)
    #debug && println("Residual: ", f(res.minimizer))

    Rmin = res.minimizer[1]
    Rmax = res.minimizer[2]
    Zmin = res.minimizer[3]
    Zmax = res.minimizer[4]
    R0 = 0.5 * (Rmax + Rmin)
    a = 0.5 * (Rmax - Rmin)
    b = 0.5 * (Zmax - Zmin)
    flat[1] = R0
    flat[2] = 0.5 * (Zmax + Zmin)
    flat[3] = a / R0
    flat[4] = b / a
    @views flat[5:end] .= res.minimizer[5:end]
    return
end

function fit_flattened!(flat::AbstractVector{<:Real}, pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real},
                        θ::AbstractVector{<:Real}, Δθᵣ::AbstractVector{<:Real}, dθ::AbstractVector{<:Real}, Fm::AbstractVector{<:Real};
                        rmin::Real=minimum(pr), rmax::Real=maximum(pr), zmin::Real=minimum(pz), zmax::Real=maximum(pz))

    @assert length(pr) == length(pz)

    R0 = 0.5 * (rmax + rmin)
    Z0 = 0.5 * (zmax + zmin)
    a = 0.5 * (rmax - rmin)
    b = 0.5 * (zmax - zmin)

    flat[1] = R0
    flat[2] = Z0
    flat[3] = a / R0
    flat[4] = b / a

    reorder_flux_surface!(pr, pz, R0, Z0)

    # Calculate angles with proper branches
    MXH_angles!(θ, Δθᵣ, pr, pz, R0, Z0, a, b)

    @inbounds @views for j in eachindex(dθ)[1:end-1]
    dθ[j] = θ[j+1] - θ[j]
    dθ[j] < 0 && (dθ[j] += 2π)
    end
    dθ[end] = dθ[1]

    Fm .= 1.0  # cos(0 * θ)
    flat[5] = MXH_moment(Δθᵣ, Fm, dθ)

    L = (length(flat) - 5) ÷ 2
    @views MXH_coeffs!(flat[(6 + L):(5 + 2L)], flat[6:(5 + L)], θ, Δθᵣ, dθ; Fm)
    return flat
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

function (mxh::MXH)(N::Integer=100; adaptive::Bool = true)
    if adaptive
        step = mxh.R0 / N
        a = mxh.ϵ * mxh.R0    
        NN = Int(ceil(2π * a * mxh.κ / step / 2.0)) * 2 + 1
    else
        NN = N
    end
    Θ = LinRange(0, 2π, NN)
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
    return R_MXH(R0, a, θr)
end
@inline R_MXH(R0, a, θr) =  R0 + a * cos(θr)

@inline function cs_sum(θ::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    tot = 0.0
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        tot += dot((S, C), sincos(m * θ))
    end
    return tot
end

@inline function D_cs_sum(θ::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})
    tot = 0.0
    @inbounds for m in eachindex(c)
        S = s[m]
        C = c[m]
        tot += m * dot((-C, S), sincos(m * θ))
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