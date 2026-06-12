using MillerExtendedHarmonic
using Test

const MXHmod = MillerExtendedHarmonic

@testset "MillerExtendedHarmonic.jl" begin
    @testset "construction and derived properties" begin
        mxh = MXH(3.0, 0.5, 0.3, 1.8, 0.0, [0.0, 0.0], [asin(0.4), -0.1])
        @test mxh.R0 == 3.0
        @test mxh.Z0 == 0.5
        @test mxh.a ≈ 0.9
        @test mxh.minor_radius ≈ 0.9
        @test mxh.elongation == 1.8
        @test mxh.tilt == 0.0
        @test mxh.triangularity ≈ 0.4
        @test mxh.δ ≈ 0.4
        @test mxh.squareness ≈ 0.1
        @test mxh.ovality == 0.0
        @test mxh.twist == 0.0

        # property aliases write through to the underlying fields
        mxh.triangularity = 0.3
        @test mxh.s[1] ≈ asin(0.3)
        mxh.squareness = 0.05
        @test mxh.s[2] ≈ -0.05
        mxh.ovality = 0.02
        @test mxh.c[1] == 0.02
        mxh.minor_radius = 1.2
        @test mxh.ϵ ≈ 1.2 / 3.0

        # MXH(R0, n_coeffs) gives an unshaped default surface
        m0 = MXH(3.0, 2)
        @test m0.κ == 1.0 && m0.ϵ == 0.3 && length(m0.c) == 2 && length(m0.s) == 2

        # mixed-type inputs promote
        mi = MXH(3, 0.5, 0.3, 1.8, 0, [0.0], [0.0])
        @test mi.R0 === 3.0 && mi.c0 === 0.0
    end

    @testset "flat_coeffs round-trip and copy_MXH!" begin
        mxh = MXH(3.0, 0.5, 0.3, 1.8, 0.1, [0.02, -0.01], [0.3, -0.05])
        flat = flat_coeffs(mxh)
        @test flat == [3.0, 0.5, 0.3, 1.8, 0.1, 0.02, -0.01, 0.3, -0.05]

        buf = zeros(9)
        @test flat_coeffs!(buf, mxh) == flat

        back = MXH(flat)
        @test flat_coeffs(back) == flat

        dest = MXH(1.0, 2)
        copy_MXH!(dest, mxh)
        @test flat_coeffs(dest) == flat
        # copy_MXH! copies values, it must not alias the vectors
        dest.c[1] = 99.0
        @test mxh.c[1] == 0.02
    end

    @testset "boundary evaluation" begin
        # κ=1 and no shaping coefficients → exact circle
        R0, Z0, ϵ = 3.0, 0.5, 0.3
        a = R0 * ϵ
        circ = MXH(R0, Z0, ϵ, 1.0, 0.0, zeros(2), zeros(2))
        for θ in (0.0, π / 3, π / 2, π, 1.7π)
            r, z = circ(θ)
            @test r ≈ R0 + a * cos(θ)
            @test z ≈ Z0 - a * sin(θ)
        end

        # shaped surface: analytic extremes and closed curve
        κ = 1.8
        mxh = MXH(R0, Z0, ϵ, κ, 0.0, zeros(3), [asin(0.4), -0.05, 0.0])
        pr, pz = mxh(201)
        @test pr[1] == pr[end] && pz[1] == pz[end]
        @test maximum(pr) ≈ R0 + a rtol = 1e-8   # R(0) = R0 + a is the global max
        @test maximum(pz) ≈ Z0 + κ * a atol = 1e-3
        @test minimum(pz) ≈ Z0 - κ * a atol = 1e-3
    end

    @testset "fit round-trip" begin
        truth = MXH(3.0, 0.5, 0.32, 1.8, 0.0, [0.0, 0.0, 0.0], [asin(0.45), -0.07, 0.0])
        pr, pz = truth(400)
        for kw in ((;), (; spline=true), (; optimize_fit=true))
            fit = MXH(pr, pz, 3; kw...)
            @test fit.R0 ≈ truth.R0 rtol = 1e-3
            @test fit.Z0 ≈ truth.Z0 atol = 1e-3
            @test fit.ϵ ≈ truth.ϵ rtol = 1e-3
            @test fit.κ ≈ truth.κ rtol = 1e-3
            @test fit.c ≈ truth.c atol = 5e-3
            @test fit.s ≈ truth.s atol = 5e-3
        end
    end

    @testset "in_surface / nearest_angle" begin
        mxh = MXH(3.0, 0.5, 0.3, 1.8, 0.0, zeros(2), [asin(0.3), 0.0])
        a = mxh.a
        @test MXHmod.in_surface(mxh.R0, mxh.Z0, mxh)
        @test MXHmod.in_surface(mxh.R0 + 0.99 * a, mxh.Z0, mxh)
        @test !MXHmod.in_surface(mxh.R0 + 1.01 * a, mxh.Z0, mxh)
        @test !MXHmod.in_surface(mxh.R0, mxh.Z0 + 1.01 * mxh.κ * a, mxh)

        # a boundary point of a circle maps back to its own angle
        circ = MXH(3.0, 0.0, 0.3, 1.0, 0.0, zeros(2), zeros(2))
        θq = 0.4
        Rq = 3.0 + circ.a * cos(θq)
        Zq = -circ.a * sin(θq)
        @test MXHmod.nearest_angle(Rq, Zq, circ) ≈ θq
        # beyond the vertical extent → ±π/2
        @test MXHmod.nearest_angle(3.0, -2 * circ.a, circ) ≈ π / 2
        @test MXHmod.nearest_angle(3.0, +2 * circ.a, circ) ≈ -π / 2
    end

    @testset "metric derivatives vs finite differences" begin
        mxh = MXH(3.0, 0.5, 0.3, 1.8, 0.1, [0.02, -0.01], [0.3, -0.05])
        h = 1e-6
        for θ in (0.3, 1.2, 2.5, 4.0, 5.9)
            num = (MXHmod.R_MXH(θ + h, mxh) - MXHmod.R_MXH(θ - h, mxh)) / (2h)
            @test MXHmod.dR_dθ(θ, mxh.R0, mxh.ϵ, mxh.c0, mxh.c, mxh.s) ≈ num atol = 1e-5
            @test MXHmod.dZ_dθ(θ, mxh.R0, mxh.ϵ, mxh.κ) ≈ -mxh.κ * mxh.R0 * mxh.ϵ * cos(θ) atol = 1e-12
        end
    end
end
