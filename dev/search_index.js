var documenterSearchIndex = {"docs":
[{"location":"api/#API-Reference","page":"API Reference","title":"API Reference","text":"","category":"section"},{"location":"api/#MillerExtendedHarmonic","page":"API Reference","title":"MillerExtendedHarmonic","text":"","category":"section"},{"location":"api/","page":"API Reference","title":"API Reference","text":"MXH\nMXH!\ncopy_MXH!\nflat_coeffs\nflat_coeffs!","category":"page"},{"location":"api/#MillerExtendedHarmonic.MXH","page":"API Reference","title":"MillerExtendedHarmonic.MXH","text":"MXH(R0::T, Z0::T, ϵ::T, κ::T, c0::T, c::U, s::U) where {T<:Real,U<:AbstractVector{<:Real}}\n\nReturn MXH for Miller-extended-harmonic representation:\n\nR(θ) = R0 + R0*ϵ*cos(θᵣ(θ)) where θᵣ(θ) = θ + c0 + sum[c[m]*cos(m*θ) + s[m]*sin(m*θ)]\nZ(θ) = Z0 - κ*R0*ϵ*sin(θ)\n\n\n\n\n\nMXH(R0::Real, Z0::Real, ϵ::Real, κ::Real, c0::Real, c::AbstractVector{<:Real}, s::AbstractVector{<:Real})\n\nReturn MXH for Miller-extended-harmonic representation:\n\nR(θ) = R0 + R0*ϵ*cos(θᵣ(θ)) where θᵣ(θ) = θ + c0 + sum[c[m]*cos(m*θ) + s[m]*sin(m*θ)]\nZ(θ) = Z0 - κ*R0*ϵ*sin(θ)\n\n\n\n\n\nMXH(R0::Real, n_coeffs::Integer)\n\nReturn MXH for example Miller-extended-harmonic representation:\n\nR(θ) = R0 + 0.3*R0*cos(θ)\nZ(θ) = -0.3*R0*sin(θ)\n\nwith n_coeffs` sin/cos coefficients all set to zero\n\n\n\n\n\nMXH(flat::AbstractVector{<:Real})\n\nReturn MXH for Miller-extended-harmonic representation from flattened coefficients\n\n\n\n\n\nMXH(\npr::AbstractVector{<:Real},\npz::AbstractVector{<:Real},\nMXH_modes::Integer=5;\noptimize_fit=false,\nspline=false,\nrmin=0.0,\nrmax=0.0,\nzmin=0.0,\nzmax=0.0)\n\nCompute Fourier coefficients for Miller-extended-harmonic representation:\n\nR(θ) = R0 + R0*ϵ*cos(θᵣ(θ)) where θᵣ(θ) = θ + c0 + sum[c[m]*cos(m*θ) + s[m]*sin(m*θ)]\nZ(θ) = Z0 - κ*R0*ϵ*sin(θ)\n\nWhere pr,pz are the flux surface coordinates and MXHmodes is the number of modes. `optimizefitkeyword indicates to optimize the fit parameters to best go through the pointssplinekeyword indicates to use spline integration for modesrmin,rmax,zmin,zmax` force certain maximum and minimum values for the fit\n\n\n\n\n\n    MXH(pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, R0::Real, Z0::Real, a::Real, b::Real, MXH_modes::Integer;\n         optimize_fit=false, spline=false)\n\nCompute Fourier coefficients for Miller-extended-harmonic representation:\n\nR(θ) = R0 + R0*ϵ*cos(θᵣ(θ)) where θᵣ(θ) = θ + c0 + sum[c[m]*cos(m*θ) + s[m]*sin(m*θ)]\nZ(θ) = Z0 - κ*R0*ϵ*sin(θ)\n\nWhere pr,pz are the flux surface coordinates and MXHmodes is the number of modes. `optimizefitkeyword indicates to optimize the fit parameters to best go through the pointsspline` keyword indicates to use spline integration for modes\n\nN.B.: If optimize_fit is false, some MXH values are fixed, namely R0=R0, Z0=Z0, ϵ=a/R0, and κ=b/a       Otherwise, these are just the starting values for the optimzation\n\n\n\n\n\n(mxh::MXH)(n_points::Int=100; adaptive::Bool=true)\n\nReturns (r,z) vectors of given MXH\n\nIf adaptive, n_points is specified for the perimeter of a unit circle; and the number will never be less than n_points.\n\n\n\n\n\n","category":"type"},{"location":"api/#MillerExtendedHarmonic.MXH!","page":"API Reference","title":"MillerExtendedHarmonic.MXH!","text":"    MXH!(\n    mxh::MXH,\n    pr::AbstractVector{<:Real},\n    pz::AbstractVector{<:Real};\n    θ=similar(pr),\n    Δθᵣ=similar(pr),\n    dθ=similar(pr),\n    Fm=similar(pr),\n    optimize_fit=false,\n    spline=false,\n    rmin=0.0,\n    rmax=0.0,\n    zmin=0.0,\n    zmax=0.0)\n\nLike MXH() but operates in place. θ, Δθᵣ, dθ, Fm are work arrays vectors that can be preallocated if desired\n\n\n\n\n\n    MXH!(mxh::MXH, pr::AbstractVector{<:Real}, pz::AbstractVector{<:Real}, R0::Real, Z0::Real, a::Real, b::Real;\nθ::AbstractVector{<:Real}, Δθᵣ::AbstractVector{<:Real}, dθ::AbstractVector{<:Real}, Fm::AbstractVector{<:Real},\noptimize_fit=false, spline=false)\n\nLike MXH() but operates in place. θ, Δθᵣ, dθ, Fm are work arrays vectors that can be preallocated if desired\n\nN.B.: This function potentially reorders pr and pz in-place\n\n\n\n\n\n","category":"function"},{"location":"api/#MillerExtendedHarmonic.copy_MXH!","page":"API Reference","title":"MillerExtendedHarmonic.copy_MXH!","text":"copy_MXH!(mxh1::MXH, mxh2::MXH)\n\ncopy all attributes from mxh2 to mxh1\n\n\n\n\n\n","category":"function"},{"location":"api/#MillerExtendedHarmonic.flat_coeffs","page":"API Reference","title":"MillerExtendedHarmonic.flat_coeffs","text":"flat_coeffs(mxh::MXH)\n\nreturn all mxh coefficients as an array of floats of length 5 + 2L where L is the number of sin/cos coefficients\n\nNOTE: autodiff compatible\n\n\n\n\n\n","category":"function"},{"location":"api/#MillerExtendedHarmonic.flat_coeffs!","page":"API Reference","title":"MillerExtendedHarmonic.flat_coeffs!","text":"flat_coeffs!(flat::AbstractVector{<:Real}, mxh::MXH)\n\nreturn all mxh coefficients as an array of floats of length 5 + 2L where L is the number of sin/cos coefficients\n\nNOTE: operates in place on flat input vector and is not autodiff compatible\n\n\n\n\n\n","category":"function"},{"location":"license/","page":"License","title":"License","text":"                             Apache License\n                       Version 2.0, January 2004\n                    http://www.apache.org/licenses/","category":"page"},{"location":"license/","page":"License","title":"License","text":"TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION","category":"page"},{"location":"license/","page":"License","title":"License","text":"Definitions.\n\"License\" shall mean the terms and conditions for use, reproduction, and distribution as defined by Sections 1 through 9 of this document.\n\"Licensor\" shall mean the copyright owner or entity authorized by the copyright owner that is granting the License.\n\"Legal Entity\" shall mean the union of the acting entity and all other entities that control, are controlled by, or are under common control with that entity. For the purposes of this definition, \"control\" means (i) the power, direct or indirect, to cause the direction or management of such entity, whether by contract or otherwise, or (ii) ownership of fifty percent (50%) or more of the outstanding shares, or (iii) beneficial ownership of such entity.\n\"You\" (or \"Your\") shall mean an individual or Legal Entity exercising permissions granted by this License.\n\"Source\" form shall mean the preferred form for making modifications, including but not limited to software source code, documentation source, and configuration files.\n\"Object\" form shall mean any form resulting from mechanical transformation or translation of a Source form, including but not limited to compiled object code, generated documentation, and conversions to other media types.\n\"Work\" shall mean the work of authorship, whether in Source or Object form, made available under the License, as indicated by a copyright notice that is included in or attached to the work (an example is provided in the Appendix below).\n\"Derivative Works\" shall mean any work, whether in Source or Object form, that is based on (or derived from) the Work and for which the editorial revisions, annotations, elaborations, or other modifications represent, as a whole, an original work of authorship. For the purposes of this License, Derivative Works shall not include works that remain separable from, or merely link (or bind by name) to the interfaces of, the Work and Derivative Works thereof.\n\"Contribution\" shall mean any work of authorship, including the original version of the Work and any modifications or additions to that Work or Derivative Works thereof, that is intentionally submitted to Licensor for inclusion in the Work by the copyright owner or by an individual or Legal Entity authorized to submit on behalf of the copyright owner. For the purposes of this definition, \"submitted\" means any form of electronic, verbal, or written communication sent to the Licensor or its representatives, including but not limited to communication on electronic mailing lists, source code control systems, and issue tracking systems that are managed by, or on behalf of, the Licensor for the purpose of discussing and improving the Work, but excluding communication that is conspicuously marked or otherwise designated in writing by the copyright owner as \"Not a Contribution.\"\n\"Contributor\" shall mean Licensor and any individual or Legal Entity on behalf of whom a Contribution has been received by Licensor and subsequently incorporated within the Work.\nGrant of Copyright License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable copyright license to reproduce, prepare Derivative Works of, publicly display, publicly perform, sublicense, and distribute the Work and such Derivative Works in Source or Object form.\nGrant of Patent License. Subject to the terms and conditions of this License, each Contributor hereby grants to You a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable (except as stated in this section) patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the Work, where such license applies only to those patent claims licensable by such Contributor that are necessarily infringed by their Contribution(s) alone or by combination of their Contribution(s) with the Work to which such Contribution(s) was submitted. If You institute patent litigation against any entity (including a cross-claim or counterclaim in a lawsuit) alleging that the Work or a Contribution incorporated within the Work constitutes direct or contributory patent infringement, then any patent licenses granted to You under this License for that Work shall terminate as of the date such litigation is filed.\nRedistribution. You may reproduce and distribute copies of the Work or Derivative Works thereof in any medium, with or without modifications, and in Source or Object form, provided that You meet the following conditions:\n(a) You must give any other recipients of the Work or     Derivative Works a copy of this License; and\n(b) You must cause any modified files to carry prominent notices     stating that You changed the files; and\n(c) You must retain, in the Source form of any Derivative Works     that You distribute, all copyright, patent, trademark, and     attribution notices from the Source form of the Work,     excluding those notices that do not pertain to any part of     the Derivative Works; and\n(d) If the Work includes a \"NOTICE\" text file as part of its     distribution, then any Derivative Works that You distribute must     include a readable copy of the attribution notices contained     within such NOTICE file, excluding those notices that do not     pertain to any part of the Derivative Works, in at least one     of the following places: within a NOTICE text file distributed     as part of the Derivative Works; within the Source form or     documentation, if provided along with the Derivative Works; or,     within a display generated by the Derivative Works, if and     wherever such third-party notices normally appear. The contents     of the NOTICE file are for informational purposes only and     do not modify the License. You may add Your own attribution     notices within Derivative Works that You distribute, alongside     or as an addendum to the NOTICE text from the Work, provided     that such additional attribution notices cannot be construed     as modifying the License.\nYou may add Your own copyright statement to Your modifications and may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Derivative Works as a whole, provided Your use, reproduction, and distribution of the Work otherwise complies with the conditions stated in this License.\nSubmission of Contributions. Unless You explicitly state otherwise, any Contribution intentionally submitted for inclusion in the Work by You to the Licensor shall be under the terms and conditions of this License, without any additional terms or conditions. Notwithstanding the above, nothing herein shall supersede or modify the terms of any separate license agreement you may have executed with Licensor regarding such Contributions.\nTrademarks. This License does not grant permission to use the trade names, trademarks, service marks, or product names of the Licensor, except as required for reasonable and customary use in describing the origin of the Work and reproducing the content of the NOTICE file.\nDisclaimer of Warranty. Unless required by applicable law or agreed to in writing, Licensor provides the Work (and each Contributor provides its Contributions) on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE. You are solely responsible for determining the appropriateness of using or redistributing the Work and assume any risks associated with Your exercise of permissions under this License.\nLimitation of Liability. In no event and under no legal theory, whether in tort (including negligence), contract, or otherwise, unless required by applicable law (such as deliberate and grossly negligent acts) or agreed to in writing, shall any Contributor be liable to You for damages, including any direct, indirect, special, incidental, or consequential damages of any character arising as a result of this License or out of the use or inability to use the Work (including but not limited to damages for loss of goodwill, work stoppage, computer failure or malfunction, or any and all other commercial damages or losses), even if such Contributor has been advised of the possibility of such damages.\nAccepting Warranty or Additional Liability. While redistributing the Work or Derivative Works thereof, You may choose to offer, and charge a fee for, acceptance of support, warranty, indemnity, or other liability obligations and/or rights consistent with this License. However, in accepting such obligations, You may act only on Your own behalf and on Your sole responsibility, not on behalf of any other Contributor, and only if You agree to indemnify, defend, and hold each Contributor harmless for any liability incurred by, or claims asserted against, such Contributor by reason of your accepting any such warranty or additional liability.","category":"page"},{"location":"license/","page":"License","title":"License","text":"END OF TERMS AND CONDITIONS","category":"page"},{"location":"license/","page":"License","title":"License","text":"APPENDIX: How to apply the Apache License to your work.","category":"page"},{"location":"license/","page":"License","title":"License","text":"  To apply the Apache License to your work, attach the following\n  boilerplate notice, with the fields enclosed by brackets \"[]\"\n  replaced with your own identifying information. (Don't include\n  the brackets!)  The text should be enclosed in the appropriate\n  comment syntax for the file format. We also recommend that a\n  file or class name and description of purpose be included on the\n  same \"printed page\" as the copyright notice for easier\n  identification within third-party archives.","category":"page"},{"location":"license/","page":"License","title":"License","text":"Copyright 2024 General Atomics","category":"page"},{"location":"license/","page":"License","title":"License","text":"Licensed under the Apache License, Version 2.0 (the \"License\");    you may not use this file except in compliance with the License.    You may obtain a copy of the License at","category":"page"},{"location":"license/","page":"License","title":"License","text":"   http://www.apache.org/licenses/LICENSE-2.0","category":"page"},{"location":"license/","page":"License","title":"License","text":"Unless required by applicable law or agreed to in writing, software    distributed under the License is distributed on an \"AS IS\" BASIS,    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    See the License for the specific language governing permissions and    limitations under the License.","category":"page"},{"location":"#MillerExtendedHarmonic.jl","page":"MillerExtendedHarmonic.jl","title":"MillerExtendedHarmonic.jl","text":"","category":"section"},{"location":"","page":"MillerExtendedHarmonic.jl","title":"MillerExtendedHarmonic.jl","text":"Implementation of flux-surface parameterization suitable for local MHD equilibrium calculations with strongly-shaped flux surfaces. The method is based on a systematic expansion in a small number of intuitive shape parameters, and reduces to the well-known Miller D-shaped parameterization in the limit where some of the coefficients are set to zero. The parameterization is valid for up-down asymmetric plasmas and provides an improvement to the Miller form. Simultaneously, the method is rapidly convergent and requires only about half the number of shape parameters as a general Fourier representation in the pedestal.","category":"page"},{"location":"","page":"MillerExtendedHarmonic.jl","title":"MillerExtendedHarmonic.jl","text":"Based on https://www.osti.gov/servlets/purl/1708848","category":"page"},{"location":"#Online-documentation","page":"MillerExtendedHarmonic.jl","title":"Online documentation","text":"","category":"section"},{"location":"","page":"MillerExtendedHarmonic.jl","title":"MillerExtendedHarmonic.jl","text":"For more details, see the online documentation.","category":"page"},{"location":"","page":"MillerExtendedHarmonic.jl","title":"MillerExtendedHarmonic.jl","text":"(Image: Docs)","category":"page"},{"location":"notice/#MillerExtendedHarmonic.jl-Notice","page":"Notice","title":"MillerExtendedHarmonic.jl Notice","text":"","category":"section"},{"location":"notice/","page":"Notice","title":"Notice","text":"The purpose of this NOTICE file is to provide legal notices and acknowledgments that must be displayed to users in any derivative works or distributions. This file does not alter the terms of the Apache 2.0 license that governs the use and distribution of the MillerExtendedHarmonic.jl package.","category":"page"},{"location":"notice/#Development-Attribution","page":"Notice","title":"Development Attribution","text":"","category":"section"},{"location":"notice/","page":"Notice","title":"Notice","text":"MillerExtendedHarmonic.jl was originally developed under the FUSE project by the Magnetic Fusion Energy group at General Atomics.","category":"page"},{"location":"notice/#Citation","page":"Notice","title":"Citation","text":"","category":"section"},{"location":"notice/","page":"Notice","title":"Notice","text":"If this software contributes to an academic publication, please cite it as follows:","category":"page"},{"location":"notice/","page":"Notice","title":"Notice","text":"@article{meneghini2024fuse,\nauthor = {Meneghini, O. and Slendebroek, T. and Lyons, B.C. and McLaughlin, K. and McClenaghan, J. and Stagner, L. and Harvey, J. and Neiser, T.F. and Ghiozzi, A. and Dose, G. and Guterl, J. and Zalzali, A. and Cote, T. and Shi, N. and Weisberg, D. and Smith, S.P. and Grierson, B.A. and Candy, J.},\ndoi = {10.48550/arXiv.2409.05894},\njournal = {arXiv},\ntitle = {{FUSE (Fusion Synthesis Engine): A Next Generation Framework for Integrated Design of Fusion Pilot Plants}},\nyear = {2024}\n}","category":"page"},{"location":"notice/#Trademark-Notice","page":"Notice","title":"Trademark Notice","text":"","category":"section"},{"location":"notice/","page":"Notice","title":"Notice","text":"The names \"General Atomics\", and any associated logos or images, are trademarks of General Atomics. Use of these trademarks without prior written consent from General Atomics is strictly prohibited. Users cannot imply endorsement by General Atomics or contributors to the project simply because the project is part of their work.","category":"page"},{"location":"notice/#Copyright","page":"Notice","title":"Copyright","text":"","category":"section"},{"location":"notice/","page":"Notice","title":"Notice","text":"Copyright (c) 2024 General Atomics","category":"page"},{"location":"notice/#Version","page":"Notice","title":"Version","text":"","category":"section"},{"location":"notice/","page":"Notice","title":"Notice","text":"Version: v2.1","category":"page"}]
}
