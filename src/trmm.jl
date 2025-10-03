"""
Recursive TRMM implementation using packed structure.
This module defines how to multiply a packed triangular matrix
with a right-hand side (X).
"""
module TRMM

using LinearAlgebra
import ..Pack: TriangularPacked, LevelGroup, LevelPack

export trmm_lower_packed!

# ------------------------------
# Off-diagonal GEMM update
# ------------------------------
"""
Perform the off-diagonal update for one recursion level:

For each off-diagonal block L21 and its corresponding X_top:
    X_bot += L21 * X_top

Here:
- `Loff` stores all L21 blocks stacked vertically.
- `Xtop_stack` stores all X_top stacked vertically.
- The product is done in one GEMM and then scattered back
  into the correct X_bot slices.
"""
function level_offdiag_gemm!(Lpack::TriangularPacked{T},
                             XTopStacks::Vector{Dict{Tuple{Int,Int},Matrix{T}}},
                             XBotSlices::Vector{Vector{Matrix{T}}},
                             lev::Int) where {T}
    groups = Lpack.levels[lev].groups
    for (m,k) in keys(groups)
        g = groups[(m,k)]
        Xtop_stack = XTopStacks[lev][(m,k)]
        # One large GEMM for all blocks of this shape
        Y = g.Loff * Xtop_stack

        # Scatter Y slices back into their respective X_bot blocks
        for (b, row0) in enumerate(g.row_offsets)
            rows = row0 : row0 + m - 1
            Xbot = XBotSlices[lev][g.slots[b]]
            @views Xbot .+= Y[rows, :]
        end
    end
end

# ------------------------------
# RHS packing (stub)
# ------------------------------
"""
Prepare the RHS blocks X for level-wise GEMMs.

Currently a stub: assumes Xdiag already corresponds to the leaf order.: Working on
A complete implementation would:
- Traverse the same recursion tree as `pack_lower_levels`.
- Stack all X_top blocks vertically per (m,k).
- Keep references to corresponding X_bot blocks.

- `Xdiag`: vector of leaf RHS blocks, one per leaf diagonal block of L.
- Returns:
    - `XTopStacks`: per level, dict of (m,k)=>stacked X_top matrices.
    - `XBotSlices`: per level, list of X_bot references, in slot order.
"""
function pack_rhs_for_levels(Lpack::TriangularPacked{T}, Xdiag::Vector{Matrix{T}}) where {T}
    l = Lpack.l
    leaf_idx = Ref(1)

    # per-level accumulators
    XTopStacks = [Dict{Tuple{Int,Int}, Matrix{T}}() for _ in 1:l]
    XBotSlices = [Matrix{T}[] for _ in 1:l]

    # recursive walk that mirrors pack_lower_levels
    function walk(sz::Int, lev::Int)::Matrix{T}
        if lev == l + 1
            Xleaf = Xdiag[leaf_idx[]]
            leaf_idx[] += 1
            return Xleaf
        end

        mid = cld(sz, 2)
        top_sz = mid
        bot_sz = sz - mid

        Xtop = walk(top_sz, lev+1)
        Xbot = walk(bot_sz, lev+1)

        # handle off-diagonal at this level
        for (m,k) in keys(Lpack.levels[lev].groups)
            if size(Xtop,1) == k && size(Xbot,1) == m
                dict = XTopStacks[lev]
                if haskey(dict, (m,k))
                    old = dict[(m,k)]
                    tall = Matrix{T}(undef, size(old,1)+k, size(old,2))
                    @views tall[1:size(old,1), :] .= old
                    @views tall[size(old,1)+1:end, :] .= Xtop
                    dict[(m,k)] = tall
                else
                    dict[(m,k)] = copy(Xtop)
                end
                push!(XBotSlices[lev], Xbot)
            end
        end

        return vcat(Xtop, Xbot) # placeholder, not used by caller
    end

    walk(Lpack.n, 1)
    return XTopStacks, XBotSlices
end


# ------------------------------
# Full TRMM driver
# ------------------------------
"""
Compute X := L * X where L is lower-triangular, using the packed structure.

Algorithm:
1. Apply TRMM on each leaf diagonal block (using `LowerTriangular`).
2. For each recursion level:
   - Multiply stacked L21 with stacked X_top in one GEMM.
   - Scatter results into X_bot blocks.
"""
function trmm_lower_packed!(Lpack::TriangularPacked{T}, Xdiag::Vector{Matrix{T}}) where {T}
    # Step 1: leaf TRMM
    @inbounds for (i, D) in enumerate(Lpack.leaves)
        lmul!(LowerTriangular(D), Xdiag[i])
    end

    # Step 2: off-diagonal GEMMs
    XTopStacks, XBotSlices = pack_rhs_for_levels(Lpack, Xdiag)
    for lev in 1:Lpack.l
        level_offdiag_gemm!(Lpack, XTopStacks, XBotSlices, lev)
    end

    return Xdiag
end

end # module
