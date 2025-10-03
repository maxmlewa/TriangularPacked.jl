"""
TriangularPacked structure with level-wise packed off-diagonal blocks.
This module builds a hierarchical representation of a triangular matrix
suitable for level-wise GEMM operations.
"""
module Pack

using LinearAlgebra

export LevelGroup, LevelPack, TriangularPacked, pack_lower_levels

# ------------------------------
# Metadata for one shape group
# ------------------------------
"""
Represents all off-diagonal blocks at one level that share the same (m,k) shape.

- `Loff`: tall matrix obtained by vertically stacking blocks L21.
- `row_offsets`: starting row index of each block inside `Loff`.
- `mks`: block shape (m,k).
- `slots`: indices identifying which X_bot slice this block will update.
"""
struct LevelGroup{T}
    Loff::Matrix{T}
    row_offsets::Vector{Int}
    mks::Tuple{Int,Int}
    slots::Vector{Int}
end

# ------------------------------
# Metadata for one recursion level
# ------------------------------
"""
Represents one recursion level of the block structure.

- `groups`: dict of LevelGroups keyed by (m,k).
- `diag_sizes_top` and `diag_sizes_bot`: store the diagonal child sizes
   (so we can know how the matrix was subdivided).
"""
struct LevelPack{T}
    groups::Dict{Tuple{Int,Int}, LevelGroup{T}}
    diag_sizes_top::Vector{Int}
    diag_sizes_bot::Vector{Int}
end

# ------------------------------
# The full packed structure
# ------------------------------
"""
The full packed triangular matrix.

- `levels`: vector of LevelPacks (one per recursion level).
- `leaves`: the final diagonal blocks at depth l+1.
- `n`: original matrix size.
- `l`: recursion depth.
- `triangular_type`: currently only supports :lower.
"""
struct TriangularPacked{T}
    levels::Vector{LevelPack{T}}
    leaves::Vector{Matrix{T}}
    n::Int
    l::Int
    triangular_type::Symbol
end

# ------------------------------
# Main builder
# ------------------------------
"""
Pack a lower-triangular matrix `A` into `TriangularPacked` form, 
recursing to depth `l`.

At each internal node:
- Copy the off-diagonal block L21 into the appropriate LevelGroup.
- Recurse on the two diagonal children (L11 and L22).
- At leaves: copy the diagonal block into `leaves`.
"""
function pack_lower_levels(A::AbstractMatrix{T}, l::Int) where {T}
    n = size(A,1)
    n == size(A,2) || error("A must be square")

    # per-level accumulators
    groups_by_level = [Dict{Tuple{Int,Int}, LevelGroup{T}}() for _ in 1:l]
    diag_top_sizes  = [Int[] for _ in 1:l]
    diag_bot_sizes  = [Int[] for _ in 1:l]
    leaves = Matrix{T}[]

    # counter to label slots for RHS blocks
    slot_counter = Ref(1)

    # ----------------------------------
    # helper to add an off-diagonal block
    # ----------------------------------
    function add_off_block!(lev::Int, L21::AbstractMatrix{T})
        m, k = size(L21)
        key = (m,k)
        d = groups_by_level[lev]

        if haskey(d, key)
            # extend existing tall stack
            g = d[key]
            old = g.Loff
            Loff = Matrix{T}(undef, size(old,1)+m, k)
            @views Loff[1:size(old,1), :] .= old
            @views Loff[size(old,1)+1:end, :] .= L21
            row_offsets = [g.row_offsets; size(old,1)+1]
            slots = [g.slots; slot_counter[]]
            d[key] = LevelGroup{T}(Loff, row_offsets, (m,k), slots)
        else
            # create new group for this shape
            Loff = Matrix{T}(undef, m, k)
            copyto!(Loff, L21)
            row_offsets = [1]
            slots = [slot_counter[]]
            d[key] = LevelGroup{T}(Loff, row_offsets, (m,k), slots)
        end

        slot_counter[] += 1
    end

    # ----------------------------------
    # recursive subdivision
    # ----------------------------------
    function walk(i::Int, j::Int, sz::Int, lev::Int)
        if lev == l + 1
            # leaf diagonal copy
            D = Matrix{T}(undef, sz, sz)
            @views copyto!(D, A[i:i+sz-1, j:j+sz-1])
            push!(leaves, D)
            return
        end

        # split current diagonal block into quadrants
        mid = cld(sz, 2)
        i1, i2 = i, i + mid
        j1, j2 = j, j + mid
        top_sz = mid
        bot_sz = sz - mid

        # record sizes of subproblems
        push!(diag_top_sizes[lev], top_sz)
        push!(diag_bot_sizes[lev], bot_sz)

        # copy off-diagonal L21 (bottom-left block)
        if bot_sz > 0 && top_sz > 0
            @views add_off_block!(lev, A[i2:i+sz-1, j1:j1+top_sz-1])
        end

        # recurse into diagonal children
        walk(i1, j1, top_sz, lev+1)
        walk(i2, j2, bot_sz, lev+1)
    end

    # start recursion from full matrix
    walk(1,1,n,1)

    # freeze level metadata
    levels = Vector{LevelPack{T}}(undef, l)
    for lev in 1:l
        levels[lev] = LevelPack{T}(groups_by_level[lev],
                                   diag_top_sizes[lev],
                                   diag_bot_sizes[lev])
    end

    return TriangularPacked{T}(levels, leaves, n, l, :lower)
end

end # module
