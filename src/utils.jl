using LinearAlgebra, SparseArrays

export vec2mat, vec2matinv, vec2matadj

"""
    vec2mat(v)
Convert the vector `v` to a symmetric matrix by stacking the entries.

# Examples
```jldoctest
julia> vec2mat([1, 2, 3, 4, 5, 6])
3×3 Array{Int64,2}:
 1  2  4
 2  3  5
 4  5  6
```
"""
function vec2mat(v::SparseVector{Tv,Ti}) where {Tv, Ti}
    (indices, values) = findnz(v)
    sd = Int( 0.5*(sqrt(8*length(v)+1) - 1) )
    mat = spzeros(Tv, sd,sd)
    col = 1
    total = 0
    for i in 1:length(indices)
        idx = indices[i]
        while !(idx in total+1:total+col)
            total += col
            col += 1
        end
        mat[col, idx-total] = values[i]
        mat[idx-total, col] = values[i]
    end
    return mat
end
function vec2mat(v::Array{T,1}) where T
    sd = Int( 0.5*(sqrt(8*length(v)+1) - 1) )
    mat = zeros(T, sd,sd)
    col = 1
    total = 0
    for j = 1:sd
        mat[1:j,j] = v[total+1:total+j]
        mat[j,1:(j-1)] = v[total+1:total+j-1]
        total += j
    end
    return mat
end

"""
    vec2matadj(M)
Convert matrix `M` to a vector such that `dot(vec2matadj(M), u) == dot(M, vec2mat(u))`
for any vector `u`.

# Examples
```jldoctest
julia> M = [1 2; 2 3];

julia> vec2matadj(M)
3-element Array{Int64,1}:
 1
 4
 3

julia> dot(vec2matadj(M), [0, 1, 0]) == dot(M, vec2mat([0, 1, 0]))
true
```
"""
function vec2matadj(M::Union{SparseMatrixCSC{Tv,Ti}, Array{Tv,2}}) where {Tv, Ti}
    @assert issymmetric(M)
    sd = size(M,1)
    v = typeof(M) <: SparseMatrixCSC ?
    spzeros(Int( 0.5*sd *(sd+1) )) : zeros(Tv, Int( 0.5*sd *(sd+1) ))
    total = 0
    for j in 1:sd
        # Double the off-diagonal entries
        v[total+1:total+j-1] = 2 * M[1:(j-1),j]
        v[total+j] = M[j,j]
        total += j
    end
    return v
end

"""
    vec2matinv(M)
Convert matrix `M` to a vector such that `vec2mat(vec2matinv(M))` equals `M`.

# Examples
```jldoctest
julia> M = vec2mat([1, 2, 3, 4, 5, 6])
3×3 Array{Int64,2}:
 1  2  4
 2  3  5
 4  5  6

julia> vec2matinv(M)
6-element Array{Int64,1}:
 1
 2
 3
 4
 5
 6
```
"""
function vec2matinv(M::Union{SparseMatrixCSC{Tv,Ti}, Array{Tv,2}, Symmetric{Tv,Ti}}) where {Tv, Ti}
    @assert issymmetric(M)
    sd = size(M,1)
    vecsize = sd * (sd+1) ÷ 2
    v = if M isa SparseMatrixCSC spzeros(vecsize)
    elseif M isa Array zeros(Tv, vecsize)
    else Array{Tv,1}(undef, vecsize)
    end
    total = 0
    for j in 1:sd
        v[total+1:total+j] = M[1:j,j]
        total += j
    end
    return v
end
