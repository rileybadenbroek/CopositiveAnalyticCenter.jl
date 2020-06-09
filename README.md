# CopositiveAnalyticCenter.jl
Solves problems of the form:
```
minimize    dot(obj, x)
subject to  norm(x) ≤ r
            oracle(x) == true,
```
where `obj` is an `AbstractVector`, and `oracle` tests membership of some convex body. The main workhorse is the `accp` function,
which solves this problem using an Analytic Center Cutting Plane method. It yields a near-optimal vector `x`.

For some vector `x`, `oracle(x)` should return `true` if `x` lies in the feasible set, or else a `Halfspace` containing the feasible
set but not `x`. `Halfspace(slope, constant)` denotes the set `{z: dot(slope, z) ≤ constant}`.

## Installation
This package (currently) requires [Gurobi.jl](https://github.com/JuliaOpt/Gurobi.jl). Once this is set up, run
```julia
julia> import Pkg; Pkg.add("https://github.com/rileybadenbroek/CopositiveAnalyticCenter.jl")
```

## Testing copositivity
The package provides the `testcopositive` function, which may be used in defining your `oracle`. For some symmetric matrix `A`,
`testcopositive(A)` returns a `Tuple` containing the optimal value and optimal solution to
```
minimize    y' * A * y
subject to  sum(y) = 1
            y ≥ 0.
```

To avoid having to set up the problem above from scratch every time `testcopositive` is called,
you can create a `CopositiveChecker` instance `cc`, e.g. `cc = CopositiveChecker(10);` to set up an environment for testing 10-by-10
matrices. To test the 10-by-10 matrix `A` for copositivity using `cc`, call `testcopositive(A, cc)`.

## Transforming vectors to symmetric matrices
To transform a vector to a symmetric matrix, you can use the `vec2mat` function included in the package. Its inverse is `vec2matinv`.
```julia
julia> using CopositiveAnalyticCenter

julia> A = vec2mat([1, 2, 3, 4, 5, 6])
3×3 Array{Int64,2}:
 1  2  4
 2  3  5
 4  5  6

julia> vec2matinv(A)
6-element Array{Int64,1}:
 1
 2
 3
 4
 5
 6
```
The adjoint of `vec2mat` with respect to `dot` from LinearAlgebra.jl is `vec2matadj`.
```julia
julia> A = [1 2; 2 3];

julia> vec2matadj(A)
3-element Array{Int64,1}:
 1
 4
 3

julia> using LinearAlgebra: dot

julia> dot(vec2matadj(A), [0, 1, 0]) == dot(A, vec2mat([0, 1, 0]))
true
```

## Example usage
To solve the problem
```
minimize    dot(A, X)
subject to  norm(vec2matinv(X)) ≤ 1
            X is copositive,
```
use the following function:
```julia
using CopositiveAnalyticCenter

function test_completely_positive(A)
    # X = vec2mat(x), so dot(A, X) = dot(vec2matadj(A), x)
    obj = vec2matadj(A)
    cc = CopositiveChecker(size(A,1))
    function oracle(x::AbstractVector)
        # Test if X = vec2mat(x) is copositive
        val, y = testcopositive(vec2mat(x), cc)
        if val >= 0
            # If val ≥ 0, X = vec2mat(x) is copositive, so the oracle returns
            # true.
            return true
        else
            # Otherwise, y ≥ 0 satisfies y' X y < 0, while any copositive matrix
            # lies in the halfspace
            # {Z: y' Z y ≥ 0} = {vec2mat(z): dot(vec2matadj(-y*y'), z) ≤ 0}.
            return Halfspace(vec2matadj(-y*y'), 0.)
        end
    end
    r = 1.
    x = accp(obj, oracle, r)
    return vec2mat(x)
end
```
The package ships with the function `completely_positive_cut` which does the same thing as `test_completely_positive` above, but with some additional options.

Users interested in a yes-no answer to the question if `A` is completely positive can call `is_completely_positive(A)`.
