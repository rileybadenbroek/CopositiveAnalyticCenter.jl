module CopositiveAnalyticCenter

using LinearAlgebra, Gurobi
include("./accp.jl")
include("./utils.jl")

export completely_positive_cut, is_completely_positive, iscopositive, CopositiveChecker

"""
    CopositiveChecker(sd::Int)
Initialize a copositivity checker for matrices with side dimension `sd`. Sets up
```math
minimize   -μ
subject to Ax + μ e - λ = 0
           e'x = 1
           0 ≤ x_j ≤ z_j
           0 ≤ λ_j ≤ (1-z_j)M_j
           z_j ∈ {0,1},
```
to check some matrix ``A`` for copositivity. It can be shown[^1] that for
suitable ``M_j`` this is equivalent to
```math
minimize   x'Ax
subject to e'x = 1
           x ≥ 0.
```
We can bound ``λ`` as follows: it follows from stationarity of the Lagrangian
that ``λ = μe + Ax``, such that ``e'x = 1`` and ``λ∘x = 0`` imply ``0 = μ +
x'Ax``. Then ``|μ| ≤ max_ij |A_ij|`` and ``λ_i ≤ max_ij |A_ij| + (Ax)_i ≤ max_ij
|A_ij| + max_j A_ij``.

[^1]:  Wei Xia, Juan Vera, and Luis F. Zuluaga. Globally solving Non-Convex
    Quadratic Programs via Linear Integer Programming techniques. **INFORMS
    Journal on Computing**, 2018.
"""
mutable struct CopositiveChecker
    side_dimension
    model
    x
    z
    KKT_con
    λ_bound
    λ_RHS_const
    λ
    function CopositiveChecker(sd::Int)
        model = Model(optimizer_with_attributes(Gurobi.Optimizer,
            "OutputFlag"=>0, "NumericFocus"=>3))
        @variable(model, x[1:sd] >= 0)
        @variable(model, μ)
        @variable(model, λ[1:sd] >= 0)
        @variable(model, z[1:sd], Bin)
        @variable(model, λ_RHS_const[1:sd])
        @constraint(model, sum(x) == 1)
        @constraint(model, [i=1:sd], x[i] <= z[i])
        @objective(model, Min, -μ)
        @constraint(model, KKT_con, μ .- λ .== 0)
        @constraint(model, λ_bound[i=1:sd], λ[i] <= λ_RHS_const[i] )
        new(sd, model, x, z, KKT_con, λ_bound, λ_RHS_const, λ)
    end
end
"""
    iscopositive(A::AbstractMatrix, cc::CopositiveChecker; maxcoef=2^32)
Set up the CopositiveChecker `cc` to check `A`, and call `optimize_copositive!`.
If any coefficient exceeds `maxcoef`, the matrix is scaled first.

See also: [`CopositiveChecker`](@ref), [`optimize_copositive!`](@ref)
"""
function iscopositive(A::AbstractMatrix, cc::CopositiveChecker; maxcoef=2^32)
    M = maximum(abs.(A))
    if M > maxcoef
        A *= maxcoef/M
        M = maximum(abs.(A))
    end
    for i in 1:cc.side_dimension
        for j in 1:cc.side_dimension
            set_normalized_coefficient(cc.KKT_con[i], cc.x[j], A[j,i])
        end
        set_normalized_coefficient(cc.λ_bound[i], cc.z[i], M + maximum(A[:,i]))
        fix(cc.λ_RHS_const[i], M + maximum(A[:,i]))
    end
    return optimize_copositive!(cc)
end
function iscopositive(A::AbstractArray{T,2}) where T
    @assert issymmetric(A)
    cc = CopositiveChecker(size(A,1))
    return iscopositive(A, cc)
end

struct CopositiveStatusError <: Exception
    status::MOI.TerminationStatusCode
end
Base.showerror(io::IO, e::CopositiveStatusError) = println(io,
"could not complete copositivity check: a problem terminated with status $(e.status)")

"""
    optimize_copositive!(cc::CopositiveChecker, ub=Inf)
Solve the model in `cc` such that ``x`` and ``λ`` are complementary and ``z`` is
binary. Refine the solution if ``x`` and ``λ`` are not complementary and the
value of that solution is below `ub`.

See also: [`CopositiveChecker`](@ref)
"""
function optimize_copositive!(cc::CopositiveChecker, ub=Inf)
    optimize!(cc.model)
    if termination_status(cc.model) != MOI.OPTIMAL
        throw(CopositiveStatusError(termination_status(cc.model)))
    end
    if any(value(cc.x[i]) > 0 && value(cc.λ[i]) > 0 for i in 1:cc.side_dimension) &&
    objective_value(cc.model) < ub
        # Fix the values of z, and the values of x and λ if z forces them to 0
        fix!(cc)
        optimize!(cc.model)
        if termination_status(cc.model) == MOI.OPTIMAL
            # If the problem is feasible, we need to make sure there are no
            # other feasible solutions with a better objective value.
            obj, x = objective_value(cc.model), value.(cc.x)
        end
        # Cut off the current solution
        zval = fix_value.(cc.z)
        undofix!(cc)
        # Add cut ∑_{i: zval[i]=0} z[i] + ∑_{i: zval[i]=1} (1-z[i]) >= 1
        cut = @constraint(cc.model, sum(cc.z[zval.==0]) + sum(1 .-cc.z[zval.==1]) >= 1)
        if termination_status(cc.model) == MOI.OPTIMAL
            # Look for other feasible solutions until they are worse than the
            # one we already have
            local altobj, altx
            try
                (altobj, altx) = optimize_copositive!(cc, obj)
            catch e
                if e isa CopositiveStatusError
                    altobj = Inf
                else
                    delete(cc.model, cut)
                    rethrow(e)
                end
            end
            if altobj < obj
                obj = altobj
                x = altx
            end
        else
            # If the problem is infeasible, the computed basic solution was only
            # numerically feasible. We cut off this solution, and apply this
            # function again.
            try
                (obj, x) = optimize_copositive!(cc)
            catch e
                delete(cc.model, cut)
                rethrow(e)
            end
        end
        delete(cc.model, cut)
    else
        obj, x = objective_value(cc.model), value.(cc.x)
    end
    return (obj, x)
end
function fix!(cc::CopositiveChecker)
    for i in 1:cc.side_dimension
        fix(cc.z[i], round(value(cc.z[i])))
        if fix_value(cc.z[i]) == 0
            fix(cc.x[i], 0, force=true)
        else
            fix(cc.λ[i], 0, force=true)
        end
    end
end
function undofix!(cc::CopositiveChecker)
    for i in 1:cc.side_dimension
        if fix_value(cc.z[i]) == 0
            unfix(cc.x[i])
            set_lower_bound(cc.x[i], 0)
        else
            unfix(cc.λ[i])
            set_lower_bound(cc.λ[i], 0)
        end
        unfix(cc.z[i])
        set_binary(cc.z[i])
    end
end

"""
    is_completely_positive(A)
Test if the matrix `A` is completely positive through an analytic center cutting
plane method.

See also: [`completely_positive_cut`](@ref)
"""
function is_completely_positive(A::AbstractMatrix)
    X = completely_positive_cut(A)
    return dot(A, X) >= 0
end

"""
    completely_positive_cut(A)
Compute a copositive matrix `X` that approximately minimizes `dot(A,X)` subject
to the condition `norm(vec2matinv(X)) ≤ 1`.
# Arguments
- `useaccp=true`: uses an analytic center cutting plane method, otherwise an
  ellipsoid method.
- `verbose=true`: display output. If the analytic center computation fails, this
  includes the spectrum of the matrix in the linear system that should be solved.
- `trackcalls=false`: the method returns `(X, calls)` when `trackcalls==true`,
  where `calls` is the number of calls to `oracle`.
"""
function completely_positive_cut(A::AbstractMatrix; verbose=true, trackcalls=false,
useaccp=true)
    @assert issymmetric(A)
    obj = vec2matadj(A)
    cc = CopositiveChecker(size(A,1))
    function oracle(y::AbstractVector)
        val, x = iscopositive(vec2mat(y), cc)
        return val < 0 ? Halfspace(vec2matadj(-x*x'), 0.) : true
    end

    method = useaccp ? accp : ellipsoid_method
    res = method(obj, oracle, 1., verbose=verbose, trackcalls=trackcalls)
    return trackcalls ? (vec2mat(res[1]), res[2]) : vec2mat(res)
end

end # module
