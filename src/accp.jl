using LinearAlgebra
using JuMP, Gurobi
using Printf

export accp, Halfspace

"""
    Halfspace(slope, constant)
Define a halfspace `{x: dot(slope, x) ≤ constant}`. Note that slope is internally
normalized.
"""
struct Halfspace{Ts<:AbstractVector,Tc<:Number}
    slope::Ts
    constant::Tc
    Halfspace(slope, constant) = new{typeof(slope),typeof(constant)}(
        slope/norm(slope), constant/norm(slope))
end

mutable struct OuterApproximation
    model::Model
    x::Array{VariableRef,1}
    cutrefs::Array{ConstraintRef,1}
    function OuterApproximation(obj::AbstractVector, A::AbstractMatrix,
    b::AbstractVector, r::Number)
        n = length(obj)
        model = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag"=>0))
        @variable(model, x[1:n])
        @constraint(model, vcat(r, x) in SecondOrderCone())
        cutrefs = @constraint(model, A * x .<= b)
        @objective(model, Min, dot(obj, x))
        new(model, x, cutrefs)
    end
end

"Compute lower bound by minimizing over the outer approximation"
function lowerbound(oa::OuterApproximation)
    optimize!(oa.model)
    if termination_status(oa.model) == MOI.OPTIMAL
        return objective_value(oa.model)
    else
        error("Could not compute lower bound."*
        " Termination status was $(termination_status(oa.model))")
    end
end
function addcut!(oa::OuterApproximation, hs::Halfspace)
    ref = @constraint(oa.model, hs.slope' * oa.x <= hs.constant)
    push!(oa.cutrefs, ref)
end

"""
    accp(obj::AbstractVector, oracle::Function, r::Number, x0::AbstractVector)
Run an analytic center cutting plane method to minimize `dot(obj, x)` such that
`norm(x) ≤ r` and `oracle(x) == true`. `oracle(x)` should either return
`true` or a `Halfspace`. The starting point `x0` does not have to be feasible.
Returns a near-optimal solution `x` at default settings.
# Arguments
- `opttol=1e-6`: the maximum relative optimality gap at termination
- `maxcons=3`: multiple of the number of variables indicating the number of
  constraints that are kept. The default `maxcons=3` keeps `3n` constraints, where
  `n` is the number of variables.
- `verbose=true`: display output. If the analytic center computation fails, this
  includes the spectrum of the matrix in the linear system that should be solved.
- `trackcalls=false`: the method returns `(x, calls)` when `trackcalls==true`,
  where `calls` is the number of calls to `oracle`.
"""
function accp(obj::AbstractVector, oracle::Function, r::Number,
x0::AbstractVector=zeros(length(obj)); opttol=1e-6, maxcons=3, verbose=true, trackcalls=false)
    n = length(obj)
    nobj = obj / norm(obj)
    x = norm(x0) > 1 ? x0 / norm(x0) : x0
    result = oracle(x)
    if result isa Halfspace
        # x is infeasible
        A = [result.slope';]
        b = [result.constant]
        local best_sol
        best_val = Inf
    else
        # x is feasible. If x ≈ -obj / norm(obj), the feasible set will be
        # (almost) empty after adding a cut.
        if x ≈ -nobj
            return x
        end
        A = [nobj';]
        b = [dot(nobj, x)]
        best_sol = x
        best_val = dot(obj, x)
    end
    trackcalls && (oraclecalls = 1)
    oa = OuterApproximation(obj, A, b, r)
    while true
        # Compute analytic center
        (status, finalnorm) = analytic_center!(A, b, r, x, verbose=verbose)
        if !status
            # Check if the point is at least in the interior of the set we
            # wanted the analytic center of
            if any(A*x .>= b) || norm(x) >= r
                verbose && println()
                error("Could not approximate analytic center: Lagrange gradient norm $finalnorm")
            end
        else
            A, b = prune(A, b, r, x, oa, ceil(Int, maxcons*n))
        end
        # Check if the analytic center is feasible
        result = oracle(x)
        trackcalls && (oraclecalls += 1)
        if result isa Halfspace
            # If the current point is not feasible, add a cut
            A = [A; result.slope']
            b = [b; result.constant]
            addcut!(oa, result)
        else
            # If the current point is feasible, tighten the objective constraint
            best_sol = x
            best_val = dot(obj, x)
            A = [A; nobj']
            b = [b; dot(nobj, x)]
            addcut!(oa, Halfspace(obj, best_val))
        end
        # Check if we are close enough to the optimum
        lb = lowerbound(oa)
        relgap = (best_val - lb) / (1 + min(abs(best_val), abs(lb)))
        verbose && @printf("\rAbs. gap: %.3e\tRel. gap: %.3e", best_val - lb, relgap)
        if relgap <= opttol
            verbose && println()
            return trackcalls ? (best_sol, oraclecalls) : best_sol
        end
    end
end

"""
    analytic_center!(A::AbstractMatrix, b::AbstractVector, r::Number, x::AbstractVector)
Approximate the analytic center of `{x: Ax ≤ b, norm(x) ≤ r}`, starting from `x`
and storing the result in `x`.
# Arguments
- `verbose=true`: displays the spectrum of the linear system if the method fails.
- `grad_atol=1e-8`: maximum norm of the Lagrangian gradient for successful termination.
- `maxiter=50`: maximum number of Newton steps.
- `α=0.9`: maximum step to boundary - should satisfy `0 < α < 1`.
"""
function analytic_center!(A::AbstractMatrix, b::AbstractVector, r::Number,
x::AbstractVector; verbose=true, grad_atol=1e-8, maxiter=50, α = 0.9)
    # We use an infeasible start Newton method, cf. page 532 in 'Convex
    # Optimization' by Boyd and Vandenberghe.
    (m,n) = size(A)
    if m == 0
        # If there are no constraints yet, the origin is the analytic center
        return (zeros(n), true, 0.)
    end
    d = r - norm(x) > sqrt(eps(Float64)) ? r^2 - x'*x : 1.
    s = [b[i] - A[i,:]'*x > sqrt(eps(Float64)) ? b[i] - A[i,:]'*x : 1. for i=1:m]
    κ = 1.
    λ = zeros(m)
    iter = 1
    LHS = Matrix{Float64}(undef, n, n)
    rhs = Vector{Float64}(undef, n)
    Δx = Vector{Float64}(undef, n)
    Δs = Vector{Float64}(undef, m)
    Δλ = Vector{Float64}(undef, m)
    while iter <= maxiter
        rhs .= ((r^2 - x'*x - d)/d^2 - 1. /d) * 2. * x + A'*(-1. ./ s + (s.^-2) .* (-s + b - A*x))
        LHS .= 4. / d^2 * x*x' + A' * ((s.^-2) .* A)
        for i in 1:n
            LHS[i,i] += 2. * κ
        end
        Δx .= LHS \ rhs
        Δd  = -d + r^2 - x'*x - 2. *x'*Δx
        Δs .= -s + b - A*x - A*Δx
        Δκ  = -κ + 1. / d - Δd / d^2
        Δλ .= -λ + 1. ./ s - Δs ./ (s.^2)
        # Let res(t) be the residual norm if we take step size t
        res = t -> norm_Lagrange_gradient(A,b,r, x+t*Δx, d+t*Δd, s+t*Δs, κ+t*Δκ, λ+t*Δλ)
        res0 = res(0)
        maxt = 1.
        t = min(maxt,
            any(Δs .< 0) ? α*minimum(-s[i]/Δs[i] for i in 1:m if Δs[i] < 0) : maxt,
            Δd < 0 ? -α*d/Δd : maxt,
            Δκ < 0 ? -α*κ/Δκ : maxt
        )

        if res0 <= min(grad_atol, res(t)) && all(λ .>= 0)
            # We have approximated the analytic center as well as we can; going
            # on is pointless
            return (true, res0)
        end
        x .+= t*Δx
        d  += t*Δd
        s .+= t*Δs
        κ  += t*Δκ
        λ .+= t*Δλ
        if iter == maxiter && res(t) <= grad_atol && all(λ .>= 0)
            # We have reached the final iteration, and we are beneath the
            # tolerance. Perhaps we could approximate the analytic center to
            # higher accuracy by continuing, but this is also fine.
            return (true, res(t))
        end
        iter += 1
    end
    verbose && @printf(" Spectrum: [%.3e, %.3e]", eigmin(LHS), eigmax(LHS))
    return (false, norm_Lagrange_gradient(A,b,r, x, d, s, κ, λ))
end
function norm_Lagrange_gradient(A::AbstractMatrix,b::AbstractVector,r::Number,
x::AbstractVector, d::Number, s::AbstractVector, κ::Number, λ::AbstractVector)
    # Returns the norm of the gradient of the Lagrangian of this problem at the
    # point (x, d, s, κ, λ). In the book by Boyd and Vandenberghe, this
    # quantity has the interpretation of the (primal-dual) residual norm.
    v1 = 2*κ * x + A' * λ
    v2 = -1 ./ s + λ
    v3 = s - b + A*x
    normsq = v1'*v1 + (-1. / d + κ)^2 + v2'*v2 + (d - r^2 + x'*x)^2 + v3'*v3
    return sqrt(normsq)
end

"""
    prune(A::AbstractMatrix, b::AbstractVector, r::Number, ac::AbstractVector, oa::OuterApproximation, maxconabs::Int=Inf)
Prune the constraints from `{x: Ax ≤ b, norm(x) ≤ r}` that are redundant, using
the set's analytic center `ac`. Also prune the least relevant constraints
such that at most `maxconabs` constraints remain. Returns `(Ap,bp)`, where `Ap`
and `bp` are the pruned versions of `A` and `b`. Also prunes the constraints
from `oa`.
"""
function prune(A::AbstractMatrix, b::AbstractVector, r::Number, ac::AbstractVector,
oa::OuterApproximation, maxconabs::Int=Inf)
    # The method follows Section 3 in Boyd, Vandenberghe, and Skaf. Returns the
    # pruned A and b.
    (m,n) = size(A)
    if m <= n
        return (A,b)
    end
    slack = b - A*ac
    H = slack.^-2 .* A
    H = sum(A[i,:] * H[i,:]' for i in 1:m)
    H += 2. / (r^2 - ac'*ac) * I + 4. / (r^2 - ac'*ac) * ac*ac'
    H_inv_AT = H \ Matrix(A')
    redcons = Int[]
    η = Float64[]
    for i = 1:m
        push!(η, slack[i] / sqrt(A[i,:]' * H_inv_AT[:,i]))
        if η[end] >= m+1
            push!(redcons, i)
        end
    end
    keepers = m - length(redcons) <= maxconabs ? setdiff(1:m, redcons) :
    sort!(sortperm(η)[1:min(maxconabs, m)])
    A = A[keepers,:]
    b = b[keepers]
    for i in 1:m
        if i ∉ keepers
            delete(oa.model, oa.cutrefs[i])
        end
    end
    oa.cutrefs = oa.cutrefs[keepers]
    return (A,b)
end

"""
    ellipsoid_method(obj::AbstractVector, oracle::Function, r::Number)
Run an ellipsoid method to minimize `dot(obj, x)` such that `norm(x) ≤ r`
and `oracle(x) == true`. `oracle(x)` should either return `true` or a `Halfspace`.
"""
function ellipsoid_method(obj::AbstractVector, oracle::Function, r::Number;
opttol=1e-6, verbose=true, trackcalls=false)
    n = length(obj)
    nobj = obj / norm(obj)
    trackcalls && (oraclecalls = 0)
    x = zeros(n)
    D = Matrix(r*I,n,n)
    # The ellipsoid we consider is {y: (y-x)' D^-1 (y-x) ≤ 1}
    while true
        # Add the constraint y: a'y >= a'x, i.e. -a'y <= -a'x
        if norm(x) > r
            a = -x / norm(x)
        else
            result = oracle(x)
            trackcalls && (oraclecalls += 1)
            if result isa Halfspace # x is infeasible
                a = -result.slope
            else # x is feasible
                # Using z = D^(-1/2)(y-x), min{obj' y: (y-x)' D^-1 (y-x) ≤ 1} =
                # min{obj' (x + D^(1/2) z): z'z ≤ 1}
                lb = obj' * x - sqrt(obj' * D * obj)
                relgap = (obj' * x - lb) / (1 + min(abs(obj' * x), abs(lb)))
                verbose && @printf("\rAbs. gap: %.3e\tRel. gap: %.3e", obj' * x - lb, relgap)
                if relgap <= opttol
                    verbose && println()
                    return trackcalls ? (x, oraclecalls) : x
                else
                    a = -nobj
                end
            end
        end
        # See Theorem 8.1 in "Introduction to Linear Optimization" by Bertsimas
        # and Tsitsiklis.
        try
            anorm = a / sqrt(a' * D * a)
            Danorm = D*anorm / (n+1)
            x .+= Danorm
            D .*= n^2/(n^2-1)
            D .+= - 2 * n^2/(n-1) * Danorm*Danorm'
            # x = x + (D*a)/(sqrt(a'*D*a) * (n+1))
            # D = n^2/(n^2-1) * (D - 2 * D*a*a'*D/(a'*D*a * (n+1)))
        catch e
            display(D)
            display(diag(D))
            display(eigen(D).values)
            rethrow(e)
        end
        if !issymmetric(D)
            D = 0.5*(D + D')
        end
    end
end
