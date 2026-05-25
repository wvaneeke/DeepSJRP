# optimization.jl - Optimization for finding the order-up-to vector (Algorithm 3)

using LBFGSB
using LinearAlgebra
using ForwardDiff
using Zygote


"""
    M_operator(nns, x0, impulse_prob; kwargs...)

Compute the order-up-to vector z* using either direct minimization or
root-finding via the stationarity condition.

Implements the preprocessing step of Algorithm 3.

Arguments:
- nns: Trained neural networks
- x0: Current inventory state
- impulse_prob: ImpulseControlProblem

Keyword Arguments:
- S_start: Starting guess for optimization
- method: :solve_inf (minimize H(z) + c·z) or :find_stationary (solve ∇V(z) = -c)
- lower: Lower bounds for z*
- upper: Upper bounds for z*
- δ_init: Scaling factor for initial point
- m: Number of L-BFGS-B correction pairs (memory)
- maxiter: Maximum iterations for optimizer
- iprint: L-BFGS-B print verbosity level

Returns:
- z_star: Optimal order-up-to vector
- intervention_value: V(x₀) - V(z*) - c(z* - x₀) - c₀
"""
function M_operator(
    nns::NeuralNetworks,
    x0::AbstractVector,
    impulse_prob::ImpulseControlProblem;
    S_start::Vector{Float64}=ones(impulse_prob.dim),
    method::Symbol=:find_stationary,
    lower::Vector{Float64}=0.0 * S_start,
    upper::Vector{Float64}=2.0 * S_start,
    δ_init::Float64=1.0,
    m::Int=10,
    maxiter::Int=100,
    iprint::Int=1
)
    ci = impulse_prob.variable_costs
    c0 = impulse_prob.fixed_cost

    # Initial point
    z_init = δ_init * S_start

    if method == :solve_inf
        # Method (a): Minimize H_θ(z) + c·z over z ∈ [lower, upper]
        f_inf(z) = V_net(nns, z) + dot(z, ci)

        function g_inf!(G, z)
            G .= Zygote.gradient(f_inf, z)[1]
        end

        obj, z_star = lbfgsb(f_inf, g_inf!, z_init;
                             lb=lower, ub=upper,
                             m=m, maxiter=maxiter,
                             iprint=iprint, pgtol=1e-2)

        println("\nMethod: solve_inf")
        println("Final objective: ", obj)
        println("z*: ", z_star)

    elseif method == :find_stationary
        # Method (b): Solve the stationarity condition ∇V(z*) = -c
        # by minimizing ||G_ϑ(z) + c||²
        function f_stat(z)
            residual = Z_net(nns, z) .+ ci
            return dot(residual, residual)
        end

        function g_stat!(G, z)
            G .= Zygote.gradient(f_stat, z)[1]
        end

        obj, z_star = lbfgsb(f_stat, g_stat!, z_init;
                             lb=lower, ub=upper,
                             m=m, maxiter=maxiter,
                             iprint=iprint, pgtol=1e-5)

        println("\nMethod: find_stationary")
        println("Final residual: ", sqrt(obj))
        println("z*: ", z_star)

    else
        error("Unknown method: $(method). Use :solve_inf or :find_stationary")
    end

    # Compute intervention value
    intervention_value = V_net(nns, x0) -
        (V_net(nns, z_star) + dot(z_star .- x0, ci)) - c0
    println("Intervention value: ", intervention_value)

    return z_star, intervention_value
end


"""
    V_operator(nns, x0, impulse_prob)

Compute the no-action term from the HJB equation (12).
Used to check if intervention is warranted (condition (21)).

Computes: LV(x) + rV(x) - f(x)
where LV = -½ tr(σσᵀ Hess(V)) + μᵀ∇V

Arguments:
- nns: Trained neural networks
- x0: Current inventory state
- impulse_prob: ImpulseControlProblem

Returns:
- no_action_value: The value of the no-action term N(x)
"""
function V_operator(
    nns::NeuralNetworks,
    x0::AbstractVector,
    impulse_prob::ImpulseControlProblem
)
    d = impulse_prob.dim
    μ = impulse_prob.μ
    Σ = impulse_prob.Σ
    r = impulse_prob.interest_rate
    h = impulse_prob.holding_costs
    p = impulse_prob.penalty_costs

    # Inventory cost function
    function f_inventory(x)
        return sum(h .* max.(x, 0) .+ p .* max.(-x, 0))
    end

    # Compute diffusion term: -½ tr(Σ * Hess(V))
    if isdiag(Σ)
        # Efficient computation for diagonal covariance
        σ_diag = diag(Σ)
        eᵢ = zeros(d)
        hess_diag = Vector{Float64}(undef, d)

        for i in 1:d
            eᵢ[i] = 1.0
            # ∂²V/∂xᵢ² = ∂Gᵢ/∂xᵢ
            hess_diag[i] = ForwardDiff.derivative(
                t -> Z_net(nns, x0 .+ t .* eᵢ)[i],
                0.0
            )
            eᵢ[i] = 0.0
        end
        diffusion_term = -0.5 * dot(σ_diag, hess_diag)
    else
        # General case: compute full Hessian via Jacobian of gradient
        Hessian_mat = Zygote.jacobian(x -> Z_net(nns, x), x0)[1]
        diffusion_term = -0.5 * tr(Σ * Hessian_mat)
    end

    # Drift term: μᵀ∇V
    drift_term = dot(μ, Z_net(nns, x0))

    # Discount term: rV(x)
    discount_term = r * V_net(nns, x0)

    # Inventory cost: f(x)
    inventory_cost = f_inventory(x0)

    # No-action condition: LV + rV - f
    return diffusion_term + drift_term + discount_term - inventory_cost
end


"""
    compute_order_up_to_vectors(nns, impulse_prob, opt_params; kwargs...)

Compute order-up-to vectors using both optimization methods.

Arguments:
- nns: Trained neural networks
- impulse_prob: ImpulseControlProblem
- opt_params: OptimizationParameters

Returns:
- results: Dict with results for both methods
"""
function compute_order_up_to_vectors(
    nns::NeuralNetworks,
    impulse_prob::ImpulseControlProblem,
    opt_params::OptimizationParameters;
    S_start::Vector{Float64}=ones(impulse_prob.dim),
    verbose::Bool=true
)
    d = impulse_prob.dim
    x0 = zeros(d)

    lower = opt_params.lower_factor * S_start
    upper = opt_params.upper_factor * S_start
    δ_init = opt_params.start_factor

    results = Dict{Symbol, Any}()

    # Method 1: Direct minimization (solve_inf)
    if verbose
        println("\n" * "="^50)
        println("Computing order-up-to vector via :solve_inf")
        println("="^50)
    end

    z_inf, int_val_inf = M_operator(
        nns, x0, impulse_prob;
        S_start=S_start,
        method=:solve_inf,
        lower=lower,
        upper=upper,
        δ_init=δ_init
    )

    results[:solve_inf] = Dict(
        :z_star => z_inf,
        :z_star_int => Int.(round.(z_inf)),
        :intervention_value => int_val_inf
    )

    # Method 2: Stationarity condition (find_stationary)
    if verbose
        println("\n" * "="^50)
        println("Computing order-up-to vector via :find_stationary")
        println("="^50)
    end

    z_stat, int_val_stat = M_operator(
        nns, x0, impulse_prob;
        S_start=S_start,
        method=:find_stationary,
        lower=lower,
        upper=upper,
        δ_init=δ_init
    )

    results[:find_stationary] = Dict(
        :z_star => z_stat,
        :z_star_int => Int.(round.(z_stat)),
        :intervention_value => int_val_stat
    )

    if verbose
        println("\n" * "="^50)
        println("Summary:")
        println("  solve_inf:       z* = $(results[:solve_inf][:z_star_int])")
        println("  find_stationary: z* = $(results[:find_stationary][:z_star_int])")
        println("="^50)
    end

    return results
end
