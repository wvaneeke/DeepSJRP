# simulation.jl - Monte Carlo simulation for policy performance
#   evaluation with unified dispatch across all policy types.

using Random
using Statistics
using Distributions
using ProgressMeter
using Base.Threads


# =====================================================================
# In-place policy dispatch: compute_order_quantity!
# =====================================================================

function compute_order_quantity!(
    order_quantity::Vector{Int},
    policy::IndividualSSPolicy,
    inventory_level::Vector{Int},
    period::Int
)
    fill!(order_quantity, 0)
    for i in eachindex(order_quantity)
        if inventory_level[i] <= policy.s_levels[i]
            order_quantity[i] =
                policy.S_levels[i] - inventory_level[i]
        end
    end
    return nothing
end

function compute_order_quantity!(
    order_quantity::Vector{Int},
    policy::PeriodicRSPolicy,
    inventory_level::Vector{Int},
    period::Int
)
    fill!(order_quantity, 0)
    if period % policy.R == 0
        for i in eachindex(order_quantity)
            order_quantity[i] =
                policy.S_levels[i] - inventory_level[i]
        end
    end
    return nothing
end

function compute_order_quantity!(
    order_quantity::Vector{Int},
    policy::AggregateDemandQSPolicy,
    inventory_level::Vector{Int},
    period::Int
)
    fill!(order_quantity, 0)
    if policy.cumulative_demand >= policy.Q || period == 0
        for i in eachindex(order_quantity)
            order_quantity[i] =
                policy.S_levels[i] - inventory_level[i]
        end
    end

    if any(q -> q > 0, order_quantity)
        policy.cumulative_demand = 0
    end
    return nothing
end

function compute_order_quantity!(
    order_quantity::Vector{Int},
    policy::HybridRQSPolicy,
    inventory_level::Vector{Int},
    period::Int
)
    fill!(order_quantity, 0)
    if policy.cumulative_demand >= policy.Q ||
       policy.periods_since_order >= policy.R ||
       period == 0
        for i in eachindex(order_quantity)
            order_quantity[i] =
                policy.S_levels[i] - inventory_level[i]
        end
    end

    if any(q -> q > 0, order_quantity)
        policy.cumulative_demand = 0
        policy.periods_since_order = 0
    end
    return nothing
end

function compute_order_quantity!(
    order_quantity::Vector{Int},
    policy::CanOrderPolicy,
    inventory_level::Vector{Int},
    period::Int
)
    fill!(order_quantity, 0)
    should_order = false

    for i in eachindex(order_quantity)
        if inventory_level[i] <= policy.s_levels[i]
            should_order = true
            break
        end
    end

    if should_order
        ω = policy.omega
        for i in eachindex(order_quantity)
            can_order_level =
                ω * policy.s_levels[i] +
                (1 - ω) * policy.S_levels[i]
            if inventory_level[i] <= can_order_level
                order_quantity[i] =
                    policy.S_levels[i] - inventory_level[i]
            end
        end
    end
    return nothing
end

function _grid_compute_order_quantity!(
    order_quantity::Vector{Int},
    policy_dict::Dict{Tuple{Int16, Int16}, Tuple{Int16, Int16}},
    min_inv::Int16,
    max_inv::Int16,
    inventory_level::Vector{Int}
)
    fill!(order_quantity, 0)
    x1 = inventory_level[1]
    x2 = inventory_level[2]

    if x1 >= 0 && x2 >= 0
        return nothing
    end

    s1 = clamp(Int16(x1), min_inv, max_inv)
    s2 = clamp(Int16(x2), min_inv, max_inv)
    action = get(policy_dict, (s1, s2), (Int16(0), Int16(0)))
    a1 = Int(action[1])
    a2 = Int(action[2])

    if (x1 < min_inv) || (x2 < min_inv)
        target1 = Int(s1) + a1
        target2 = Int(s2) + a2
        order_quantity[1] = max(target1 - x1, 0)
        order_quantity[2] = max(target2 - x2, 0)
        return nothing
    end

    order_quantity[1] = max(a1, 0)
    order_quantity[2] = max(a2, 0)
    return nothing
end

function compute_order_quantity!(
    order_quantity::Vector{Int},
    policy::MDPPolicy,
    inventory_level::Vector{Int},
    period::Int
)
    return _grid_compute_order_quantity!(
        order_quantity, policy.policy,
        policy.min_inv, policy.max_inv,
        inventory_level
    )
end

function compute_order_quantity!(
    order_quantity::Vector{Int},
    policy::NNGridPolicy,
    inventory_level::Vector{Int},
    period::Int
)
    return _grid_compute_order_quantity!(
        order_quantity, policy.policy,
        policy.min_inv, policy.max_inv,
        inventory_level
    )
end

function compute_order_quantity!(
    order_quantity::Vector{Int},
    policy::NeuralNetworkPolicy,
    inventory_level::Vector{Int},
    period::Int
)
    fill!(order_quantity, 0)

    no_action_value = V_operator(
        policy.nns, Float64.(inventory_level),
        policy.impulse_prob
    )

    rules_active = (
        length(inventory_level) == 2 &&
        policy.min_inv != typemin(Int)
    )

    if rules_active
        x1 = inventory_level[1]
        x2 = inventory_level[2]
        if x1 >= 0 && x2 >= 0
            return nothing
        end
        if x1 < policy.min_inv || x2 < policy.min_inv
            order_quantity[1] = max(policy.z_star[1] - x1, 0)
            order_quantity[2] = max(policy.z_star[2] - x2, 0)
            return nothing
        end
    end

    if no_action_value < policy.epsilon
        for i in eachindex(order_quantity)
            order_quantity[i] =
                max(policy.z_star[i] - inventory_level[i], 0)
        end
    end
    return nothing
end


# =====================================================================
# State management for stateful policies
# =====================================================================

reset_state!(::AbstractStatelessPolicy) = nothing

function reset_state!(policy::AggregateDemandQSPolicy)
    policy.cumulative_demand = 0
    return nothing
end

function reset_state!(policy::HybridRQSPolicy)
    policy.cumulative_demand = 0
    policy.periods_since_order = 0
    return nothing
end

function update_state_after_demand!(
    ::AbstractStatelessPolicy,
    ::Vector{Int}
)
    return nothing
end

function update_state_after_demand!(
    policy::AggregateDemandQSPolicy,
    demand::Vector{Int}
)
    policy.cumulative_demand += sum(demand)
    return nothing
end

function update_state_after_demand!(
    policy::HybridRQSPolicy,
    demand::Vector{Int}
)
    policy.cumulative_demand += sum(demand)
    policy.periods_since_order += 1
    return nothing
end


# =====================================================================
# Unified Monte Carlo simulation
# =====================================================================

"""
    MCS_DiscreteInventory(inventory_prob, x_initial, policy; kwargs...)

Monte Carlo simulation for evaluating policy performance in
the discrete-time inventory problem. Supports all policy types
via multiple dispatch.

Arguments:
- inventory_prob: InventoryProblem
- x_initial: Initial inventory state
- policy: Any AbstractPolicy subtype

Keyword Arguments:
- num_iterations: Number of sample paths
- time_horizon: Number of periods per path

Returns:
- mean_cost: Average total discounted cost
- std_error: Standard error of the estimate
- results_dict: Detailed results dictionary
"""
function MCS_DiscreteInventory(
    inventory_prob::InventoryProblem,
    x_initial::Vector{Int},
    policy::AbstractPolicy;
    num_iterations::Int=100_000,
    time_horizon::Int=10_000
)
    γ = inventory_prob.discount_factor
    p = inventory_prob.backlog_costs
    h = inventory_prob.holding_costs
    c0 = inventory_prob.fixed_cost
    ci = inventory_prob.variable_costs
    nb_demand = inventory_prob.nb_demand

    total_costs = Vector{Float64}(undef, num_iterations)
    ordering_costs_array = Vector{Float64}(
        undef, num_iterations
    )
    fixed_costs_array = Vector{Float64}(
        undef, num_iterations
    )
    inventory_costs_array = Vector{Float64}(
        undef, num_iterations
    )
    backlogging_costs_array = Vector{Float64}(
        undef, num_iterations
    )
    holding_costs_array = Vector{Float64}(
        undef, num_iterations
    )
    ordering_frequency_array = Vector{Float64}(
        undef, num_iterations
    )

    d = length(x_initial)

    # @showprogress
    wall_time = @elapsed @threads for iter in 1:num_iterations
        # Thread-local copies (Distribution objects may
        # cache sampler state that is not thread-safe)
        local_policy = deepcopy(policy)
        reset_state!(local_policy)
        local_nb_demand = deepcopy(nb_demand)

        discounted_costs = 0.0
        ordering_costs = 0.0
        total_fixed_costs = 0.0
        inventory_costs = 0.0
        backlogging_costs = 0.0
        holding_costs = 0.0
        order_count = 0

        inventory_level = copy(x_initial)
        order_quantity = zeros(Int, d)
        D_n = Vector{Int}(undef, d)

        for n in 0:time_horizon
            compute_order_quantity!(
                order_quantity, local_policy,
                inventory_level, n
            )

            if any(q -> q > 0, order_quantity)
                order_count += 1
                var_cost = 0.0
                for i in 1:d
                    inventory_level[i] += order_quantity[i]
                    var_cost += order_quantity[i] * ci[i]
                end
                var_cost *= γ^n
                fix_cost = γ^n * c0
                ordering_costs += var_cost
                total_fixed_costs += fix_cost
                discounted_costs += var_cost + fix_cost
            end

            @. D_n = rand(local_nb_demand)
            @. inventory_level -= D_n
            update_state_after_demand!(
                local_policy, D_n
            )

            discount = γ^n
            backlog_cost = 0.0
            hold_cost = 0.0
            for i in 1:d
                xi = inventory_level[i]
                if xi < 0
                    backlog_cost += p[i] * (-xi)
                else
                    hold_cost += h[i] * xi
                end
            end
            backlog_cost *= discount
            hold_cost *= discount
            backlogging_costs += backlog_cost
            holding_costs += hold_cost
            inventory_costs += backlog_cost + hold_cost
            discounted_costs += backlog_cost + hold_cost
        end

        total_costs[iter] = discounted_costs
        ordering_costs_array[iter] = ordering_costs
        fixed_costs_array[iter] = total_fixed_costs
        inventory_costs_array[iter] = inventory_costs
        backlogging_costs_array[iter] = backlogging_costs
        holding_costs_array[iter] = holding_costs
        ordering_frequency_array[iter] =
            order_count / (time_horizon + 1)
    end

    mean_cost = mean(total_costs)
    std_error = std(total_costs) / sqrt(num_iterations)
    mean_freq = mean(ordering_frequency_array)

    results_dict = Dict(
        :mean_cost => mean_cost,
        :std_error => std_error,
        :std_cost => std(total_costs),
        :mean_inventory_costs => mean(inventory_costs_array),
        :mean_variable_costs => mean(ordering_costs_array),
        :mean_fixed_costs => mean(fixed_costs_array),
        :mean_backlogging_costs => mean(
            backlogging_costs_array
        ),
        :mean_holding_costs => mean(holding_costs_array),
        :ordering_frequency => mean_freq,
        :wall_time => wall_time,
        :epsilon => policy isa NeuralNetworkPolicy ?
            policy.epsilon : NaN,
        :nn_S => policy isa NeuralNetworkPolicy ?
            policy.z_star : Int[]
    )

    return mean_cost, std_error, results_dict
end


# =====================================================================
# Epsilon evaluation (for NN policy during training pipeline)
# =====================================================================

"""
    evaluate_all_epsilons(
        inventory_prob, nns, nn_S, impulse_prob,
        epsilon_factors; kwargs...
    )

Evaluate NN policy performance across all epsilon values.
"""
function evaluate_all_epsilons(
    inventory_prob::InventoryProblem,
    nns::NeuralNetworks,
    nn_S::Vector{Int},
    impulse_prob::ImpulseControlProblem,
    epsilon_factors::Vector{Float64};
    num_iterations::Int=8,
    time_horizon::Int=5000,
    value_grid::Union{Nothing, Dict{Tuple{Int16, Int16}, Float64}}=nothing,
    nn_grid_min_inv::Int=-25,
    nn_grid_max_inv::Int=75
)
    x_initial = zeros(Int, inventory_prob.dim)

    best_epsilon = epsilon_factors[1]
    best_cost = Inf
    best_std_error = 0.0

    for ε in epsilon_factors
        println("\n" * "-"^40)
        println("Evaluating ε = $(ε)")
        println("-"^40)

        policy = if isnothing(value_grid)
            NeuralNetworkPolicy(nns, nn_S, ε, impulse_prob)
        else
            bake_grid_policy(
                value_grid, nn_S, ε,
                nn_grid_min_inv, nn_grid_max_inv
            )
        end
        ε_start = time()
        mean_cost, std_error, results =
            MCS_DiscreteInventory(
                inventory_prob, x_initial, policy;
                num_iterations=num_iterations,
                time_horizon=time_horizon
            )
        ε_elapsed = time() - ε_start

        println("Mean cost: $(mean_cost) ± $(std_error)")
        println(
            "  Inventory costs: " *
            "$(results[:mean_inventory_costs])"
        )
        println(
            "  Variable costs:  " *
            "$(results[:mean_variable_costs])"
        )
        println(
            "  Fixed costs:     " *
            "$(results[:mean_fixed_costs])"
        )
        println(
            "  Simulation time: " *
            "$(round(ε_elapsed; digits=1))s"
        )

        if mean_cost < best_cost
            best_cost = mean_cost
            best_std_error = std_error
            best_epsilon = ε
        end
        flush(stdout)
    end

    println("\n" * "="^50)
    println(
        "Best epsilon: $(best_epsilon) " *
        "with cost: $(best_cost) ± $(best_std_error)"
    )
    println("="^50)
    flush(stdout)

    return best_epsilon, best_cost, best_std_error
end


"""
    run_full_evaluation(
        inventory_prob, nns, impulse_prob, opt_params;
        kwargs...
    )

Run complete policy evaluation for both optimization
methods and all epsilon values.
"""
function run_full_evaluation(
    inventory_prob::InventoryProblem,
    nns::NeuralNetworks,
    impulse_prob::ImpulseControlProblem,
    opt_params::OptimizationParameters;
    S_start::Vector{Float64}=ones(impulse_prob.dim)
)
    println("\n" * "="^60)
    println("INVENTORY PROBLEM (unscaled, weekly)")
    println("="^60)
    println("Dimension:       $(inventory_prob.dim)")
    println("Discount factor: $(inventory_prob.discount_factor)")
    println("Fixed cost:      $(inventory_prob.fixed_cost)")
    println("Variable costs:  $(inventory_prob.variable_costs)")
    println("Holding costs:   $(inventory_prob.holding_costs)")
    println("Backlog costs:   $(inventory_prob.backlog_costs)")
    println("Demand dists:    $(typeof.(inventory_prob.nb_demand))")
    println("Demand means:    $(mean.(inventory_prob.nb_demand))")
    println("="^60)
    flush(stdout)

    results = Dict{Symbol, Any}()

    z_results = compute_order_up_to_vectors(
        nns, impulse_prob, opt_params;
        S_start=S_start,
        verbose=true
    )

    is_2d = impulse_prob.dim == 2
    value_grid = nothing
    if is_2d
        println("\n" * "="^60)
        println("Building 2D value grid for NN policy:")
        println("  bounds: [$(opt_params.nn_grid_min_inv), " *
            "$(opt_params.nn_grid_max_inv)]")
        println("="^60)
        flush(stdout)
        grid_start = time()
        value_grid = compute_nn_value_grid(
            nns, impulse_prob,
            opt_params.nn_grid_min_inv,
            opt_params.nn_grid_max_inv
        )
        println(
            "Grid built in " *
            "$(round(time() - grid_start; digits=1))s " *
            "($(length(value_grid)) states)"
        )
        flush(stdout)
    end

    for method in [:solve_inf, :find_stationary]
        println("\n" * "="^60)
        println("Evaluating method: $(method)")
        println("="^60)

        nn_S = z_results[method][:z_star_int]

        best_ε, best_cost, best_std_error =
            evaluate_all_epsilons(
                inventory_prob, nns, nn_S, impulse_prob,
                opt_params.epsilon_factors;
                num_iterations=opt_params.num_simulation_runs,
                time_horizon=opt_params.simulation_horizon,
                value_grid=value_grid,
                nn_grid_min_inv=opt_params.nn_grid_min_inv,
                nn_grid_max_inv=opt_params.nn_grid_max_inv
            )

        method_result = Dict{Symbol, Any}(
            :z_star => z_results[method][:z_star],
            :z_star_int => nn_S,
            :best_epsilon => best_ε,
            :best_cost => best_cost,
            :best_std_error => best_std_error
        )
        if is_2d
            method_result[:nn_grid_policy] = bake_grid_policy(
                value_grid, nn_S, best_ε,
                opt_params.nn_grid_min_inv,
                opt_params.nn_grid_max_inv
            )
        end
        results[method] = method_result
    end

    return results
end
