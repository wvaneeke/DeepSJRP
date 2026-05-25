# benchmark_policies.jl - Benchmark policy parameter search,
#   analytical cost formulas, and helper functions.

using Distributions
using Statistics
using Random
using Base.Threads


# =====================================================================
# Expected cost helpers (used by (R,S) and (Q,S) analytical formulas)
# =====================================================================

"""
    cumulative_demand_distribution(dist, r)

Compute the r-fold convolution of a demand distribution.
Uses additive properties for Poisson and NegativeBinomial.
"""
function cumulative_demand_distribution(
    dist::Poisson,
    r::Int
)
    return Poisson(r * mean(dist))
end

function cumulative_demand_distribution(
    dist::NegativeBinomial,
    r::Int
)
    # NB(n, p) + NB(m, p) = NB(n+m, p)
    return NegativeBinomial(
        r * dist.r,
        dist.p
    )
end

"""
    expected_holding_backlog_cost(y, dist_conv, p, h; D_max)

Compute E[h * max(y - D, 0) + p * max(D - y, 0)] where
D follows the (convolved) demand distribution `dist_conv`.
"""
function expected_holding_backlog_cost(
    y::Int,
    dist_conv,
    penalty::Float64,
    holding::Float64;
    D_max::Int=1_000
)
    cost = 0.0
    for d in 0:D_max
        prob_d = pdf(dist_conv, d)
        prob_d > 1e-9 || continue
        inv_level = y - d
        cost += max(holding * inv_level, penalty * (-inv_level)) *
            prob_d
    end
    return cost
end

"""
    expected_cost_over_cycle(y, R, nb_dist, p, h)

Compute the expected holding/backlog cost for order-up-to
level y over R periods of demand for a single item.
Uses the r-fold convolution of the base distribution.
"""
function expected_cost_over_cycle(
    y::Int,
    R::Int,
    nb_dist,
    penalty::Float64,
    holding::Float64
)
    dist_conv = nb_dist
    if R > 1
        for r in 1:(R - 1)
            dist_conv = convolve(dist_conv, nb_dist)
        end
    end
    return expected_holding_backlog_cost(
        y, dist_conv, penalty, holding
    )
end


# =====================================================================
# (R,S) Policy: optimal S search and analytical cost
# =====================================================================

"""
    objective_single_item_RS(y, i, R, inventory_prob)

Compute the single-item cost for order-up-to level y
over a cycle of R periods.
"""
function objective_single_item_RS(
    y::Int,
    i::Int,
    R::Int,
    inventory_prob::InventoryProblem
)
    γ = inventory_prob.discount_factor
    p = inventory_prob.backlog_costs
    h = inventory_prob.holding_costs
    ci = inventory_prob.variable_costs
    nb_demand = inventory_prob.nb_demand

    item_cost = (1 - γ^R) * ci[i] * y
    for r in 0:(R - 1)
        item_cost += γ^r * expected_cost_over_cycle(
            y, r + 1, nb_demand[i], p[i], h[i]
        )
    end
    return item_cost
end

"""
    find_optimal_y_RS(R, inventory_prob)

Find the optimal order-up-to vector S for a given review
interval R by univariate search per item.
"""
function find_optimal_y_RS(
    R::Int,
    inventory_prob::InventoryProblem
)
    d = inventory_prob.dim
    optimal_y = Vector{Int}(undef, d)
    total_cost = 0.0

    for i in 1:d
        y = 0
        current_cost = objective_single_item_RS(
            y, i, R, inventory_prob
        )
        while true
            next_cost = objective_single_item_RS(
                y + 1, i, R, inventory_prob
            )
            if next_cost > current_cost
                break
            end
            y += 1
            current_cost = next_cost
        end
        optimal_y[i] = y
        total_cost += current_cost
    end

    return optimal_y, total_cost
end

"""
    analytical_cost_RS(R, optimal_y, inventory_prob)

Analytical formula for total expected discounted cost
under the (R,S) periodic review policy.
"""
function analytical_cost_RS(
    R::Int,
    optimal_y::Vector{Int},
    inventory_prob::InventoryProblem
)
    γ = inventory_prob.discount_factor
    c0 = inventory_prob.fixed_cost
    ci = inventory_prob.variable_costs
    p = inventory_prob.backlog_costs
    h = inventory_prob.holding_costs
    nb_demand = inventory_prob.nb_demand
    d = inventory_prob.dim

    total_cost = c0 / (1 - γ^R)

    for i in 1:d
        Si = optimal_y[i]
        μi = mean(nb_demand[i])

        initial_var_cost = ci[i] * Si

        cycle_hb_cost = 0.0
        for r in 0:(R - 1)
            cycle_hb_cost += γ^r * expected_cost_over_cycle(
                Si, r + 1, nb_demand[i], p[i], h[i]
            )
        end

        ongoing_var_cost = γ^R * ci[i] * R * μi

        total_cost += initial_var_cost +
            (cycle_hb_cost + ongoing_var_cost) / (1 - γ^R)
    end

    return total_cost
end


# =====================================================================
# (Q,S) Policy: RQ demand cycle sampling, S search, analytical cost
# =====================================================================

"""
    PrecomputedCycles

Precomputed cycle simulation data, reusable across all y-searches
and analytical cost evaluations for a given (Q) or (R,Q) config.

Fields:
- `N`: number of simulated cycles
- `d`: problem dimension (number of items)
- `stopping_times`: vector of length N with cycle stopping times
- `item_cumulative`:  N × d matrix; `item_cumulative[n, i]` =
      total demand for item i during cycle n
- `demand_histories`: vector of length N; each element is a
      `T_n × d` matrix where row t gives cumulative demand per
      item through period t of that cycle
"""
struct PrecomputedCycles
    N::Int
    d::Int
    stopping_times::Vector{Int}
    item_cumulative::Matrix{Int}
    demand_histories::Vector{Matrix{Int}}
end

"""
    precompute_QS_cycles(Q, nb_demand; N, seed)

Simulate N (Q,S) cycles once and store all paths for reuse.
"""
function precompute_QS_cycles(
    Q::Int,
    nb_demand::Vector;
    N::Int = 1_000,
    seed::Int = 777
)
    Random.seed!(seed)
    d = length(nb_demand)

    stopping_times = Vector{Int}(undef, N)
    item_cumulative = Matrix{Int}(undef, N, d)
    demand_histories = Vector{Matrix{Int}}(undef, N)

    for n in 1:N
        agg = 0
        t = 0
        item_cum = zeros(Int, d)
        history_rows = Vector{Vector{Int}}()

        while agg < Q
            t += 1
            for i in 1:d
                ξ = rand(nb_demand[i])
                item_cum[i] += ξ
                agg += ξ
            end
            push!(history_rows, copy(item_cum))
        end

        stopping_times[n] = t
        item_cumulative[n, :] .= item_cum
        # Convert vector-of-vectors to T×d matrix
        demand_histories[n] = reduce(
            vcat, transpose.(history_rows)
        )
    end

    return PrecomputedCycles(
        N, d, stopping_times,
        item_cumulative, demand_histories
    )
end

"""
    precompute_RQS_cycles(R, Q, nb_demand; N, seed)

Simulate N (R,Q,S) cycles once and store all paths for reuse.
Cycle stops at min(R_Q, R).
"""
function precompute_RQS_cycles(
    R::Int,
    Q::Int,
    nb_demand::Vector;
    N::Int = 1_000,
    seed::Int = 777
)
    Random.seed!(seed)
    d = length(nb_demand)

    stopping_times = Vector{Int}(undef, N)
    item_cumulative = Matrix{Int}(undef, N, d)
    demand_histories = Vector{Matrix{Int}}(undef, N)

    for n in 1:N
        agg = 0
        t = 0
        item_cum = zeros(Int, d)
        history_rows = Vector{Vector{Int}}()

        while agg < Q && t < R
            t += 1
            for i in 1:d
                ξ = rand(nb_demand[i])
                item_cum[i] += ξ
                agg += ξ
            end
            push!(history_rows, copy(item_cum))
        end

        stopping_times[n] = t
        item_cumulative[n, :] .= item_cum
        demand_histories[n] = reduce(
            vcat, transpose.(history_rows)
        )
    end

    return PrecomputedCycles(
        N, d, stopping_times,
        item_cumulative, demand_histories
    )
end

"""
    evaluate_y_for_item(y, i, inventory_prob, cycles)

Evaluate the expected cost for item i with order-up-to level y
using precomputed cycle data.
"""
function evaluate_y_for_item(
    y::Int,
    i::Int,
    inventory_prob::InventoryProblem,
    cycles::PrecomputedCycles
)
    γ = inventory_prob.discount_factor
    p = inventory_prob.backlog_costs[i]
    h = inventory_prob.holding_costs[i]
    ci = inventory_prob.variable_costs[i]
    N = cycles.N

    total_cost = 0.0
    total_γ_R = 0.0

    for n in 1:N
        R_n = cycles.stopping_times[n]
        hist = cycles.demand_histories[n]

        # Holding/backlog costs during cycle
        cycle_cost = 0.0
        for t in 1:R_n
            inv_level = y - hist[t, i]
            if inv_level >= 0
                cycle_cost += γ^(t - 1) * h * inv_level
            else
                cycle_cost += γ^(t - 1) * p * (-inv_level)
            end
        end

        total_cost += cycle_cost
        total_γ_R += γ^R_n
    end

    E_γ_R = total_γ_R / N
    E_cycle_cost = total_cost / N

    return (1 - E_γ_R) * ci * y + E_cycle_cost
end

"""
    find_optimal_y_joint(inventory_prob, cycles)

Find optimal order-up-to vector S using precomputed cycle data.
Works for both (Q,S) and (R,Q,S) policies.
"""
function find_optimal_y_joint(
    inventory_prob::InventoryProblem,
    cycles::PrecomputedCycles
)
    d = inventory_prob.dim
    optimal_y = Vector{Int}(undef, d)
    total_cost = 0.0

    for i in 1:d
        y = 0
        current_cost = evaluate_y_for_item(
            y, i, inventory_prob, cycles
        )

        while true
            next_cost = evaluate_y_for_item(
                y + 1, i, inventory_prob, cycles
            )
            if next_cost > current_cost
                break
            end
            y += 1
            current_cost = next_cost
        end

        optimal_y[i] = y
        total_cost += current_cost
    end

    return optimal_y, total_cost
end

"""
    analytical_cost_joint(
        optimal_y, inventory_prob, cycles
    )

Compute total expected discounted cost using precomputed
cycle data. Works for both (Q,S) and (R,Q,S) policies.
"""
function analytical_cost_joint(
    optimal_y::Vector{Int},
    inventory_prob::InventoryProblem,
    cycles::PrecomputedCycles
)
    γ = inventory_prob.discount_factor
    c0 = inventory_prob.fixed_cost
    ci = inventory_prob.variable_costs
    p = inventory_prob.backlog_costs
    h = inventory_prob.holding_costs
    d = inventory_prob.dim
    N = cycles.N

    total_cycle_cost = 0.0
    total_γ_R = 0.0

    for n in 1:N
        R_n = cycles.stopping_times[n]
        hist = cycles.demand_histories[n]

        # Fixed cost
        cost = c0

        # Holding/backlog costs during cycle
        for t in 1:R_n
            for i in 1:d
                inv_level = optimal_y[i] - hist[t, i]
                if inv_level >= 0
                    cost += γ^(t - 1) * h[i] * inv_level
                else
                    cost += γ^(t - 1) * p[i] * (-inv_level)
                end
            end
        end

        # Variable replenishment cost at end of cycle
        for i in 1:d
            cost += γ^R_n * ci[i] *
                cycles.item_cumulative[n, i]
        end

        total_cycle_cost += cost
        total_γ_R += γ^R_n
    end

    E_γ_R = total_γ_R / N
    E_cycle_cost = total_cycle_cost / N

    # Initial variable cost + renewal sum of cycle costs
    initial_var_cost = sum(ci .* optimal_y)

    return initial_var_cost + E_cycle_cost / (1 - E_γ_R)
end

# =====================================================================
# (R,Q,S) Policy: uses precompute_RQS_cycles + unified functions
#   (find_optimal_y_joint, analytical_cost_joint)
# =====================================================================

# =====================================================================
# Parameter search functions
# =====================================================================

"""
    search_individual_sS(inventory_prob; kwargs...)

Search over α (fixed cost fraction) to find the best
individual (s,S) policy.
"""
function search_individual_sS(
    inventory_prob::InventoryProblem;
    alpha_range=0.05:0.05:1.0,
    num_iterations::Int=10_000,
    time_horizon::Int=10_000,
    seed::Int=777
)
    lowest_cost = Inf
    best_result = nothing

    for α in alpha_range
        Random.seed!(seed)
        c0_distributed = (inventory_prob.fixed_cost * α) *
            ones(inventory_prob.dim)

        sS_policies = compute_sS_policies(
            inventory_prob, c0_distributed
        )

        policy = IndividualSSPolicy(
            sS_policies[:, 1],
            sS_policies[:, 2],
            α
        )

        println("\n(s,S) search: α = $(α)")
        x_initial = zeros(Int, inventory_prob.dim)
        t_sim = time()
        mean_cost, std_error, results_dict =
            MCS_DiscreteInventory(
                inventory_prob, x_initial, policy;
                num_iterations=num_iterations,
                time_horizon=time_horizon
            )
        t_sim = time() - t_sim
        @printf(
            "  Cost: %.2f ± %.2f (%.1fs)\n",
            mean_cost, std_error, t_sim
        )
        flush(stdout)

        if mean_cost < lowest_cost
            lowest_cost = mean_cost
            best_result = BenchmarkSearchResult(
                policy,
                mean_cost,
                std_error,
                results_dict[:ordering_frequency],
                nothing,
                Dict{Symbol, Any}(:best_alpha => α)
            )
        end
    end

    println("\n=== Individual (s,S) Results ===")
    println("Best α: $(best_result.search_params[:best_alpha])")
    println("Best cost: $(best_result.cost)")
    flush(stdout)
    return best_result
end


"""
    search_periodic_RS(inventory_prob; kwargs...)

Search over review interval R to find the best (R,S) policy.
"""
function search_periodic_RS(
    inventory_prob::InventoryProblem;
    R_range=1:100,
    num_iterations::Int=10_000,
    time_horizon::Int=10_000,
    seed::Int=777
)
    num_R = length(R_range)
    costs = Vector{Float64}(undef, num_R)
    std_errors = Vector{Float64}(undef, num_R)
    analytical_costs = Vector{Float64}(undef, num_R)
    policies = Vector{PeriodicRSPolicy}(undef, num_R)
    freqs = Vector{Float64}(undef, num_R)

    for idx in 1:num_R
        Random.seed!(seed)
        R = R_range[idx]

        t_anal = time()
        optimal_y, _ = find_optimal_y_RS(R, inventory_prob)
        a_cost = analytical_cost_RS(
            R, optimal_y, inventory_prob
        )
        t_anal = time() - t_anal

        policy = PeriodicRSPolicy(optimal_y, R)
        x_initial = zeros(Int, inventory_prob.dim)

        t_sim = time()
        mean_cost, std_error, results_dict =
            MCS_DiscreteInventory(
                inventory_prob, x_initial, policy;
                num_iterations=num_iterations,
                time_horizon=time_horizon
            )
        t_sim = time() - t_sim

        costs[idx] = mean_cost
        std_errors[idx] = std_error
        analytical_costs[idx] = a_cost
        policies[idx] = policy
        freqs[idx] = results_dict[:ordering_frequency]

        @printf(
            "  (R,S) R=%d: sim=%.2f anal=%.2f (anal %.1fs, sim %.1fs)\n",
            R, mean_cost, a_cost, t_anal, t_sim
        )
        flush(stdout)
    end

    best_idx = argmin(costs)
    best_policy = policies[best_idx]

    println("\n=== (R,S) Results ===")
    println("Best R: $(best_policy.R)")
    println("Best cost: $(costs[best_idx])")
    flush(stdout)

    return BenchmarkSearchResult(
        best_policy,
        costs[best_idx],
        std_errors[best_idx],
        freqs[best_idx],
        analytical_costs[best_idx],
        Dict{Symbol, Any}(:best_R => best_policy.R)
    )
end


"""
    search_aggregate_QS(inventory_prob; kwargs...)

Search over aggregate demand threshold Q to find the best
(Q,S) policy. Q range is derived from best_R (from the
(R,S) search) and weekly demand rates.
"""
function search_aggregate_QS(
    inventory_prob::InventoryProblem;
    best_R::Int=10,
    num_rq_samples::Int=100_000,
    num_iterations::Int=10_000,
    time_horizon::Int=10_000,
    seed::Int=777
)
    weekly_demand = sum(mean.(inventory_prob.nb_demand))
    Q_start = max(
        1,
        floor(Int, weekly_demand * (best_R - 5))
    )
    Q_end = floor(Int, weekly_demand * (best_R + 5))
    Q_values = Q_start:1:Q_end
    num_Q = length(Q_values)

    println(
        "QS search: Q ∈ [$(Q_start), $(Q_end)] " *
        "($(num_Q) values)"
    )
    flush(stdout)

    # Phase 1: Analytical search over Q values (parallel)
    analytical_costs = Vector{Float64}(undef, num_Q)
    optimal_ys = Vector{Vector{Int}}(undef, num_Q)

    t_anal_total = time()
    @threads for idx in 1:num_Q
        Q = Q_values[idx]

        t_anal = time()
        cycles = precompute_QS_cycles(
            Q, inventory_prob.nb_demand;
            N=num_rq_samples, seed=seed
        )

        optimal_y, _ = find_optimal_y_joint(
            inventory_prob, cycles
        )

        a_cost = analytical_cost_joint(
            optimal_y, inventory_prob, cycles
        )
        t_anal = time() - t_anal

        analytical_costs[idx] = a_cost
        optimal_ys[idx] = optimal_y

        @printf(
            "  (Q,S) Q=%d: anal=%.2f (%.1fs)\n",
            Q, a_cost, t_anal
        )
        flush(stdout)
    end
    t_anal_total = time() - t_anal_total

    # Phase 2: Find best Q by analytical cost
    best_idx = argmin(analytical_costs)
    best_Q = Q_values[best_idx]
    best_S = optimal_ys[best_idx]
    best_analytical_cost = analytical_costs[best_idx]

    @printf(
        "\nBest analytical: Q=%d, cost=%.2f\n",
        best_Q, best_analytical_cost
    )
    @printf(
        "Analytical search time: %.1fs\n", t_anal_total
    )
    flush(stdout)

    # Phase 3: Final simulation with best parameters
    println("Running final simulation...")
    flush(stdout)
    Random.seed!(seed)
    best_policy = AggregateDemandQSPolicy(best_S, best_Q)
    x_initial = zeros(Int, inventory_prob.dim)

    t_sim = time()
    mean_cost, std_error, results_dict =
        MCS_DiscreteInventory(
            inventory_prob, x_initial, best_policy;
            num_iterations=num_iterations,
            time_horizon=time_horizon
        )
    t_sim = time() - t_sim

    freq = results_dict[:ordering_frequency]
    avg_weeks = freq > 0 ? 1.0 / freq : Inf

    println("\n=== (Q,S) Results ===")
    println("Best Q: $(best_Q)")
    @printf(
        "Best cost: %.2f ± %.2f\n",
        mean_cost, std_error
    )
    println("Avg weeks between orders: $(avg_weeks)")
    @printf("Final simulation time: %.1fs\n", t_sim)
    flush(stdout)

    return BenchmarkSearchResult(
        best_policy,
        mean_cost,
        std_error,
        freq,
        best_analytical_cost,
        Dict{Symbol, Any}(
            :best_Q => best_Q,
            :average_weeks_between_orders => avg_weeks
        )
    )
end

# =====================================================================
# (R,Q,S) Parameter search
# =====================================================================

"""
    search_hybrid_RQS(inventory_prob; R_range, Q_range, ...)

Search over (R, Q) pairs to find the best (R,Q,S) policy.
Uses analytical cost to select the best (R, Q), then runs
a single final simulation.
"""
function search_hybrid_RQS(
    inventory_prob::InventoryProblem;
    R_range::AbstractVector{Int},
    Q_range::AbstractVector{Int},
    num_rq_samples::Int = 100_000,
    num_iterations::Int = 10_000,
    time_horizon::Int = 10_000,
    seed::Int = 777
)
    # Build linearized list of (R, Q) pairs
    rq_pairs = [(R, Q) for R in R_range for Q in Q_range]
    num_pairs = length(rq_pairs)

    println(
        "RQS search: R ∈ [$(first(R_range)), " *
        "$(last(R_range))], " *
        "Q ∈ [$(first(Q_range)), " *
        "$(last(Q_range))], " *
        "$(num_pairs) total configs"
    )
    flush(stdout)

    # Phase 1: Analytical search (parallel)
    a_costs = Vector{Float64}(undef, num_pairs)
    opt_ys = Vector{Vector{Int}}(undef, num_pairs)

    t_anal_total = time()
    @threads for idx in 1:num_pairs
        R, Q = rq_pairs[idx]

        t_anal = time()
        cycles = precompute_RQS_cycles(
            R, Q, inventory_prob.nb_demand;
            N=num_rq_samples, seed=seed
        )

        optimal_y, _ = find_optimal_y_joint(
            inventory_prob, cycles
        )

        a_cost = analytical_cost_joint(
            optimal_y, inventory_prob, cycles
        )
        t_anal = time() - t_anal

        a_costs[idx] = a_cost
        opt_ys[idx] = optimal_y

        @printf(
            "  (R,Q,S) R=%d, Q=%d: anal=%.2f (%.1fs)\n",
            R, Q, a_cost, t_anal
        )
        flush(stdout)
    end
    t_anal_total = time() - t_anal_total

    # Phase 2: Find best (R, Q) by analytical cost
    best_idx = argmin(a_costs)
    best_R_found, best_Q_found = rq_pairs[best_idx]
    best_S = opt_ys[best_idx]
    best_analytical_cost = a_costs[best_idx]

    @printf(
        "\nBest analytical: R=%d, Q=%d, cost=%.2f\n",
        best_R_found, best_Q_found,
        best_analytical_cost
    )
    @printf(
        "Analytical search time: %.1fs\n",
        t_anal_total
    )
    flush(stdout)

    # Phase 3: Final simulation with best parameters
    println("Running final simulation...")
    flush(stdout)
    Random.seed!(seed)
    best_policy = HybridRQSPolicy(
        best_S, best_R_found, best_Q_found
    )
    x_initial = zeros(Int, inventory_prob.dim)

    t_sim = time()
    mean_cost, std_error, results_dict =
        MCS_DiscreteInventory(
            inventory_prob, x_initial, best_policy;
            num_iterations=num_iterations,
            time_horizon=time_horizon
        )
    t_sim = time() - t_sim

    freq = results_dict[:ordering_frequency]
    avg_weeks = freq > 0 ? 1.0 / freq : Inf

    println("\n=== (R,Q,S) Results ===")
    println("Best R: $(best_R_found)")
    println("Best Q: $(best_Q_found)")
    @printf(
        "Best cost: %.2f ± %.2f\n",
        mean_cost, std_error
    )
    println("Avg weeks between orders: $(avg_weeks)")
    @printf("Final simulation time: %.1fs\n", t_sim)
    flush(stdout)

    return BenchmarkSearchResult(
        best_policy,
        mean_cost,
        std_error,
        freq,
        best_analytical_cost,
        Dict{Symbol, Any}(
            :best_R => best_R_found,
            :best_Q => best_Q_found,
            :average_weeks_between_orders => avg_weeks
        )
    )
end


"""
    search_can_order(inventory_prob; kwargs...)

Grid search over (α, ω) to find the best can-order policy.
Caches ZF results across ω values for the same α.
"""
function search_can_order(
    inventory_prob::InventoryProblem;
    alpha_range=0.05:0.05:1.0,
    omega_range=0.0:0.1:1.0,
    num_iterations::Int=10_000,
    time_horizon::Int=10_000,
    seed::Int=777
)
    lowest_cost = Inf
    best_result = nothing

    # Cache ZF results per alpha (same across omega)
    for α in alpha_range
        c0_distributed = (inventory_prob.fixed_cost * α) *
            ones(inventory_prob.dim)
        sS_policies = compute_sS_policies(
            inventory_prob, c0_distributed
        )
        s_levels = sS_policies[:, 1]
        S_levels = sS_policies[:, 2]

        for ω in omega_range
            Random.seed!(seed)

            policy = CanOrderPolicy(
                s_levels, S_levels, ω, α
            )
            x_initial = zeros(Int, inventory_prob.dim)

            t_sim = time()
            mean_cost, std_error, results_dict =
                MCS_DiscreteInventory(
                    inventory_prob, x_initial, policy;
                    num_iterations=num_iterations,
                    time_horizon=time_horizon
                )
            t_sim = time() - t_sim

            @printf(
                "  Can-order (α=%.2f, ω=%.1f): %.2f (%.1fs)\n",
                α, ω, mean_cost, t_sim
            )
            flush(stdout)

            if mean_cost < lowest_cost
                lowest_cost = mean_cost
                best_result = BenchmarkSearchResult(
                    policy,
                    mean_cost,
                    std_error,
                    results_dict[:ordering_frequency],
                    nothing,
                    Dict{Symbol, Any}(
                        :best_alpha => α,
                        :best_omega => ω
                    )
                )
            end
        end
    end

    avg_weeks = if best_result.ordering_frequency > 0
        1.0 / best_result.ordering_frequency
    else
        Inf
    end

    println("\n=== Can-Order Results ===")
    println(
        "Best α: " *
        "$(best_result.search_params[:best_alpha])"
    )
    println(
        "Best ω: " *
        "$(best_result.search_params[:best_omega])"
    )
    println("Best cost: $(best_result.cost)")
    println("Avg weeks between orders: $(avg_weeks)")
    flush(stdout)

    return best_result
end


# =====================================================================
# Relative difference computation (delta method)
# =====================================================================

"""
    compute_relative_difference(
        cost_nn, se_nn, cost_bench, se_bench
    )

Compute percentage difference and its standard error
using the delta method.

    Δ = (C_bench - C_NN) / C_NN × 100%

Returns `(pct_diff, pct_se)`.
"""
function compute_relative_difference(
    cost_nn::Float64,
    se_nn::Float64,
    cost_bench::Float64,
    se_bench::Float64
)
    pct_diff = (cost_bench - cost_nn) / cost_nn * 100.0
    pct_se = 100.0 * sqrt(
        (se_bench / cost_nn)^2 +
        (cost_bench * se_nn / cost_nn^2)^2
    )
    return pct_diff, pct_se
end




