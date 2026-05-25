# zheng_federgruen.jl - Zheng-Federgruen (1991) algorithm for
#   optimal single-item (s,S) policy under the average cost criterion.

using Distributions
using Base.Threads


"""
    expected_one_period_cost(y, params; cache)

Compute G(y), the expected one-period holding and backlog cost:
    G(y) = E[h * max(y - D, 0) + p * max(D - y, 0)]

Uses a Dict-based cache to avoid recomputation.
"""
function expected_one_period_cost(
    y::Int,
    params::ZhengFedergruenParameters;
    cache::Dict{Int, Float64}=Dict{Int, Float64}()
)
    haskey(cache, y) && return cache[y]

    cost = 0.0
    for d in 0:params.demand_support_max
        prob_d = pdf(params.demand_dist, d)
        prob_d > 1e-9 || continue
        inv_level = y - d
        if inv_level > 0
            cost += params.h * inv_level * prob_d
        else
            cost += -params.p * inv_level * prob_d
        end
    end
    cache[y] = cost
    return cost
end


"""
    precompute_renewal_functions(params, max_n)

Pre-compute the renewal functions m(j) and M(j) up to
horizon `max_n`, as defined in Equations (2a, 2b, 2c)
of Zheng and Federgruen (1991).

Returns arrays where m[j+1] = m(j) and M[j+1] = M(j)
(1-based indexing).
"""
function precompute_renewal_functions(
    params::ZhengFedergruenParameters,
    max_n::Int
)
    probs = [
        pdf(params.demand_dist, j)
        for j in 0:params.demand_support_max
    ]
    p0 = probs[1]

    m = zeros(max_n + 1)
    M = zeros(max_n + 1)

    if isapprox(p0, 1.0)
        return m, M
    end

    # m(0) = 1 / (1 - p₀)
    m[1] = 1.0 / (1.0 - p0)

    # m(j) = m(0) * Σ_{l=1}^{j} p_l * m(j-l)
    for j in 1:max_n
        m_sum = sum(
            probs[l + 1] * m[j - l + 1]
            for l in 1:min(j, params.demand_support_max)
        )
        m[j + 1] = m[1] * m_sum
    end

    # M(n) = Σ_{j=0}^{n-1} m(j)
    for j in 1:max_n
        M[j + 1] = M[j] + m[j]
    end

    return m, M
end


"""
    evaluate_average_cost(s, S, params, m, M; cache)

Compute the long-run average cost c(s,S) using Equation (1)
from Zheng and Federgruen (1991):

    c(s,S) = [K + Σ_{j=0}^{n-1} m(j) G(S-j)] / M(n)

where n = S - s.
"""
function evaluate_average_cost(
    s::Int,
    S::Int,
    params::ZhengFedergruenParameters,
    m::Vector{Float64},
    M::Vector{Float64};
    cache::Dict{Int, Float64}=Dict{Int, Float64}()
)
    n = S - s
    if n <= 0 || n >= length(M)
        return Inf
    end

    numerator = params.K
    for j in 0:(n - 1)
        numerator += m[j + 1] *
            expected_one_period_cost(S - j, params; cache=cache)
    end

    denominator = M[n + 1]
    return denominator > 1e-9 ? numerator / denominator : Inf
end


"""
    find_optimal_sS_single_item(params)

Find the optimal (s,S) policy using the average cost
algorithm from Zheng and Federgruen (1991).

Returns `(s=s_opt, S=S_opt)`.
"""
function find_optimal_sS_single_item(
    params::ZhengFedergruenParameters
)
    cache = Dict{Int, Float64}()

    mean_demand = mean(params.demand_dist)
    std_demand = std(params.demand_dist)
    search_horizon = ceil(
        Int, mean_demand + 4 * std_demand + 250
    )
    m, M = precompute_renewal_functions(params, search_horizon)

    # Find y* (minimizer of G) as initial S₀
    y_start = round(Int, mean_demand)
    y_opt = y_start
    g_opt = expected_one_period_cost(
        y_start, params; cache=cache
    )
    for y_test in (y_start - 150):(y_start + 150)
        g_test = expected_one_period_cost(
            y_test, params; cache=cache
        )
        if g_test < g_opt
            g_opt = g_test
            y_opt = y_test
        end
    end
    S0 = y_opt

    # Find initial s₀ by searching downward
    s_current = S0 - 1
    while s_current > S0 - search_horizon
        cost = evaluate_average_cost(
            s_current, S0, params, m, M; cache=cache
        )
        if cost <= expected_one_period_cost(
            s_current, params; cache=cache
        )
            break
        end
        s_current -= 1
    end

    s_opt, S_opt = s_current, S0
    c_opt = evaluate_average_cost(
        s_opt, S_opt, params, m, M; cache=cache
    )

    # Main search loop: iterate over increasing S
    for S_test in (S_opt + 1):search_horizon
        # Stopping condition (Lemma 2c)
        if expected_one_period_cost(
            S_test, params; cache=cache
        ) > c_opt
            break
        end

        # Check for improvement (Lemma 3a)
        cost_test = evaluate_average_cost(
            s_current, S_test, params, m, M; cache=cache
        )

        if cost_test < c_opt
            S_current = S_test
            cost_of_new_S = cost_test

            # Search upward for new optimal s (Lemma 3b)
            s_search = s_current
            while s_search < S_current - 1
                cost_at_next = evaluate_average_cost(
                    s_search + 1, S_current, params, m, M;
                    cache=cache
                )
                if cost_at_next < cost_of_new_S
                    cost_of_new_S = cost_at_next
                    s_search += 1
                else
                    break
                end
            end

            s_current = s_search
            c_opt = cost_of_new_S
            s_opt, S_opt = s_current, S_current
        end
    end

    return (s=s_opt, S=S_opt)
end


"""
    compute_sS_policies(inventory_prob, K_distributed)

Compute optimal (s,S) policies for all items using the
Zheng-Federgruen algorithm. Runs items in parallel.

Arguments:
- inventory_prob: InventoryProblem
- K_distributed: Vector of fixed costs per item

Returns:
- policies: Matrix{Int} of size (d, 2) where
  column 1 = s levels, column 2 = S levels
"""
function compute_sS_policies(
    inventory_prob::InventoryProblem,
    K_distributed::Vector{Float64}
)
    d = inventory_prob.dim
    policies = zeros(Int, d, 2)

    @threads for i in 1:d
        params_i = ZhengFedergruenParameters(
            K_distributed[i],
            inventory_prob.holding_costs[i],
            inventory_prob.backlog_costs[i],
            inventory_prob.variable_costs[i],
            inventory_prob.discount_factor,
            inventory_prob.nb_demand[i],
            200
        )

        result = find_optimal_sS_single_item(params_i)
        policies[i, 1] = result.s
        policies[i, 2] = result.S
    end

    return policies
end
