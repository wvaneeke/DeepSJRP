#!/usr/bin/env julia
#
# run_mpdsolver2d.jl - Exact MDP solver for 2D joint replenishment
#
# Solves the 2-item inventory control MDP via policy iteration with
# precomputed transition matrices and matrix operations. Outputs the
# optimal policy and value function.
#
# Usage:
#   julia run_mpdsolver2d.jl <CV> <fixed_cost> <penalty> <instance_name>
#
# Arguments:
#   CV              Annual coefficient of variation (0.2, 0.5, or 1.0)
#   fixed_cost      Fixed ordering cost per replenishment
#   penalty         Per-unit annual backlog penalty cost
#   instance_name   Name used for output file naming
#
# Naming convention: 2D_Case<CV><Fixed><Penalty>
#   L=Low, M=Medium, H=High
#   CV:      L=0.2, M=0.5, H=1.0
#   Fixed:   L=20,  M=50,  H=100
#   Penalty: L=10,  M=50,  H=100
#
# Examples:
#   julia run_mpdsolver2d.jl 0.5 50.0 50.0 2D_CaseMMM
#   julia run_mpdsolver2d.jl 0.2 50.0 50.0 2D_CaseLMM
#   julia run_mpdsolver2d.jl 1.0 20.0 100.0 2D_CaseHLH
#
# Output:
#   output/2d_mdp/mdppolicy_<instance_name>.jld2

using Random, Distributions
using LinearAlgebra
using JLD2

if length(ARGS) != 4
    error("Usage: julia run_mpdsolver2d.jl <CV> <fixed_cost> <penalty> <instance_name>")
end

# Parse arguments
D_annualCV_arg = parse(Float64, ARGS[1])
fixed_cost_arg = parse(Float64, ARGS[2])
penalty_cost_arg = parse(Float64, ARGS[3])
instance_name = ARGS[4]

# ============================================================================
# Problem parameters
# ============================================================================

const Δn = 1 / 52
const D_weeklymean = [40.0, 20.0] .* Δn
const holding = [2.0, 2.0] .* Δn
const backlog = [penalty_cost_arg, penalty_cost_arg] .* Δn
const variable_cost = [0.1, 0.4]
const fixed_cost = fixed_cost_arg
const γ = 0.999039

const MIN_INV = -250
const MAX_INV = 100
const Z_MIN = 10
const Z_MAX = 100
const D_TRUNCATED = 250

# State space dimensions
const N_STATES = MAX_INV - MIN_INV + 1  # 256

# ============================================================================
# Setup demand distributions
# ============================================================================

function setup_distributions(cv::Float64)
    if cv == 0.5 || cv == 1.0
        D_weeklyvariance = (sqrt(1 / Δn) * cv .* D_weeklymean).^2
        nb_succesprob = D_weeklymean ./ D_weeklyvariance
        nb_succesnumber = D_weeklymean.^2 ./ (D_weeklyvariance .- D_weeklymean)
        return [NegativeBinomial(r, p) for (r, p) in zip(nb_succesnumber, nb_succesprob)]
    elseif cv == 0.2
        return [Poisson(λ) for λ in D_weeklymean]
    else
        error("Unsupported CV value: $cv. Please use 0.2, 0.5, or 1.0.")
    end
end

# ============================================================================
# Precomputation functions
# ============================================================================

"""
Precompute demand probabilities for a single item, truncated at D_TRUNCATED.
Returns a vector of length D_TRUNCATED+1.
"""
function precompute_demand_probs(dist)
    probs = zeros(D_TRUNCATED + 1)
    for d in 0:D_TRUNCATED-1
        probs[d+1] = pdf(dist, d)
    end
    # Lump tail probability onto the truncation point
    probs[D_TRUNCATED+1] = 1.0 - cdf(dist, D_TRUNCATED - 1)
    return probs
end

"""
Precompute holding/backlog costs for all states.
Returns an N_STATES × N_STATES matrix.
"""
function precompute_hb_costs()
    costs = zeros(N_STATES, N_STATES)
    for i1 in 1:N_STATES
        s1 = i1 + MIN_INV - 1
        h1 = holding[1] * max(s1, 0)
        b1 = backlog[1] * max(-s1, 0)
        for i2 in 1:N_STATES
            s2 = i2 + MIN_INV - 1
            h2 = holding[2] * max(s2, 0)
            b2 = backlog[2] * max(-s2, 0)
            costs[i1, i2] = h1 + b1 + h2 + b2
        end
    end
    return costs
end

"""
Precompute transition probabilities from order-up-to level z to next state.
Returns an (Z_MAX - Z_MIN + 1) × N_STATES matrix for one item.
Each row iz corresponds to order-up-to level z = Z_MIN + iz - 1.
"""
function precompute_trans_from_z(demand_probs)
    n_z = Z_MAX - Z_MIN + 1
    trans = zeros(n_z, N_STATES)
    
    for iz in 1:n_z
        z = Z_MIN + iz - 1
        for d in 0:D_TRUNCATED
            s_prime = z - d
            s_prime = clamp(s_prime, MIN_INV, MAX_INV)
            idx = s_prime - MIN_INV + 1
            trans[iz, idx] += demand_probs[d+1]
        end
    end
    return trans
end

"""
Precompute transition probabilities for no-order case (s' = s - d).
Returns an N_STATES × N_STATES matrix for one item.
"""
function precompute_trans_no_order(demand_probs)
    trans = zeros(N_STATES, N_STATES)
    
    for is in 1:N_STATES
        s = is + MIN_INV - 1
        for d in 0:D_TRUNCATED
            s_prime = s - d
            s_prime = clamp(s_prime, MIN_INV, MAX_INV)
            idx = s_prime - MIN_INV + 1
            trans[is, idx] += demand_probs[d+1]
        end
    end
    return trans
end

# ============================================================================
# Core algorithm using matrix operations
# ============================================================================

"""
Compute E[-f(s') + γV(s')] for all (z1, z2) combinations using matrix operations.

The key insight: since demands are independent,
    E[W(s1', s2') | z1, z2] = Σ P(s1'|z1) P(s2'|z2) W(s1', s2')
                            = (trans1[z1, :])' * W * trans2[z2, :]
                            = trans1 * W * trans2'

Returns a matrix of size (n_z1, n_z2).
"""
function compute_expected_values_order_both(W, trans1_z, trans2_z)
    return trans1_z * W * trans2_z'
end

"""
Compute E[W] for ordering only item 1, for all (z1, s2) combinations.
Returns a matrix of size (n_z, N_STATES).
"""
function compute_expected_values_order_1_only(W, trans1_z, trans2_no)
    # For each z1 and each current s2:
    # E[W | z1, s2] = trans1_z[z1,:] * W * trans2_no[s2,:]'
    # = (trans1_z * W) * trans2_no'   but we want rows indexed by z1 and s2
    # Result[iz1, is2] = sum over s1', s2' of trans1_z[iz1, s1'] * W[s1', s2'] * trans2_no[is2, s2']
    # = (trans1_z * W * trans2_no')[iz1, is2]
    return trans1_z * W * trans2_no'
end

"""
Compute E[W] for ordering only item 2, for all (s1, z2) combinations.
Returns a matrix of size (N_STATES, n_z).
"""
function compute_expected_values_order_2_only(W, trans1_no, trans2_z)
    return trans1_no * W * trans2_z'
end

"""
Compute E[W] for no order, for all (s1, s2) combinations.
Returns a matrix of size (N_STATES, N_STATES).
"""
function compute_expected_values_no_order(W, trans1_no, trans2_no)
    return trans1_no * W * trans2_no'
end

"""
Policy iteration with precomputed transitions and matrix operations.
"""
function policy_iteration_optimized(demand_dists)
    println("Precomputing transition matrices..."); flush(stdout)
    
    # Precompute demand probabilities
    demand_probs1 = precompute_demand_probs(demand_dists[1])
    demand_probs2 = precompute_demand_probs(demand_dists[2])
    
    # Precompute holding/backlog costs
    hb_costs = precompute_hb_costs()
    
    # Precompute transition matrices
    trans1_z = precompute_trans_from_z(demand_probs1)   # (n_z, N_STATES)
    trans2_z = precompute_trans_from_z(demand_probs2)   # (n_z, N_STATES)
    trans1_no = precompute_trans_no_order(demand_probs1) # (N_STATES, N_STATES)
    trans2_no = precompute_trans_no_order(demand_probs2) # (N_STATES, N_STATES)
    
    n_z = Z_MAX - Z_MIN + 1
    
    # Precompute variable ordering costs
    # var_cost1[iz] = variable_cost[1] * (z - s1) where z = Z_MIN + iz - 1
    # But this depends on s1, so we compute: var_cost1[iz] = variable_cost[1] * z
    # and subtract variable_cost[1] * s1 later
    var_cost1_z = [variable_cost[1] * (Z_MIN + iz - 1) for iz in 1:n_z]
    var_cost2_z = [variable_cost[2] * (Z_MIN + iz - 1) for iz in 1:n_z]
    
    println("Precomputation done. Starting policy iteration..."); flush(stdout)
    
    # Initialize value function and policy
    V = zeros(N_STATES, N_STATES)
    # Policy: store (z1, z2) as order-up-to levels; z_i = s_i means don't order item i
    policy_z1 = zeros(Int, N_STATES, N_STATES)
    policy_z2 = zeros(Int, N_STATES, N_STATES)
    
    # Initialize policy to no-order
    for i1 in 1:N_STATES, i2 in 1:N_STATES
        policy_z1[i1, i2] = i1 + MIN_INV - 1  # z1 = s1
        policy_z2[i1, i2] = i2 + MIN_INV - 1  # z2 = s2
    end
    
    iteration = 0
    while true
        iteration += 1
        println("Policy iteration: $iteration"); flush(stdout)
        
        # ====================================================================
        # Policy Evaluation (iterate until convergence)
        # ====================================================================
        @time begin
            eval_iters = 0
            while true
                eval_iters += 1
                W = -hb_costs .+ γ .* V
                
                # Compute expected values for all action types
                EV_no_order = compute_expected_values_no_order(W, trans1_no, trans2_no)
                EV_order_1 = compute_expected_values_order_1_only(W, trans1_z, trans2_no)
                EV_order_2 = compute_expected_values_order_2_only(W, trans1_no, trans2_z)
                EV_order_both = compute_expected_values_order_both(W, trans1_z, trans2_z)
                
                V_new = similar(V)
                
                Threads.@threads for i1 in 1:N_STATES
                    s1 = i1 + MIN_INV - 1
                    for i2 in 1:N_STATES
                        s2 = i2 + MIN_INV - 1
                        
                        z1 = policy_z1[i1, i2]
                        z2 = policy_z2[i1, i2]
                        
                        order_1 = z1 > s1
                        order_2 = z2 > s2
                        
                        if !order_1 && !order_2
                            # No order
                            V_new[i1, i2] = EV_no_order[i1, i2]
                        elseif order_1 && !order_2
                            # Order item 1 only
                            iz1 = z1 - Z_MIN + 1
                            order_cost = fixed_cost + variable_cost[1] * (z1 - s1)
                            V_new[i1, i2] = -order_cost + EV_order_1[iz1, i2]
                        elseif !order_1 && order_2
                            # Order item 2 only
                            iz2 = z2 - Z_MIN + 1
                            order_cost = fixed_cost + variable_cost[2] * (z2 - s2)
                            V_new[i1, i2] = -order_cost + EV_order_2[i1, iz2]
                        else
                            # Order both
                            iz1 = z1 - Z_MIN + 1
                            iz2 = z2 - Z_MIN + 1
                            order_cost = fixed_cost + variable_cost[1] * (z1 - s1) + variable_cost[2] * (z2 - s2)
                            V_new[i1, i2] = -order_cost + EV_order_both[iz1, iz2]
                        end
                    end
                end
                
                max_delta = maximum(abs.(V_new .- V))
                V .= V_new
                
                if max_delta < 1e-4
                    println("  Policy evaluation converged in $eval_iters iterations (δ = $max_delta)")
                    break
                end
            end
        end
        flush(stdout)
        
        # Report V(0,0)
        i0 = 0 - MIN_INV + 1
        println("  Cost at V(0,0) = ", -V[i0, i0]); flush(stdout)
        
        # ====================================================================
        # Policy Improvement
        # ====================================================================
        @time begin
            W = -hb_costs .+ γ .* V
            
            # Precompute all expected values
            EV_no_order = compute_expected_values_no_order(W, trans1_no, trans2_no)
            EV_order_1 = compute_expected_values_order_1_only(W, trans1_z, trans2_no)
            EV_order_2 = compute_expected_values_order_2_only(W, trans1_no, trans2_z)
            EV_order_both = compute_expected_values_order_both(W, trans1_z, trans2_z)
            
            policy_stable = true
            
            Threads.@threads for i1 in 1:N_STATES
                s1 = i1 + MIN_INV - 1
                for i2 in 1:N_STATES
                    s2 = i2 + MIN_INV - 1
                    
                    old_z1 = policy_z1[i1, i2]
                    old_z2 = policy_z2[i1, i2]
                    
                    best_val = -Inf
                    best_z1, best_z2 = s1, s2
                    
                    # Option 1: No order
                    val = EV_no_order[i1, i2]
                    if val > best_val
                        best_val = val
                        best_z1, best_z2 = s1, s2
                    end
                    
                    # Option 2: Order item 1 only
                    z1_start = max(Z_MIN, s1 + 1)
                    for z1 in z1_start:Z_MAX
                        iz1 = z1 - Z_MIN + 1
                        order_cost = fixed_cost + variable_cost[1] * (z1 - s1)
                        val = -order_cost + EV_order_1[iz1, i2]
                        if val > best_val
                            best_val = val
                            best_z1, best_z2 = z1, s2
                        end
                    end
                    
                    # Option 3: Order item 2 only
                    z2_start = max(Z_MIN, s2 + 1)
                    for z2 in z2_start:Z_MAX
                        iz2 = z2 - Z_MIN + 1
                        order_cost = fixed_cost + variable_cost[2] * (z2 - s2)
                        val = -order_cost + EV_order_2[i1, iz2]
                        if val > best_val
                            best_val = val
                            best_z1, best_z2 = s1, z2
                        end
                    end
                    
                    # Option 4: Order both items
                    for z1 in z1_start:Z_MAX
                        iz1 = z1 - Z_MIN + 1
                        cost1 = variable_cost[1] * (z1 - s1)
                        for z2 in z2_start:Z_MAX
                            iz2 = z2 - Z_MIN + 1
                            order_cost = fixed_cost + cost1 + variable_cost[2] * (z2 - s2)
                            val = -order_cost + EV_order_both[iz1, iz2]
                            if val > best_val
                                best_val = val
                                best_z1, best_z2 = z1, z2
                            end
                        end
                    end
                    
                    policy_z1[i1, i2] = best_z1
                    policy_z2[i1, i2] = best_z2
                    
                    if best_z1 != old_z1 || best_z2 != old_z2
                        policy_stable = false
                    end
                end
            end
            
            println("  Policy stable: $policy_stable")
        end
        flush(stdout)
        
        if policy_stable
            println("Policy iteration converged!")
            break
        end
    end
    
    # Convert policy to action format (order quantities, not order-up-to levels)
    policy_actions = Dict{Tuple{Int16, Int16}, Tuple{Int16, Int16}}()
    for i1 in 1:N_STATES
        s1 = i1 + MIN_INV - 1
        for i2 in 1:N_STATES
            s2 = i2 + MIN_INV - 1
            z1 = policy_z1[i1, i2]
            z2 = policy_z2[i1, i2]
            a1 = z1 - s1
            a2 = z2 - s2
            policy_actions[(Int16(s1), Int16(s2))] = (Int16(a1), Int16(a2))
        end
    end
    
    # Also convert V to Dict format for compatibility
    V_dict = Dict{Tuple{Int16, Int16}, Float64}()
    for i1 in 1:N_STATES
        s1 = i1 + MIN_INV - 1
        for i2 in 1:N_STATES
            s2 = i2 + MIN_INV - 1
            V_dict[(Int16(s1), Int16(s2))] = V[i1, i2]
        end
    end
    
    return policy_actions, V_dict, V
end

# ============================================================================
# Main
# ============================================================================

println("="^60)
println("MDP Solver 2D (Optimized)")
println("="^60)

println("\n--- Instance ---")
println("Instance name: $instance_name")

println("\n--- Demand Parameters ---")
println("Annual demand rates: ", [40.0, 20.0])
println("Weekly demand means: ", D_weeklymean)
println("Annual CV: $D_annualCV_arg")
println("Distribution: ", D_annualCV_arg == 0.2 ? "Poisson" : "NegativeBinomial")

println("\n--- Cost Parameters ---")
println("Holding costs (annual): ", [2.0, 2.0])
println("Holding costs (weekly): ", holding)
println("Penalty costs (annual): ", [penalty_cost_arg, penalty_cost_arg])
println("Penalty costs (weekly): ", backlog)
println("Variable costs: ", variable_cost)
println("Fixed cost: ", fixed_cost)

println("\n--- Discount ---")
println("Weekly discount factor γ: ", γ)
println("Implied annual interest rate: ", round(-52 * log(γ), digits=4))

println("\n--- State & Action Space ---")
println("State space: [$MIN_INV, $MAX_INV]² = $(N_STATES)² = $(N_STATES^2) states")
println("Order-up-to range: [$Z_MIN, $Z_MAX] (n_z = $(Z_MAX - Z_MIN + 1))")

println("\n--- Numerical Parameters ---")
println("Demand truncation: $D_TRUNCATED")
println("Time unit Δn: $Δn (weekly)")

println("="^60)
flush(stdout)

# Setup distributions
demand_dists = setup_distributions(D_annualCV_arg)

# Print distribution details
println("\n--- Distribution Details ---")
for (i, dist) in enumerate(demand_dists)
    println("Item $i: ", dist)
    println("  Mean: ", round(mean(dist), digits=4))
    println("  Variance: ", round(var(dist), digits=4))
    println("  P(D > $D_TRUNCATED): ", round(100 * (1 - cdf(dist, D_TRUNCATED)), digits=4), "%")
end
println("="^60)
flush(stdout)

# Solve
@time mdp_policy, V_dict, V_array = policy_iteration_optimized(demand_dists)

# Save results
isdir("output/2d_mdp") || mkpath("output/2d_mdp")
JLD2.@save "output/2d_mdp/mdppolicy_$instance_name.jld2" mdp_policy

# Report results
i0 = 0 - MIN_INV + 1
println("="^60)
println("Final Results")
println("="^60)
println("Cost at V(0,0) = ", -V_array[i0, i0])
println("Action at (0, 0): ", mdp_policy[(Int16(0), Int16(0))])
println("Action at (-50, -50): ", mdp_policy[(Int16(-50), Int16(-50))])
println("Policy saved to: output/2d_mdp/mdppolicy_$instance_name.jld2")