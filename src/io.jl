# io.jl - JSON input/output utilities

using JSON
using JLD2
using Dates
using Distributions


"""
    load_problem_json(filepath::String)

Load problem parameters from JSON file.

Expected JSON structure:
{
    "name": "12D_CaseLLL",
    "dimension": 12,
    "annual_demand_rates": [...],
    "annual_cv": 0.2,
    "distribution": "Poisson",  // or "NegativeBinomial"
    "interest_rate": 0.05,
    "variable_costs": [...],
    "holding_costs": [...],
    "penalty_costs": [...],
    "fixed_cost": 20
}

Returns:
- impulse_prob: ImpulseControlProblem
- inventory_prob: InventoryProblem
- instance_name: Problem instance name
"""
function load_problem_json(filepath::String; cost_scaling::Float64=100.0)
    data = JSON.parsefile(filepath)
    
    # Extract parameters
    instance_name = data["name"]
    dim = data["dimension"]
    D_annualmean_raw = Float64.(data["annual_demand_rates"])
    D_annualCV = data["annual_cv"]
    D_distribution = Symbol(data["distribution"])
    interest_rate = data["interest_rate"]
    
    # Cost parameters (apply scaling)
    κ = cost_scaling
    variable_costs = Float64.(data["variable_costs"]) * κ
    holding_costs = Float64.(data["holding_costs"]) * κ
    penalty_costs = Float64.(data["penalty_costs"]) * κ
    fixed_cost = data["fixed_cost"] * κ
    
    # Time conversion (weekly)
    Δn = 1 / 52
    D_weeklymean = D_annualmean_raw * Δn
    
    # Create demand distributions
    if D_distribution == :NegativeBinomial
        D_weeklyvariance = (sqrt(1 / Δn) * D_annualCV * D_weeklymean).^2
        nb_succesprob = D_weeklymean ./ D_weeklyvariance
        nb_succesnumber = D_weeklymean.^2 ./ (D_weeklyvariance .- D_weeklymean)
        nb_dists = NegativeBinomial.(nb_succesnumber, nb_succesprob)
    elseif D_distribution == :Poisson
        D_weeklyvariance = 1.0 * D_weeklymean
        nb_dists = Poisson.(D_weeklymean)
    else
        error("Unknown distribution type: $(D_distribution)")
    end
    
    # Compute annual parameters for impulse control
    μ = D_weeklymean * (1 / Δn)  # Annual demand rate
    Σ = Matrix(Diagonal(D_weeklyvariance * (1 / Δn)))  # Annual covariance
    
    # Create ImpulseControlProblem
    impulse_prob = ImpulseControlProblem(
        dim, μ, Σ, interest_rate,
        holding_costs, penalty_costs, variable_costs, fixed_cost
    )
    
    # Create InventoryProblem (for discrete-time simulation)
    inventory_prob = InventoryProblem(
        dim,
        variable_costs / κ,     # Variable cost (unscaled)
        holding_costs * Δn / κ, # Weekly holding cost
        penalty_costs * Δn / κ, # Weekly backlog cost
        fixed_cost / κ,         # Fixed cost (unscaled)
        nb_dists,               # Demand distributions
        exp(-interest_rate * Δn) # Weekly discount factor
    )
    
    return impulse_prob, inventory_prob, instance_name
end


"""
    load_hyperparameters_json(filepath::String)

Load hyperparameters from JSON file.

Expected structure for reference_process:
{
    "benchmark_source": "QS",  // or "can_order"
    "nu_radius": 0.2,
    "alpha_factor": 0.4,
    "S_distribution": "lognormal",
    "QS": {
        "average_weeks_between_orders": 13.0,
        "S_input": [...]
    },
    "can_order": {
        "average_weeks_between_orders": 6.5,
        "S_input": [...]
    }
}

Returns:
- alg: AlgorithmHyperParameters
- nn_params: NeuralNetworkParameters
- opt_params: OptimizationParameters
- S_input: Selected S vector
- benchmark_source: Which benchmark was selected ("QS" or "can_order")
"""
function load_hyperparameters_json(filepath::String)
    data = JSON.parsefile(filepath)
    
    # Neural network parameters
    nn = data["neural_network"]
    nn_params = NeuralNetworkParameters(
        num_hidden_layers=nn["num_hidden_layers"],
        nn_width=nn["nn_width"],
        activation_name=Symbol(nn["activation"])
    )
    
    # Training parameters
    train = data["training"]
    lr = data["learning_rates"]
    penalty = data["penalty_rates"]
    ref = data["reference_process"]
    
    # Get benchmark source and select appropriate parameters
    benchmark_source = get(ref, "benchmark_source", "QS")
    
    # Get the selected benchmark's parameters
    benchmark_params = ref[benchmark_source]
    S_input = Float64.(benchmark_params["S_input"])
    avg_weeks = benchmark_params["average_weeks_between_orders"]
    λ_rate = 52.0 / avg_weeks
    
    alg = AlgorithmHyperParameters(
        precision=Float32,
        batch_size=train["batch_size"],
        num_intervals=train["num_intervals"],
        T_horizon=train["T_horizon"],
        num_iterations=train["num_iterations"],
        learning_rates=Float64.(lr["values"]),
        decay_steps=Int.(lr["steps"]),
        β_penalty=Float64.(penalty["values"]),
        penalty_steps=Int.(penalty["steps"]),
        λ_rate=λ_rate,
        S_guess=[S_input],
        ν_radius=ref["nu_radius"],
        S_distribution=Symbol(ref["S_distribution"]),
        αjs=Float64[],  # Placeholder; recomputed with actual μ in run()
        cost_scaling=train["cost_scaling"]
    )
    
    # Optimization parameters
    opt = data["optimization"]
    grid = get(data, "nn_grid", Dict{String, Any}())
    opt_params = OptimizationParameters(
        start_factor=opt["start_factor"],
        lower_factor=opt["lower_factor"],
        upper_factor=opt["upper_factor"],
        epsilon_factors=Float64.(opt["epsilon_factors"]),
        num_simulation_runs=get(opt, "num_simulation_runs", 8),
        simulation_horizon=get(opt, "simulation_horizon", 5000),
        nn_grid_min_inv=get(grid, "min_inv", -25),
        nn_grid_max_inv=get(grid, "max_inv", 75)
    )
    
    alpha_factor = ref["alpha_factor"]

    return alg, nn_params, opt_params, S_input, benchmark_source,
        alpha_factor
end


"""
    load_results_json(filepath::String)

Load a previously saved per-method results JSON file.

Returns the parsed Dict if the file exists, or `nothing`
if the file does not exist.
"""
function load_results_json(filepath::String)
    if !isfile(filepath)
        return nothing
    end
    return JSON.parsefile(filepath)
end


"""
    save_method_results_json(
        filepath, method, method_results,
        instance_name, alg; kwargs...
    )

Save evaluation results for a single method to a JSON
file. Before writing, checks if `filepath` already exists.
If so, loads the existing file and compares `best_cost`.
The file is only overwritten when the new `best_cost` is
strictly lower (better) than the existing one.

Returns `true` if the file was written, `false` if the
existing result was better.
"""
function save_method_results_json(
    filepath::String,
    method::Symbol,
    method_results::Dict,
    instance_name::String,
    alg::AlgorithmHyperParameters;
    μ::Vector{Float64}=Float64[],
    V0_estimate::Float64=0.0,
    benchmark_source::String="QS",
    training_time::Float64=0.0
)
    new_cost = method_results[:best_cost]

    # Check existing file for best-so-far comparison
    existing = load_results_json(filepath)
    if !isnothing(existing)
        existing_cost = existing["best_cost"]
        if new_cost >= existing_cost
            println(
                "Skipping save for $(method): " *
                "existing cost $(existing_cost)" *
                " <= new cost $(new_cost)"
            )
            return false
        end
        println(
            "Improving $(method): " *
            "$(existing_cost) -> $(new_cost)"
        )
    end

    mkpath(dirname(filepath))

    alpha_factor = (
        length(alg.αjs) > 0 && length(μ) > 0
        ? alg.αjs[1] / (μ[1] / alg.λ_rate)
        : 0.0
    )

    output = Dict(
        "instance_name" => instance_name,
        "timestamp" => string(Dates.now()),
        "benchmark_source" => benchmark_source,
        "method" => string(method),
        "hyperparameters" => Dict(
            "T_horizon" => alg.T_horizon,
            "beta_final" => last(alg.β_penalty),
            "nu_radius" => alg.ν_radius,
            "alpha_factor" => alpha_factor,
            "lambda_rate" => alg.λ_rate,
            "cost_scaling" => alg.cost_scaling
        ),
        "V0_estimate" => V0_estimate,
        "z_star" => method_results[:z_star],
        "z_star_int" => method_results[:z_star_int],
        "best_epsilon" => method_results[:best_epsilon],
        "best_cost" => method_results[:best_cost],
        "best_std_error" =>
            method_results[:best_std_error],
        "training_time_seconds" => training_time
    )

    open(filepath, "w") do f
        JSON.print(f, output, 2)
    end

    println("Results saved to: $(filepath)")
    return true
end


"""
    load_mdp_policy_jld2(filepath::String)

Load a pre-computed MDP optimal policy from a JLD2 file.
The file must contain a `mdp_policy` variable of type
`Dict{Tuple{Int16,Int16}, Tuple{Int16,Int16}}` mapping
inventory states to order quantities.

Returns an `MDPPolicy` if the file exists, or `nothing`
if the file does not exist.
"""
function load_mdp_policy_jld2(filepath::String)
    if !isfile(filepath)
        return nothing
    end
    data = JLD2.load(filepath)
    policy_dict = data["mdp_policy"]
    return MDPPolicy(policy_dict)
end


"""
    save_nn_grid_policy_jld2(filepath, grid_policy)

Persist an `NNGridPolicy` to a JLD2 file. Stores the
state-action dict, grid bounds, z*, and ε.
"""
function save_nn_grid_policy_jld2(
    filepath::String, grid_policy::NNGridPolicy
)
    mkpath(dirname(filepath))
    JLD2.jldsave(
        filepath;
        policy=grid_policy.policy,
        min_inv=grid_policy.min_inv,
        max_inv=grid_policy.max_inv,
        z_star=grid_policy.z_star,
        epsilon=grid_policy.epsilon
    )
    return filepath
end


"""
    load_nn_grid_policy_jld2(filepath::String)

Load a saved `NNGridPolicy` from JLD2, or return `nothing`
if the file does not exist.
"""
function load_nn_grid_policy_jld2(filepath::String)
    if !isfile(filepath)
        return nothing
    end
    data = JLD2.load(filepath)
    return NNGridPolicy(
        data["policy"],
        Int16(data["min_inv"]),
        Int16(data["max_inv"]),
        Int.(data["z_star"]),
        Float64(data["epsilon"])
    )
end


"""
    convert_xlsx_to_json(xlsx_path::String, json_path::String)

Convert old XLSX input format to new JSON format.
"""
function convert_xlsx_to_json(xlsx_path::String, json_path::String)
    # This would require XLSX.jl, so just document the conversion
    error("XLSX conversion requires XLSX.jl package. Please convert manually.")
end


"""
    print_problem_summary(impulse_prob, inventory_prob, instance_name)

Print a summary of the loaded problem.
"""
function print_problem_summary(
    impulse_prob::ImpulseControlProblem,
    inventory_prob::InventoryProblem,
    instance_name::String;
    cost_scaling::Float64=100.0
)
    κ = cost_scaling

    println("\n" * "="^60)
    println("IMPULSE CONTROL PROBLEM (scaled, annual)")
    println("="^60)
    println("Instance: $(instance_name)")
    println("Dimension: $(impulse_prob.dim)")
    println("Interest rate: $(impulse_prob.interest_rate)")
    println("Cost scaling κ: $(κ)")
    println("\nDrift μ (annual demand rates): $(impulse_prob.μ)")
    println("Covariance Σ diagonal: $(diag(impulse_prob.Σ))")
    println("\nScaled costs (× κ = $(κ)):")
    println("  Fixed cost K:    $(impulse_prob.fixed_cost)")
    println("  Variable costs c: $(impulse_prob.variable_costs)")
    println("  Holding costs h:  $(impulse_prob.holding_costs)")
    println("  Penalty costs p:  $(impulse_prob.penalty_costs)")
    println("\nOriginal costs (unscaled):")
    println("  Fixed cost K:    $(impulse_prob.fixed_cost / κ)")
    println("  Variable costs c: $(impulse_prob.variable_costs / κ)")
    println("  Holding costs h:  $(impulse_prob.holding_costs / κ)")
    println("  Penalty costs p:  $(impulse_prob.penalty_costs / κ)")
    println("="^60)
end


"""
    print_simulation_summary(
        impulse_prob, inventory_prob, instance_name;
        cost_scaling
    )

Print a summary of the problem for simulation runs, showing
the original annual input parameters (unscaled) and the
weekly inventory problem used in discrete-time simulation.
"""
function print_simulation_summary(
    impulse_prob::ImpulseControlProblem,
    inventory_prob::InventoryProblem,
    instance_name::String;
    cost_scaling::Float64=100.0
)
    κ = cost_scaling

    println("\n" * "="^60)
    println("PROBLEM INPUT (annual, unscaled)")
    println("="^60)
    println("Instance:          $(instance_name)")
    println("Dimension:         $(impulse_prob.dim)")
    println("Interest rate:     $(impulse_prob.interest_rate)")
    println(
        "Annual demand μ:   " *
        "$(impulse_prob.μ)"
    )
    println(
        "Fixed cost K:      " *
        "$(impulse_prob.fixed_cost / κ)"
    )
    println(
        "Variable costs c:  " *
        "$(impulse_prob.variable_costs / κ)"
    )
    println(
        "Holding costs h:   " *
        "$(impulse_prob.holding_costs / κ)"
    )
    println(
        "Penalty costs p:   " *
        "$(impulse_prob.penalty_costs / κ)"
    )

    println("\n" * "-"^60)
    println("INVENTORY PROBLEM (weekly, unscaled)")
    println("-"^60)
    println(
        "Discount factor:   " *
        "$(inventory_prob.discount_factor)"
    )
    println(
        "Fixed cost:        " *
        "$(inventory_prob.fixed_cost)"
    )
    println(
        "Variable costs:    " *
        "$(inventory_prob.variable_costs)"
    )
    println(
        "Holding costs:     " *
        "$(inventory_prob.holding_costs)"
    )
    println(
        "Backlog costs:     " *
        "$(inventory_prob.backlog_costs)"
    )
    println(
        "Demand dists:      " *
        "$(typeof.(inventory_prob.nb_demand))"
    )
    println(
        "Demand means:      " *
        "$(mean.(inventory_prob.nb_demand))"
    )
    println("="^60)
end


# =====================================================================
# Benchmark simulation I/O
# =====================================================================

"""
    save_benchmark_params_json(filepath, results; kwargs...)

Save preliminary benchmark search results to JSON.
"""
function save_benchmark_params_json(
    filepath::String,
    results::Dict{Symbol, BenchmarkSearchResult};
    instance_name::String="",
    iterations::Int=10_000,
    time_horizon::Int=10_000,
    seed::Int=777
)
    mkpath(dirname(filepath))

    output = Dict{String, Any}(
        "instance_name" => instance_name,
        "timestamp" => string(Dates.now()),
        "search_settings" => Dict(
            "iterations" => iterations,
            "time_horizon" => time_horizon,
            "seed" => seed
        )
    )

    for (key, result) in results
        key_str = string(key)
        entry = Dict{String, Any}(
            "cost" => result.cost,
            "std_error" => result.std_error,
            "ordering_frequency" => result.ordering_frequency
        )

        if !isnothing(result.analytical_cost)
            entry["analytical_cost"] = result.analytical_cost
        end

        # Store policy-specific parameters
        policy = result.policy
        if policy isa IndividualSSPolicy
            entry["best_alpha"] = policy.alpha
            entry["s_levels"] = policy.s_levels
            entry["S_levels"] = policy.S_levels
        elseif policy isa PeriodicRSPolicy
            entry["best_R"] = policy.R
            entry["S_levels"] = policy.S_levels
        elseif policy isa AggregateDemandQSPolicy
            entry["best_Q"] = policy.Q
            entry["S_levels"] = policy.S_levels
            entry["average_weeks_between_orders"] =
                get(
                    result.search_params,
                    :average_weeks_between_orders,
                    Inf
                )
        elseif policy isa HybridRQSPolicy
            entry["best_R"] = policy.R
            entry["best_Q"] = policy.Q
            entry["S_levels"] = policy.S_levels
            entry["average_weeks_between_orders"] =
                get(
                    result.search_params,
                    :average_weeks_between_orders,
                    Inf
                )
        elseif policy isa CanOrderPolicy
            entry["best_alpha"] = policy.alpha
            entry["best_omega"] = policy.omega
            entry["s_levels"] = policy.s_levels
            entry["S_levels"] = policy.S_levels
            freq = result.ordering_frequency
            entry["average_weeks_between_orders"] =
                freq > 0 ? 1.0 / freq : Inf
        elseif policy isa MDPPolicy
            # No policy-specific params to save; the
            # policy is loaded from JLD2 at runtime.
        end

        output[key_str] = entry
    end

    open(filepath, "w") do f
        JSON.print(f, output, 2)
    end

    println("Benchmark params saved to: $(filepath)")
    return nothing
end


"""
    load_benchmark_params_json(filepath)

Load previously saved benchmark parameters.
Returns the parsed Dict, or `nothing` if missing.
"""
function load_benchmark_params_json(filepath::String)
    if !isfile(filepath)
        return nothing
    end
    return JSON.parsefile(filepath)
end


"""
    save_final_comparison_json(
        filepath, nn_result, benchmark_results,
        relative_diffs; kwargs...
    )

Save the final high-precision comparison results.
"""
function save_final_comparison_json(
    filepath::String,
    nn_result::Dict{Symbol, Any},
    benchmark_results::Dict{Symbol, Dict{Symbol, Any}},
    relative_diffs::Dict{Symbol, Tuple{Float64, Float64}};
    instance_name::String="",
    benchmark_iterations::Int=100_000,
    nn_iterations::Int=100,
    time_horizon::Int=10_000,
    seed::Int=777,
    training_hyperparameters::Dict{String, Any}=Dict{String, Any}()
)
    mkpath(dirname(filepath))

    output = Dict{String, Any}(
        "instance_name" => instance_name,
        "timestamp" => string(Dates.now()),
        "simulation_settings" => Dict(
            "benchmark_iterations" => benchmark_iterations,
            "nn_iterations" => nn_iterations,
            "time_horizon" => time_horizon,
            "seed" => seed
        ),
        "neural_network" => Dict{String, Any}(
            "method" => string(
                get(nn_result, :method, :unknown)
            ),
            "z_star" => get(nn_result, :z_star, Int[]),
            "epsilon" => get(nn_result, :epsilon, NaN),
            "cost" => nn_result[:cost],
            "std_error" => nn_result[:std_error],
            "pct_std_error" =>
                nn_result[:std_error] /
                nn_result[:cost] * 100.0,
            "cost_breakdown" => get(
                nn_result, :cost_breakdown, Dict()
            ),
            "training_hyperparameters" =>
                training_hyperparameters
        )
    )

    benchmarks = Dict{String, Any}()
    for (key, res) in benchmark_results
        benchmarks[string(key)] = Dict{String, Any}(
            "cost" => res[:cost],
            "std_error" => res[:std_error],
            "pct_std_error" =>
                res[:std_error] / res[:cost] * 100.0,
            "cost_breakdown" => get(
                res, :cost_breakdown, Dict()
            )
        )
    end
    output["benchmarks"] = benchmarks

    rel_diffs = Dict{String, Any}()
    for (key, (pct_diff, pct_se)) in relative_diffs
        rel_diffs["vs_$(key)"] = Dict(
            "pct_difference" => round(pct_diff; digits=4),
            "pct_se" => round(pct_se; digits=4)
        )
    end
    output["relative_differences"] = rel_diffs

    open(filepath, "w") do f
        JSON.print(f, output, 2)
    end

    println("Final comparison saved to: $(filepath)")
    return nothing
end


"""
    update_config_benchmarks!(
        config_path, qs_result, can_order_result
    )

Update the hyperparameter config file's reference_process
section with optimal benchmark parameters. Also sets
`benchmark_source` to whichever policy achieved the lowest
cost. Preserves all other config fields.
"""
function update_config_benchmarks!(
    config_path::String,
    qs_result::BenchmarkSearchResult,
    can_order_result::BenchmarkSearchResult
)
    data = JSON.parsefile(config_path)

    # Update QS section
    qs_policy = qs_result.policy::AggregateDemandQSPolicy
    qs_freq = qs_result.ordering_frequency
    data["reference_process"]["QS"]["S_input"] =
        qs_policy.S_levels
    data["reference_process"]["QS"][
        "average_weeks_between_orders"
    ] = qs_freq > 0 ? 1.0 / qs_freq : Inf

    # Update can_order section
    can_policy = can_order_result.policy::CanOrderPolicy
    can_freq = can_order_result.ordering_frequency
    data["reference_process"]["can_order"]["S_input"] =
        can_policy.S_levels
    data["reference_process"]["can_order"][
        "average_weeks_between_orders"
    ] = can_freq > 0 ? 1.0 / can_freq : Inf

    # Set benchmark_source to whichever had lower cost
    if qs_result.cost <= can_order_result.cost
        data["reference_process"]["benchmark_source"] = "QS"
        @printf(
            "  benchmark_source → QS (%.2f ≤ %.2f)\n",
            qs_result.cost, can_order_result.cost
        )
    else
        data["reference_process"]["benchmark_source"] =
            "can_order"
        @printf(
            "  benchmark_source → can_order (%.2f < %.2f)\n",
            can_order_result.cost, qs_result.cost
        )
    end

    open(config_path, "w") do f
        JSON.print(f, data, 2)
    end

    println("Config updated: $(config_path)")
    return nothing
end
