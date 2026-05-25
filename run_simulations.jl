#!/usr/bin/env julia
#
# run_simulations.jl - Benchmark and NN policy simulations
#
# Usage:
#   julia run_simulations.jl <instance> --mode preliminary
#   julia run_simulations.jl <instance> --mode final [--nn-iterations 100]
#   julia run_simulations.jl <instance> --mode rqs_search
#
# Modes:
#   preliminary  Search optimal parameters for all 4 benchmarks
#                (10K runs × 10K periods per config). Writes results
#                and updates the hyperparameter config file.
#   final        High-precision simulations (100K benchmark runs,
#                configurable NN runs). Reports absolute NN cost ± %SE
#                and relative % differences vs each benchmark ± %SE.
#   rqs_search   Broad (R,Q,S) parameter search over R=1..75 with
#                coarse Q step (~50 values per R) and 100 iterations.
#
# Options:
#   --mode <preliminary|final|rqs_search>  Required.
#   --nn-iterations <int>       NN simulation runs in final mode (default: 100)
#   --nn-method <method>        Force NN method: solve_inf or find_stationary
#   --seed <int>                Random seed (default: 777)
#   --config-dir <path>         Config directory (default: "configs")
#   --help                      Show this help message

using Pkg
Pkg.activate(@__DIR__)

push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using DeepSJRP
using Printf
using Random
using Statistics
using Dates
using JSON


"""
    parse_simulation_args(args)

Parse command-line arguments for run_simulations.jl.
"""
function parse_simulation_args(args)
    if isempty(args) || "--help" in args
        print_simulation_usage()
        exit(0)
    end

    instance_name = args[1]

    options = Dict{Symbol, Any}(
        :mode => nothing,
        :nn_iterations => 100,
        :nn_method => nothing,
        :seed => 777,
        :config_dir => "configs",
        :input_dir => "input",
        :output_dir => "output",
        :policy_dir => "policies"
    )

    i = 2
    while i <= length(args)
        arg = args[i]
        if arg == "--mode" && i < length(args)
            options[:mode] = Symbol(args[i + 1])
            i += 2
        elseif arg == "--nn-iterations" && i < length(args)
            options[:nn_iterations] = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--seed" && i < length(args)
            options[:seed] = parse(Int, args[i + 1])
            i += 2
        elseif arg == "--nn-method" && i < length(args)
            options[:nn_method] = Symbol(args[i + 1])
            i += 2
        elseif arg == "--config-dir" && i < length(args)
            options[:config_dir] = args[i + 1]
            i += 2
        else
            println("Unknown argument: $(arg)")
            i += 1
        end
    end

    if isnothing(options[:mode])
        error(
            "--mode is required " *
            "(preliminary, final, or rqs_search)"
        )
    end

    return instance_name, options
end


function print_simulation_usage()
    println("""
    DeepSJRP - Benchmark & NN Policy Simulations

    Usage:
      julia run_simulations.jl <instance> --mode <mode> [options]

    Modes:
      preliminary   Search optimal benchmark parameters
                    (10K runs × 10K periods per config)
      final         High-precision simulations
                    (100K benchmark, configurable NN)
      rqs_search    Broad (R,Q,S) search: R=1..75,
                    coarse Q, 100 iterations per config

    Options:
      --mode <mode>               Required (preliminary|final|rqs_search)
      --nn-iterations <int>       NN runs in final mode (default: 100)
      --nn-method <method>        Force NN method: solve_inf or find_stationary
      --seed <int>                Random seed (default: 777)
      --config-dir <path>         Config directory (default: "configs")
      --help                      Show this help message

    Examples:
      julia run_simulations.jl 12D_CaseLLL --mode preliminary
      julia run_simulations.jl 12D_CaseLLL --mode final --nn-iterations 50
      julia run_simulations.jl 12D_CaseLLL --mode rqs_search
    """)
end


# =====================================================================
# Preliminary mode
# =====================================================================

function run_preliminary(instance_name::String, options)
    seed = options[:seed]
    config_dir = options[:config_dir]
    input_dir = options[:input_dir]
    output_dir = options[:output_dir]

    config_path = joinpath(
        config_dir, "hyperparams_$(instance_name).json"
    )
    if !isfile(config_path)
        error("Config not found: $(config_path)")
    end
    input_path = joinpath(
        input_dir, "$(instance_name).json"
    )

    # Load problem (no cost scaling for benchmarks)
    alg, _, _, _, _ = load_hyperparameters_json(config_path)
    impulse_prob, inventory_prob, _ = load_problem_json(
        input_path; cost_scaling=alg.cost_scaling
    )

    print_simulation_summary(
        impulse_prob, inventory_prob, instance_name;
        cost_scaling=alg.cost_scaling
    )

    println("\n" * "="^60)
    println("PRELIMINARY BENCHMARK SIMULATIONS")
    println("Instance: $(instance_name)")
    println("="^60)
    flush(stdout)

    results = Dict{Symbol, BenchmarkSearchResult}()

    # Check for MDP policy (2D instances only)
    dim = impulse_prob.dim
    mdp_path = joinpath(
        output_dir, "2d_mdp",
        "mdppolicy_$(instance_name).jld2"
    )
    has_mdp = dim == 2 && isfile(mdp_path)
    total_steps = has_mdp ? 6 : 5
    step = 0

    # 1. Individual (s,S) policy
    step += 1
    println("\n" * "="^50)
    println("$(step)/$(total_steps): Individual (s,S) policy search")
    println("="^50)
    results[:individual_sS] = search_individual_sS(
        inventory_prob;
        num_iterations=10_000,
        time_horizon=10_000,
        seed=seed
    )
    flush(stdout)

    # MDP optimal policy (2D only)
    if has_mdp
        step += 1
        println("\n" * "="^50)
        println(
            "$(step)/$(total_steps): " *
            "MDP optimal policy (2D)"
        )
        println("="^50)
        mdp_policy = load_mdp_policy_jld2(mdp_path)
        println("Loaded MDP policy from: $(mdp_path)")
        x_initial = zeros(Int, dim)
        Random.seed!(seed)
        mdp_cost, mdp_se, mdp_rd = MCS_DiscreteInventory(
            inventory_prob, x_initial, mdp_policy;
            num_iterations=10_000,
            time_horizon=10_000
        )
        results[:mdp] = BenchmarkSearchResult(
            mdp_policy, mdp_cost, mdp_se,
            mdp_rd[:ordering_frequency], nothing,
            Dict{Symbol, Any}()
        )
        @printf(
            "  MDP optimal: %.2f ± %.2f\n",
            mdp_cost, mdp_se
        )
        flush(stdout)
    end

    # (R,S) periodic review
    step += 1
    println("\n" * "="^50)
    println("$(step)/$(total_steps): (R,S) periodic review search")
    println("="^50)
    results[:periodic_RS] = search_periodic_RS(
        inventory_prob;
        num_iterations=10_000,
        time_horizon=10_000,
        seed=seed
    )
    flush(stdout)

    # (Q,S) aggregate demand (depends on best R)
    best_R = results[:periodic_RS].policy.R
    step += 1
    println("\n" * "="^50)
    println(
        "$(step)/$(total_steps): (Q,S) aggregate demand search " *
        "(using best R=$(best_R))"
    )
    println("="^50)
    results[:aggregate_QS] = search_aggregate_QS(
        inventory_prob;
        best_R=best_R,
        num_rq_samples=100_000,
        num_iterations=10_000,
        time_horizon=10_000,
        seed=seed
    )
    flush(stdout)

    # 4. (R,Q,S) hybrid (depends on best R and best Q)
    best_Q = results[:aggregate_QS].policy.Q
    weekly_demand = sum(
        mean.(inventory_prob.nb_demand)
    )

    # R range: best_R - 2 to best_R + 10 (skewed upward)
    # rqs_R_range = max(1, best_R - 2):(best_R + 10)
    rqs_R_range = union(1:100, [1000])

    # Q range: best_Q - 5 week to best_Q + 15 weeks (skewed upward)
    rqs_Q_start = max(1, floor(Int, best_Q - 5 * weekly_demand))
    rqs_Q_end = ceil(Int, best_Q + 15 * weekly_demand)
    
    # Adaptive Q step: target ~200 points
    Q_range_width = rqs_Q_end - rqs_Q_start
    target_points = 200
    Q_step = max(1, floor(Int, Q_range_width / target_points))
    rqs_Q_range = rqs_Q_start:Q_step:rqs_Q_end
    
    step += 1
    println("\n" * "="^50)
    println(
        "$(step)/$(total_steps): (R,Q,S) hybrid search " *
        "(best_R=$(best_R), best_Q=$(best_Q))"
    )
    println(
        "     R ∈ [$(first(rqs_R_range)), $(last(rqs_R_range))], " *
        "Q ∈ [$(rqs_Q_start), $(rqs_Q_end)] (step=$(Q_step))"
    )
    println("="^50)
    results[:hybrid_RQS] = search_hybrid_RQS(
        inventory_prob;
        R_range=rqs_R_range,
        Q_range=rqs_Q_range,
        num_rq_samples=100_000,
        num_iterations=10_000,
        time_horizon=10_000,
        seed=seed
    )
    flush(stdout)

    # Can-order policy
    step += 1
    println("\n" * "="^50)
    println("$(step)/$(total_steps): Can-order policy search")
    println("="^50)
    results[:can_order] = search_can_order(
        inventory_prob;
        num_iterations=10_000,
        time_horizon=10_000,
        seed=seed
    )
    flush(stdout)

    # Save benchmark parameters
    benchmark_path = joinpath(
        output_dir, "benchmark_params",
        "benchmark_params_$(instance_name).json"
    )
    save_benchmark_params_json(
        benchmark_path, results;
        instance_name=instance_name,
        iterations=10_000,
        time_horizon=10_000,
        seed=seed
    )

    # Update config with QS and can-order parameters
    println("\nUpdating config with optimal parameters...")
    update_config_benchmarks!(
        config_path,
        results[:aggregate_QS],
        results[:can_order]
    )

    # Print summary
    println("\n" * "="^60)
    println("PRELIMINARY RESULTS SUMMARY")
    println("="^60)
    for (name, res) in results
        @printf(
            "  %-16s: %10.2f ± %6.2f\n",
            name, res.cost, res.std_error
        )
    end
    println("="^60)
    println(
        "Results saved to: $(benchmark_path)"
    )
    println(
        "Config updated:   $(config_path)"
    )
    flush(stdout)

    return results
end


# =====================================================================
# Broad (R,Q,S) search mode
# =====================================================================

function run_rqs_search(instance_name::String, options)
    seed = options[:seed]
    config_dir = options[:config_dir]
    input_dir = options[:input_dir]
    output_dir = options[:output_dir]

    config_path = joinpath(
        config_dir, "hyperparams_$(instance_name).json"
    )
    if !isfile(config_path)
        error("Config not found: $(config_path)")
    end
    input_path = joinpath(
        input_dir, "$(instance_name).json"
    )

    # Check that preliminary results exist
    benchmark_path = joinpath(
        output_dir, "benchmark_params",
        "benchmark_params_$(instance_name).json"
    )
    bench_data = load_benchmark_params_json(benchmark_path)
    if isnothing(bench_data)
        error(
            "Benchmark params not found: " *
            "$(benchmark_path). " *
            "Run preliminary mode first."
        )
    end

    alg, _, _, _, _ = load_hyperparameters_json(config_path)
    _, inventory_prob, _ = load_problem_json(
        input_path; cost_scaling=alg.cost_scaling
    )

    # Extract best R and Q from preliminary results
    best_R = if haskey(bench_data, "periodic_RS")
        Int(bench_data["periodic_RS"]["best_R"])
    else
        error(
            "No periodic_RS in benchmark params. " *
            "Run preliminary first."
        )
    end
    best_Q = if haskey(bench_data, "aggregate_QS")
        Int(bench_data["aggregate_QS"]["best_Q"])
    else
        error(
            "No aggregate_QS in benchmark params. " *
            "Run preliminary first."
        )
    end

    # Wide range centered on best values
    weekly_demand = sum(mean.(inventory_prob.nb_demand))
    R_values = max(1, best_R - 10):1:(best_R + 20)
    Q_lo = max(1, floor(Int, best_Q - 5 * weekly_demand))
    Q_hi = ceil(Int, best_Q + 15 * weekly_demand)

    # Adaptive Q step: target ~100 points
    Q_range_width = Q_hi - Q_lo
    target_points = 100
    Q_step = max(1, floor(Int, Q_range_width / target_points))
    Q_values = Q_lo:Q_step:Q_hi

    println("\n" * "="^60)
    println("BROAD (R,Q,S) PARAMETER SEARCH")
    println("Instance:       $(instance_name)")
    println("Best R (prelim): $(best_R)")
    println("Best Q (prelim): $(best_Q)")
    println(
        "R range:        " *
        "$(first(R_values)):$(step(R_values)):" *
        "$(last(R_values)) ($(length(R_values)) values)"
    )
    println(
        "Q range:        " *
        "$(first(Q_values)):$(step(Q_values)):" *
        "$(last(Q_values)) ($(length(Q_values)) values)"
    )
    println("Configs:        $(length(R_values) * length(Q_values))")
    println("RQ samples:     10,000")
    println("="^60)
    flush(stdout)

    result = search_hybrid_RQS(
        inventory_prob;
        R_range=R_values,
        Q_range=Q_values,
        num_iterations=10_000,
        num_rq_samples=100_000,
        time_horizon=10_000,
        seed=seed
    )

    # Compare with existing RQS result
    old_cost = if haskey(bench_data, "hybrid_RQS")
        bench_data["hybrid_RQS"]["cost"]
    else
        Inf
    end

    println("\n" * "="^60)
    println("RQS SEARCH COMPLETE")
    @printf(
        "New best: R=%d, Q=%d, cost=%.2f ± %.2f\n",
        result.search_params[:best_R],
        result.search_params[:best_Q],
        result.cost,
        result.std_error
    )

    if result.cost < old_cost
        if old_cost < Inf
            @printf(
                "Improvement over previous: %.2f → %.2f\n",
                old_cost, result.cost
            )
        end
        # Update the hybrid_RQS entry in-place
        policy = result.policy::HybridRQSPolicy
        freq = result.ordering_frequency
        bench_data["hybrid_RQS"] = Dict{String, Any}(
            "cost" => result.cost,
            "std_error" => result.std_error,
            "ordering_frequency" => freq,
            "analytical_cost" => result.analytical_cost,
            "best_R" => policy.R,
            "best_Q" => policy.Q,
            "S_levels" => policy.S_levels,
            "average_weeks_between_orders" =>
                freq > 0 ? 1.0 / freq : Inf
        )
        bench_data["timestamp"] = string(Dates.now())
        open(benchmark_path, "w") do f
            JSON.print(f, bench_data, 2)
        end
        println(
            "Updated: $(benchmark_path)"
        )
    else
        @printf(
            "No improvement (existing: %.2f). Keeping current params.\n",
            old_cost
        )
    end
    println("="^60)
    flush(stdout)

    return Dict(:hybrid_RQS => result)
end


# =====================================================================
# Final mode
# =====================================================================

function run_final(instance_name::String, options)
    seed = options[:seed]
    nn_iterations = options[:nn_iterations]
    config_dir = options[:config_dir]
    input_dir = options[:input_dir]
    output_dir = options[:output_dir]
    policy_dir = options[:policy_dir]

    config_path = joinpath(
        config_dir, "hyperparams_$(instance_name).json"
    )
    input_path = joinpath(
        input_dir, "$(instance_name).json"
    )

    # Load problem
    alg, _, opt_params, _, benchmark_source = load_hyperparameters_json(config_path)
    impulse_prob, inventory_prob, _ = load_problem_json(
        input_path; cost_scaling=alg.cost_scaling
    )

    dim = impulse_prob.dim
    time_horizon = 10_000
    benchmark_iterations = 100_000

    print_simulation_summary(
        impulse_prob, inventory_prob, instance_name;
        cost_scaling=alg.cost_scaling
    )

    println("\n" * "="^60)
    println("FINAL HIGH-PRECISION SIMULATIONS")
    println("Instance: $(instance_name)")
    println(
        "Benchmarks: $(benchmark_iterations) runs × " *
        "$(time_horizon) periods"
    )
    println(
        "NN policy:  $(nn_iterations) runs × " *
        "$(time_horizon) periods"
    )
    println("="^60)

    # Load benchmark parameters
    benchmark_path = joinpath(
        output_dir, "benchmark_params",
        "benchmark_params_$(instance_name).json"
    )
    bench_data = load_benchmark_params_json(benchmark_path)
    if isnothing(bench_data)
        error(
            "Benchmark params not found: $(benchmark_path). " *
            "Run preliminary mode first."
        )
    end

    # Load NN results (pick best method, or use override)
    nn_result = nothing
    best_nn_cost = Inf
    best_method = :unknown
    nn_method_override = options[:nn_method]

    methods_to_check = if !isnothing(nn_method_override)
        [nn_method_override]
    else
        [:solve_inf, :find_stationary]
    end

    for method in methods_to_check
        result_path = joinpath(
            output_dir, "nn_training",
            "results_$(instance_name)_$(method).json"
        )
        data = load_results_json(result_path)
        if !isnothing(data) && data["best_cost"] < best_nn_cost
            best_nn_cost = data["best_cost"]
            nn_result = data
            best_method = method
        end
    end

    if !isnothing(nn_method_override) && isnothing(nn_result)
        error(
            "No NN results found for method " *
            "$(nn_method_override). Run training first."
        )
    end

    if isnothing(nn_result)
        println(
            "WARNING: No NN training results found. " *
            "Skipping NN evaluation."
        )
    end

    # Load NN weights if available
    nns = nothing
    if !isnothing(nn_result)
        bson_path = joinpath(
            policy_dir,
            "policies$(dim)D",
            "neural_networks_$(instance_name)" *
            "_$(best_method).bson"
        )
        if isfile(bson_path)
            nns = load_neural_networks(bson_path)
            println("Loaded NN from: $(bson_path)")
        else
            println(
                "WARNING: NN weights not found at " *
                "$(bson_path). Skipping NN evaluation."
            )
        end
    end

    x_initial = zeros(Int, dim)
    benchmark_results = Dict{Symbol, Dict{Symbol, Any}}()

    # Reconstruct and simulate each benchmark policy
    # 1. Individual (s,S)
    if haskey(bench_data, "individual_sS")
        bd = bench_data["individual_sS"]
        policy = IndividualSSPolicy(
            Int.(bd["s_levels"]),
            Int.(bd["S_levels"]),
            bd["best_alpha"]
        )
        println("\nSimulating Individual (s,S)...")
        Random.seed!(seed)
        cost, se, rd = MCS_DiscreteInventory(
            inventory_prob, x_initial, policy;
            num_iterations=benchmark_iterations,
            time_horizon=time_horizon
        )
        benchmark_results[:individual_sS] = Dict(
            :cost => cost, :std_error => se,
            :wall_time => rd[:wall_time],
            :cost_breakdown => Dict(
                :holding => rd[:mean_holding_costs],
                :backlog => rd[:mean_backlogging_costs],
                :variable => rd[:mean_variable_costs],
                :fixed => rd[:mean_fixed_costs]
            )
        )
        @printf(
            "  Individual (s,S): %.2f ± %.2f (%.1fs)\n",
            cost, se, rd[:wall_time]
        )
        flush(stdout)
    end

    # MDP optimal (2D only)
    mdp_path = joinpath(
        output_dir, "2d_mdp",
        "mdppolicy_$(instance_name).jld2"
    )
    if dim == 2 && isfile(mdp_path)
        mdp_policy = load_mdp_policy_jld2(mdp_path)
        println("\nSimulating MDP optimal...")
        Random.seed!(seed)
        cost, se, rd = MCS_DiscreteInventory(
            inventory_prob, x_initial, mdp_policy;
            num_iterations=benchmark_iterations,
            time_horizon=time_horizon
        )
        benchmark_results[:mdp] = Dict(
            :cost => cost, :std_error => se,
            :wall_time => rd[:wall_time],
            :cost_breakdown => Dict(
                :holding => rd[:mean_holding_costs],
                :backlog => rd[:mean_backlogging_costs],
                :variable => rd[:mean_variable_costs],
                :fixed => rd[:mean_fixed_costs]
            )
        )
        @printf(
            "  MDP optimal: %.2f ± %.2f (%.1fs)\n",
            cost, se, rd[:wall_time]
        )
        flush(stdout)
    elseif dim == 2
        println(
            "\nWARNING: MDP policy not found at " *
            "$(mdp_path). Skipping MDP evaluation."
        )
    end

    # (R,S)
    if haskey(bench_data, "periodic_RS")
        bd = bench_data["periodic_RS"]
        policy = PeriodicRSPolicy(
            Int.(bd["S_levels"]),
            Int(bd["best_R"])
        )
        println("Simulating (R,S)...")
        Random.seed!(seed)
        cost, se, rd = MCS_DiscreteInventory(
            inventory_prob, x_initial, policy;
            num_iterations=benchmark_iterations,
            time_horizon=time_horizon
        )
        benchmark_results[:periodic_RS] = Dict(
            :cost => cost, :std_error => se,
            :wall_time => rd[:wall_time],
            :cost_breakdown => Dict(
                :holding => rd[:mean_holding_costs],
                :backlog => rd[:mean_backlogging_costs],
                :variable => rd[:mean_variable_costs],
                :fixed => rd[:mean_fixed_costs]
            )
        )
        @printf(
            "  (R,S): %.2f ± %.2f (%.1fs)\n",
            cost, se, rd[:wall_time]
        )
        flush(stdout)
    end

    # 3. (Q,S)
    if haskey(bench_data, "aggregate_QS")
        bd = bench_data["aggregate_QS"]
        policy = AggregateDemandQSPolicy(
            Int.(bd["S_levels"]),
            Int(bd["best_Q"])
        )
        println("Simulating (Q,S)...")
        Random.seed!(seed)
        cost, se, rd = MCS_DiscreteInventory(
            inventory_prob, x_initial, policy;
            num_iterations=benchmark_iterations,
            time_horizon=time_horizon
        )
        benchmark_results[:aggregate_QS] = Dict(
            :cost => cost, :std_error => se,
            :wall_time => rd[:wall_time],
            :cost_breakdown => Dict(
                :holding => rd[:mean_holding_costs],
                :backlog => rd[:mean_backlogging_costs],
                :variable => rd[:mean_variable_costs],
                :fixed => rd[:mean_fixed_costs]
            )
        )
        @printf(
            "  (Q,S): %.2f ± %.2f (%.1fs)\n",
            cost, se, rd[:wall_time]
        )
        flush(stdout)
    end

    # 4. (R,Q,S)
    if haskey(bench_data, "hybrid_RQS")
        bd = bench_data["hybrid_RQS"]
        policy = HybridRQSPolicy(
            Int.(bd["S_levels"]),
            Int(bd["best_R"]),
            Int(bd["best_Q"])
        )
        println("Simulating (R,Q,S)...")
        Random.seed!(seed)
        cost, se, rd = MCS_DiscreteInventory(
            inventory_prob, x_initial, policy;
            num_iterations=benchmark_iterations,
            time_horizon=time_horizon
        )
        benchmark_results[:hybrid_RQS] = Dict(
            :cost => cost, :std_error => se,
            :wall_time => rd[:wall_time],
            :cost_breakdown => Dict(
                :holding => rd[:mean_holding_costs],
                :backlog => rd[:mean_backlogging_costs],
                :variable => rd[:mean_variable_costs],
                :fixed => rd[:mean_fixed_costs]
            )
        )
        @printf(
            "  (R,Q,S): %.2f ± %.2f (%.1fs)\n",
            cost, se, rd[:wall_time]
        )
        flush(stdout)
    end

    # 5. Can-order
    if haskey(bench_data, "can_order")
        bd = bench_data["can_order"]
        policy = CanOrderPolicy(
            Int.(bd["s_levels"]),
            Int.(bd["S_levels"]),
            bd["best_omega"],
            bd["best_alpha"]
        )
        println("Simulating Can-order...")
        Random.seed!(seed)
        cost, se, rd = MCS_DiscreteInventory(
            inventory_prob, x_initial, policy;
            num_iterations=benchmark_iterations,
            time_horizon=time_horizon
        )
        benchmark_results[:can_order] = Dict(
            :cost => cost, :std_error => se,
            :wall_time => rd[:wall_time],
            :cost_breakdown => Dict(
                :holding => rd[:mean_holding_costs],
                :backlog => rd[:mean_backlogging_costs],
                :variable => rd[:mean_variable_costs],
                :fixed => rd[:mean_fixed_costs]
            )
        )
        @printf(
            "  Can-order: %.2f ± %.2f (%.1fs)\n",
            cost, se, rd[:wall_time]
        )
        flush(stdout)
    end

    # 6. NN policy
    nn_final = Dict{Symbol, Any}()
    if !isnothing(nns) && !isnothing(nn_result)
        epsilon = nn_result["best_epsilon"]

        z_star = Int.(nn_result["z_star_int"])
        # S_start = Float64.(max.(alg.S_guess...))
        # println(
        #     "\nRecomputing z* via M_operator " *
        #     "($(best_method))..."
        # )
        # z_star_float, _ = M_operator(
        #     nns, zeros(dim), impulse_prob;
        #     S_start=S_start,
        #     method=best_method,
        #     lower=opt_params.lower_factor * S_start,
        #     upper=opt_params.upper_factor * S_start,
        #     δ_init=opt_params.start_factor
        # )
        # z_star = Int.(round.(z_star_float))

        # nn_policy = if dim == 2
        #     NeuralNetworkPolicy(
        #         nns, z_star, epsilon, impulse_prob,
        #         opt_params.nn_grid_min_inv,
        #         opt_params.nn_grid_max_inv
        #     )
        # else
        #     NeuralNetworkPolicy(
        #         nns, z_star, epsilon, impulse_prob
        #     )
        # end

        println("\nSimulating NN policy ($(best_method))...")
        Random.seed!(seed)
        cost, se, rd = MCS_DiscreteInventory(
            inventory_prob, x_initial, nn_policy;
            num_iterations=nn_iterations,
            time_horizon=time_horizon
        )

        nn_final = Dict{Symbol, Any}(
            :method => best_method,
            :z_star => z_star,
            :epsilon => epsilon,
            :cost => cost,
            :std_error => se,
            :simulation_time => rd[:wall_time],
            :cost_breakdown => Dict(
                :holding => rd[:mean_holding_costs],
                :backlog => rd[:mean_backlogging_costs],
                :variable => rd[:mean_variable_costs],
                :fixed => rd[:mean_fixed_costs]
            )
        )
        @printf("  NN (%s): %.2f ± %.2f (%.1fs)\n",
            best_method, cost, se, rd[:wall_time])
    end

    # Compute relative differences
    relative_diffs = Dict{Symbol, Tuple{Float64, Float64}}()
    if haskey(nn_final, :cost)
        nn_cost = nn_final[:cost]
        nn_se = nn_final[:std_error]

        println("\n" * "="^60)
        println("FINAL RESULTS")
        println("="^60)
        @printf(
            "\nNN policy cost: \$%.2f \\pm %.2f\\%%\$\n",
            nn_cost, nn_se / nn_cost * 100.0
        )

        println("\nRelative differences (benchmark - NN) / NN:")
        for (name, res) in benchmark_results
            pct_diff, pct_se = compute_relative_difference(
                nn_cost, nn_se, res[:cost], res[:std_error]
            )
            relative_diffs[name] = (pct_diff, pct_se)
            @printf(
                "  %-16s: %+.2f%% ± %.2f%%\n",
                name, pct_diff, pct_se
            )
        end
        println("="^60)

        # Print simulation timing summary
        println("\nSimulation times (wall):")
        nt = Threads.nthreads()
        for (name, res) in benchmark_results
            wt = res[:wall_time]
            @printf(
                "  %-16s: %6.1fs  (%d threads)\n",
                name, wt, nt
            )
        end
        if haskey(nn_final, :simulation_time)
            wt = nn_final[:simulation_time]
            @printf(
                "  %-16s: %6.1fs  (%d threads)\n",
                "NN", wt, nt
            )
        end

        # Print copy-pasteable LaTeX table row
        nn_se_pct = nn_se / nn_cost * 100.0
        bench_order = if dim == 2
            [
                :mdp, :periodic_RS, :aggregate_QS,
                :hybrid_RQS, :can_order, :individual_sS
            ]
        else
            [
                :periodic_RS, :aggregate_QS, :hybrid_RQS,
                :can_order, :individual_sS
            ]
        end
        latex_parts = String[]
        for name in bench_order
            if haskey(relative_diffs, name)
                pct_diff, pct_se = relative_diffs[name]
                push!(latex_parts, @sprintf(
                    "\$%.2f\\%% \\pm %.2f\\%%\$",
                    pct_diff, pct_se
                ))
            else
                push!(latex_parts, "---")
            end
        end
        latex_line = @sprintf(
            "\$%.2f \\pm %.2f\\%%\$",
            nn_cost, nn_se_pct
        ) * " & " * join(latex_parts, " & ") * " \\\\"

        println("\nLaTeX performance row:")
        println(latex_line)

        # Print copy-pasteable LaTeX hyperparameter row
        # Use hyperparameters from the best NN result file
        # (not config defaults, which may differ from grid
        # search winner)
        nn_hp = nn_result["hyperparameters"]
        hp_T = get(nn_hp, "T_horizon", alg.T_horizon)
        hp_λ = get(nn_hp, "lambda_rate", alg.λ_rate)
        hp_ν = get(nn_hp, "nu_radius", alg.ν_radius)
        hp_α = get(nn_hp, "alpha_factor", 0.0)
        hp_β_final = get(
            nn_hp, "beta_final", last(alg.β_penalty)
        )
        β_exp = Int(round(log10(hp_β_final)))
        Φ_name = uppercase(
            string(alg.S_distribution)[1:1]
        ) * string(alg.S_distribution)[2:end]
        hp_source = get(
            nn_result, "benchmark_source",
            benchmark_source
        )
        ez_source = hp_source == "QS" ?
            "\$(Q,S)\$" : "Can-order"
        ε_val = nn_result["best_epsilon"]
        opt_method_str = best_method == :solve_inf ?
            "Eq.~\\eqref{eq:argminimpulse}" :
            "Eq.~\\eqref{eq:stationary_conditionZ}"
        training_time_sec = get(
            nn_result, "training_time_seconds", 0.0
        )
        training_time_min = training_time_sec / 60.0
        nn_sim_time_sec = get(
            nn_final, :simulation_time, 0.0
        )
        nn_sim_time_min = nn_sim_time_sec / 60.0
        hp = [
            @sprintf("%.2f", hp_T),
            @sprintf("%.2f", hp_λ),
            @sprintf("%.2f", hp_ν),
            @sprintf("%.2f", hp_α),
            @sprintf("\$10^{%d}\$", β_exp),
            # Φ_name,
            ez_source,
            @sprintf("\$%.2f\$", ε_val),
            opt_method_str,
            @sprintf(
                "[%.2f, %.2f]",
                opt_params.lower_factor,
                opt_params.upper_factor
            ),
            # @sprintf("%.2f", opt_params.start_factor),
            @sprintf("%.2f", training_time_min),
            @sprintf("%.2f", nn_sim_time_min)
        ]
        hp_line = join(hp, " & ") * " \\\\"

        println("\nLaTeX hyperparameter row:")
        println(hp_line); flush(stdout)
    end

    # Save final comparison
    final_path = joinpath(
        output_dir, "final_results",
        "final_results_$(instance_name).json"
    )
    # Pass training hyperparameters from the best NN
    # result file (not config defaults)
    train_hp = if !isnothing(nn_result) &&
                  haskey(nn_result, "hyperparameters")
        Dict{String, Any}(nn_result["hyperparameters"])
    else
        Dict{String, Any}()
    end
    save_final_comparison_json(
        final_path, nn_final, benchmark_results,
        relative_diffs;
        instance_name=instance_name,
        benchmark_iterations=benchmark_iterations,
        nn_iterations=nn_iterations,
        time_horizon=time_horizon,
        seed=seed,
        training_hyperparameters=train_hp
    )

    return nn_final, benchmark_results, relative_diffs
end


# =====================================================================
# Main
# =====================================================================

function main()
    instance_name, options = parse_simulation_args(ARGS)

    println("Julia threads: $(Threads.nthreads())")
    flush(stdout)

    if options[:mode] == :preliminary
        run_preliminary(instance_name, options)
    elseif options[:mode] == :final
        run_final(instance_name, options)
    elseif options[:mode] == :rqs_search
        run_rqs_search(instance_name, options)
    else
        error(
            "Unknown mode: $(options[:mode]). " *
            "Use 'preliminary', 'final', " *
            "or 'rqs_search'."
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
