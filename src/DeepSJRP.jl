# DeepSJRP.jl - Main module for the Stochastic Joint Replenishment Problem solver

module DeepSJRP

# External dependencies
using Random
using LinearAlgebra
using Statistics
using Printf
using Dates
using Distributions
using JSON
using BSON
using JLD2

using CUDA
using cuDNN

using Flux
using Zygote
using ForwardDiff

using LBFGSB

using ProgressMeter
using TickTock

# Avoid BLAS × Julia thread oversubscription when MCS_DiscreteInventory
# (@threads) calls into Flux NN inference inside NeuralNetworkPolicy.
# OpenBLAS defaults to Sys.CPU_THREADS with no coordination with Threads
LinearAlgebra.BLAS.set_num_threads(1)

# Include source files in dependency order
include("types.jl")
include("device.jl")
include("neural_networks.jl")
include("zheng_federgruen.jl")
include("sampling.jl")
include("loss.jl")
include("training.jl")
include("optimization.jl")
include("simulation.jl")
include("nn_grid.jl")
include("benchmark_policies.jl")
include("io.jl")

# Export types
export ImpulseControlProblem, InventoryProblem
export NeuralNetworkParameters, AlgorithmHyperParameters
export OptimizationParameters
export NeuralNetworks

# Export policy types
export AbstractPolicy, AbstractStatelessPolicy
export AbstractStatefulPolicy
export IndividualSSPolicy, PeriodicRSPolicy
export AggregateDemandQSPolicy, HybridRQSPolicy
export CanOrderPolicy
export MDPPolicy
export NeuralNetworkPolicy
export NNGridPolicy
export compute_nn_value_grid, bake_grid_policy
export ZhengFedergruenParameters
export BenchmarkSearchResult

# Export device utilities
export set_device!, get_device, is_gpu
export to_device, to_cpu
export randn_device, rand_device, zeros_device, ones_device
export set_random_seed!, get_flux_device

# Export neural network functions
export create_flux_chain, create_neural_networks
export forward_pass, V_net, Z_net

# Export sampling functions
export sample_dB, sample_dU, sample_sde

# Export loss functions
export inventory_cost, ordering_cost
export create_cost_functions, lagrangian_loss

# Export training functions
export train_neural_networks
export save_neural_networks, load_neural_networks

# Export optimization functions
export M_operator, V_operator
export compute_order_up_to_vectors

# Export simulation functions
export compute_order_quantity!, reset_state!
export update_state_after_demand!
export MCS_DiscreteInventory
export evaluate_all_epsilons, run_full_evaluation

# Export Zheng-Federgruen functions
export expected_one_period_cost
export precompute_renewal_functions
export evaluate_average_cost
export find_optimal_sS_single_item
export compute_sS_policies

# Export benchmark policy functions
export search_individual_sS, search_periodic_RS
export search_aggregate_QS, search_hybrid_RQS
export search_can_order
export compute_relative_difference
export find_optimal_y_RS
export empirical_RQ_distribution
export analytical_cost_RS
export PrecomputedCycles
export precompute_QS_cycles, precompute_RQS_cycles
export find_optimal_y_joint, analytical_cost_joint

# Export I/O functions
export load_problem_json, load_hyperparameters_json
export save_method_results_json, load_results_json
export save_benchmark_params_json
export load_benchmark_params_json
export load_mdp_policy_jld2
export save_nn_grid_policy_jld2, load_nn_grid_policy_jld2
export save_final_comparison_json
export update_config_benchmarks!
export print_problem_summary, print_simulation_summary


"""
    run(instance_name; kwargs...)

Main entry point for training and evaluating the SJRP solver.

Arguments:
- instance_name: Name of the problem instance (without extension)

Keyword Arguments:
- input_dir: Directory containing input JSON files (default: "input")
- config_dir: Directory containing config files (default: "configs")
- output_dir: Directory for output files (default: "output")
- policy_dir: Directory for policy BSON files (default: "policies")
- T_override: Override T_horizon from command line
- beta_override: Override final beta from command line
- nu_override: Override nu_radius from command line
- alpha_override: Override alpha_factor from command line
- device: Compute device (:cpu or :gpu, default: :auto)
- seed: Random seed (default: 777)

Returns:
- results: Complete results dictionary
"""
function run(
    instance_name::String;
    input_dir::String="input",
    config_dir::String="configs",
    output_dir::String="output",
    policy_dir::String="policies",
    T_override::Union{Nothing,Float64}=nothing,
    beta_override::Union{Nothing,Float64}=nothing,
    nu_override::Union{Nothing,Float64}=nothing,
    alpha_override::Union{Nothing,Float64}=nothing,
    device::Symbol=:auto,
    seed::Int=777
)
    # Set up device
    if device == :auto
        set_device!(check_cuda_available() ? :gpu : :cpu)
    else
        set_device!(device)
    end

    # Set random seed
    set_random_seed!(seed)

    # Load problem
    input_path = joinpath(input_dir, "$(instance_name).json")

    # Load hyperparameters (instance-specific config)
    config_path = joinpath(
        config_dir, "hyperparams_$(instance_name).json"
    )
    if !isfile(config_path)
        error(
            "Config file not found: $(config_path). " *
            "Each instance needs its own hyperparams file."
        )
    end
    alg, nn_params, opt_params, S_input, benchmark_source,
        alpha_factor = load_hyperparameters_json(config_path)

    # Load problem with the cost scaling from config
    impulse_prob, inventory_prob, name =
        load_problem_json(
            input_path; cost_scaling=alg.cost_scaling
        )

    # Recompute αjs with actual μ (notebook: αjs = α_factor * μ / λ_rate)
    alg = AlgorithmHyperParameters(
        precision=alg.precision,
        batch_size=alg.batch_size,
        num_intervals=alg.num_intervals,
        T_horizon=alg.T_horizon,
        num_iterations=alg.num_iterations,
        learning_rates=alg.learning_rates,
        decay_steps=alg.decay_steps,
        β_penalty=alg.β_penalty,
        penalty_steps=alg.penalty_steps,
        λ_rate=alg.λ_rate,
        S_guess=alg.S_guess,
        ν_radius=alg.ν_radius,
        S_distribution=alg.S_distribution,
        αjs=alpha_factor * (impulse_prob.μ / alg.λ_rate),
        cost_scaling=alg.cost_scaling
    )

    # Apply command-line overrides
    if !isnothing(T_override)
        alg = AlgorithmHyperParameters(
            precision=alg.precision,
            batch_size=alg.batch_size,
            num_intervals=alg.num_intervals,
            T_horizon=T_override,

            num_iterations=alg.num_iterations,
            learning_rates=alg.learning_rates,
            decay_steps=alg.decay_steps,
            β_penalty=alg.β_penalty,
            penalty_steps=alg.penalty_steps,
            λ_rate=alg.λ_rate,
            S_guess=alg.S_guess,
            ν_radius=alg.ν_radius,
            S_distribution=alg.S_distribution,
            αjs=alg.αjs,
            cost_scaling=alg.cost_scaling
        )
    end

    if !isnothing(beta_override)
        new_beta = copy(alg.β_penalty)
        new_beta[end] = beta_override
        alg = AlgorithmHyperParameters(
            precision=alg.precision,
            batch_size=alg.batch_size,
            num_intervals=alg.num_intervals,
            T_horizon=alg.T_horizon,

            num_iterations=alg.num_iterations,
            learning_rates=alg.learning_rates,
            decay_steps=alg.decay_steps,
            β_penalty=new_beta,
            penalty_steps=alg.penalty_steps,
            λ_rate=alg.λ_rate,
            S_guess=alg.S_guess,
            ν_radius=alg.ν_radius,
            S_distribution=alg.S_distribution,
            αjs=alg.αjs,
            cost_scaling=alg.cost_scaling
        )
    end

    if !isnothing(nu_override)
        alg = AlgorithmHyperParameters(
            precision=alg.precision,
            batch_size=alg.batch_size,
            num_intervals=alg.num_intervals,
            T_horizon=alg.T_horizon,

            num_iterations=alg.num_iterations,
            learning_rates=alg.learning_rates,
            decay_steps=alg.decay_steps,
            β_penalty=alg.β_penalty,
            penalty_steps=alg.penalty_steps,
            λ_rate=alg.λ_rate,
            S_guess=alg.S_guess,
            ν_radius=nu_override,
            S_distribution=alg.S_distribution,
            αjs=alg.αjs,
            cost_scaling=alg.cost_scaling
        )
    end

    if !isnothing(alpha_override)
        new_αjs =
            alpha_override * (impulse_prob.μ / alg.λ_rate)
        alg = AlgorithmHyperParameters(
            precision=alg.precision,
            batch_size=alg.batch_size,
            num_intervals=alg.num_intervals,
            T_horizon=alg.T_horizon,

            num_iterations=alg.num_iterations,
            learning_rates=alg.learning_rates,
            decay_steps=alg.decay_steps,
            β_penalty=alg.β_penalty,
            penalty_steps=alg.penalty_steps,
            λ_rate=alg.λ_rate,
            S_guess=alg.S_guess,
            ν_radius=alg.ν_radius,
            S_distribution=alg.S_distribution,
            αjs=new_αjs,
            cost_scaling=alg.cost_scaling
        )
    end

    # Print problem summary
    println("\nBenchmark source: $(benchmark_source)")
    print_problem_summary(
        impulse_prob, inventory_prob, instance_name;
        cost_scaling=alg.cost_scaling
    )

    # Print training hyperparameters
    println("\n" * "="^60)
    println("TRAINING HYPERPARAMETERS")
    println("="^60)
    println("Benchmark source: $(benchmark_source)")
    println("Precision:        $(alg.precision)")
    println("Batch size:       $(alg.batch_size)")
    println("Num intervals:    $(alg.num_intervals)")
    println("T horizon:        $(alg.T_horizon)")
    println("Num iterations:   $(alg.num_iterations)")
    println("Learning rates:   $(alg.learning_rates)")
    println("LR steps:         $(alg.decay_steps)")
    println("Beta penalty:     $(alg.β_penalty)")
    println("Penalty steps:    $(alg.penalty_steps)")
    println("Lambda rate:      $(alg.λ_rate)")
    println("S guess:          $(alg.S_guess)")
    println("Nu radius:        $(alg.ν_radius)")
    println("S distribution:   $(alg.S_distribution)")
    println("Alpha factor:     $(alpha_factor)")
    println("Alpha js:         $(alg.αjs)")
    println("Cost scaling:     $(alg.cost_scaling)")
    println("\nNeural network architecture:")
    println("  Hidden layers:  $(nn_params.num_hidden_layers)")
    println("  Width:          $(nn_params.nn_width)")
    println("  Activation:     $(nn_params.activation_name)")
    println("\nOptimization parameters:")
    println("  Start factor:   $(opt_params.start_factor)")
    println("  Lower factor:   $(opt_params.lower_factor)")
    println("  Upper factor:   $(opt_params.upper_factor)")
    println("  Sim runs:       $(opt_params.num_simulation_runs)")
    println("  Sim horizon:    $(opt_params.simulation_horizon)")
    println("  Epsilons:       $(opt_params.epsilon_factors)")
    println("="^60)
    flush(stdout)

    # Create neural networks
    device_fn = get_flux_device()
    nns = create_neural_networks(
        impulse_prob.dim, nn_params, device_fn
    )

    println("\nGradient network G (∇V):")
    nns.G |> display
    println("\nValue network H (V):")
    nns.H |> display
    println()
    flush(stdout)

    # Train neural networks
    println("\n" * "="^60)
    println("Starting neural network training...")
    println("="^60 * "\n")

    nns, training_time =
        train_neural_networks(nns, impulse_prob, alg)

    # Move networks to CPU for optimization, simulation, and saving
    nns = Flux.cpu(nns)

    # Run full evaluation
    println("\n" * "="^60)
    println("Running policy evaluation...")
    println("="^60 * "\n")

    S_start = max.(alg.S_guess...)
    results = run_full_evaluation(
        inventory_prob, nns, impulse_prob,
        opt_params;
        S_start=S_start
    )

    # Compute V(0) estimate
    dim = impulse_prob.dim
    V0_estimate = V_net(nns, zeros(dim)) / alg.cost_scaling

    # Save per-method results and neural networks
    for (method, method_results) in results
        method_str = string(method)

        output_path = joinpath(
            output_dir, "nn_training",
            "results_$(instance_name)_$(method_str).json"
        )

        saved = save_method_results_json(
            output_path, method, method_results,
            instance_name, alg;
            μ=impulse_prob.μ,
            V0_estimate=V0_estimate,
            benchmark_source=benchmark_source,
            training_time=training_time
        )

        if saved
            bson_path = joinpath(
                policy_dir,
                "policies$(dim)D",
                "neural_networks_$(instance_name)" *
                "_$(method_str).bson"
            )
            save_neural_networks(nns, bson_path)

            if haskey(method_results, :nn_grid_policy)
                grid_path = joinpath(
                    policy_dir,
                    "policies$(dim)D",
                    "nn_grid_$(instance_name)" *
                    "_$(method_str).jld2"
                )
                save_nn_grid_policy_jld2(
                    grid_path,
                    method_results[:nn_grid_policy]
                )
                println("Grid saved to: $(grid_path)")
            end
        end
    end

    return results
end

end # module
