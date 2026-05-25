#!/usr/bin/env julia
#
# run_training.jl - Main entry point for the DeepSJRP solver
#
# Usage:
#   julia run_training.jl <instance_name> [options]
#
# Options:
#   --T <value>       Override T_horizon
#   --beta <value>    Override final beta penalty
#   --nu <value>      Override nu_radius
#   --alpha <value>   Override alpha_factor
#   --device <cpu|gpu> Set compute device (default: auto)
#   --seed <value>    Random seed (default: 777)
#   --help            Show this help message
#
# Examples:
#   julia run_training.jl 12D_CaseLLL
#   julia run_training.jl 12D_CaseLLL --T 0.05 --beta 1e7
#   julia run_training.jl 50D_CaseLL --device gpu --nu 0.4

using Pkg
Pkg.activate(@__DIR__)

# Add src to load path
push!(LOAD_PATH, joinpath(@__DIR__, "src"))

using DeepSJRP


"""
    parse_args(args)

Parse command-line arguments.
"""
function parse_args(args)
    if isempty(args) || "--help" in args
        print_usage()
        exit(0)
    end
    
    instance_name = args[1]
    
    options = Dict{Symbol, Any}(
        :T_override => nothing,
        :beta_override => nothing,
        :nu_override => nothing,
        :alpha_override => nothing,
        :device => :auto,
        :seed => 777
    )
    
    i = 2
    while i <= length(args)
        arg = args[i]
        
        if arg == "--T" && i < length(args)
            options[:T_override] = parse(Float64, args[i+1])
            i += 2
        elseif arg == "--beta" && i < length(args)
            options[:beta_override] = parse(Float64, args[i+1])
            i += 2
        elseif arg == "--nu" && i < length(args)
            options[:nu_override] = parse(Float64, args[i+1])
            i += 2
        elseif arg == "--alpha" && i < length(args)
            options[:alpha_override] = parse(Float64, args[i+1])
            i += 2
        elseif arg == "--device" && i < length(args)
            options[:device] = Symbol(args[i+1])
            i += 2
        elseif arg == "--seed" && i < length(args)
            options[:seed] = parse(Int, args[i+1])
            i += 2
        else
            println("Unknown argument: $(arg)")
            i += 1
        end
    end
    
    return instance_name, options
end


"""
    print_usage()

Print usage information.
"""
function print_usage()
    println("""
    DeepSJRP - Deep Learning Solver for the Stochastic Joint Replenishment Problem
    
    Usage:
      julia run_training.jl <instance_name> [options]
    
    Arguments:
      instance_name    Name of the problem instance (without .json extension)
    
    Options:
      --T <value>       Override T_horizon parameter
      --beta <value>    Override final beta penalty parameter
      --nu <value>      Override nu_radius parameter
      --alpha <value>   Override alpha_factor parameter
      --device <cpu|gpu> Set compute device (default: auto-detect)
      --seed <value>    Random seed (default: 777)
      --help            Show this help message
    
    Examples:
      julia run_training.jl 12D_CaseLLL
      julia run_training.jl 12D_CaseLLL --T 0.05 --beta 1e7
      julia run_training.jl 50D_CaseLL --device gpu --nu 0.4
    
    Input files should be placed in the 'input/' directory as JSON files.
    Hyperparameter configs go in 'config/' as 'hyperparams_<instance_name>.json'.
    Results are saved to 'output/nn_training/' directory.
    Trained neural networks are saved to 'policies/' directory.
    """)
end


"""
    main()

Main entry point.
"""
function main()
    instance_name, options = parse_args(ARGS)
    
    println("\n" * "="^60)
    println("DeepSJRP - Stochastic Joint Replenishment Problem Solver")
    println("="^60)
    println("Instance: $(instance_name)")
    println("Options: $(options)")
    println("="^60 * "\n")
    flush(stdout)

    # Run the solver
    results = DeepSJRP.run(
        instance_name;
        T_override=options[:T_override],
        beta_override=options[:beta_override],
        nu_override=options[:nu_override],
        alpha_override=options[:alpha_override],
        device=options[:device],
        seed=options[:seed]
    )
    
    # Print final summary
    println("\n" * "="^60)
    println("FINAL RESULTS SUMMARY")
    println("="^60)
    
    for (method, method_results) in results
        println("\nMethod: $(method)")
        println("  Order-up-to vector: $(method_results[:z_star_int])")
        println("  Best epsilon: $(method_results[:best_epsilon])")
        println("  Best cost: $(method_results[:best_cost]) ± $(method_results[:best_std_error])")
    end
    
    println("\n" * "="^60)
    println("Complete! Per-method results saved to:")
    for method in keys(results)
        method_str = string(method)
        println(
            "  output/nn_training/results_" *
            "$(instance_name)_$(method_str).json"
        )
    end
    println("="^60 * "\n")
    flush(stdout)

    return results
end


# Run main if this is the entry point
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
