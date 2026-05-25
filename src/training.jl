# training.jl - Training loop implementation (Algorithm 2)

using Flux
using Zygote
using TickTock
using Statistics


"""
    get_current_lr(iteration, learning_rates, decay_steps)

Get the current learning rate based on iteration and schedule.
"""
function get_current_lr(iteration::Int, learning_rates::Vector, decay_steps::Vector)
    idx = findlast(s -> iteration >= s, decay_steps)
    return isnothing(idx) ? learning_rates[1] : learning_rates[idx]
end


"""
    get_current_beta(iteration, β_penalty, penalty_steps)

Get the current penalty parameter based on iteration and schedule.
"""
function get_current_beta(iteration::Int, β_penalty::Vector, penalty_steps::Vector)
    idx = findlast(s -> iteration >= s, penalty_steps)
    return isnothing(idx) ? β_penalty[1] : β_penalty[idx]
end



"""
    train_neural_networks(nns, impulse_prob, alg, nn_params; kwargs...)

Main training loop implementing Algorithm 2 from the paper.

Arguments:
- nns: Neural networks to train
- impulse_prob: ImpulseControlProblem instance
- alg: AlgorithmHyperParameters
- nn_params: NeuralNetworkParameters

Returns:
- nns: Trained neural networks
- elapsed_total: Total training time in seconds
"""
function train_neural_networks(
    nns::NeuralNetworks,
    impulse_prob::ImpulseControlProblem,
    alg::AlgorithmHyperParameters;
    initial_state::Vector{Float64}=zeros(impulse_prob.dim),
    verbose_rate::Int=1000
)
    # Extract parameters
    d = impulse_prob.dim
    _type = alg.precision
    _convert = x -> convert.(_type, x)
    device_fn = get_flux_device()

    T = alg.T_horizon |> _convert
    K = alg.batch_size
    N = alg.num_intervals
    M = alg.num_iterations

    learning_rates = alg.learning_rates |> _convert
    decay_steps = alg.decay_steps
    penalty_steps = alg.penalty_steps
    β_penalty = alg.β_penalty |> _convert

    λ = alg.λ_rate |> _convert
    S_guess = alg.S_guess
    ν = alg.ν_radius |> _convert
    αs = alg.αjs
    dist_type = alg.S_distribution
    κ = alg.cost_scaling

    # Problem-specific data
    μ = impulse_prob.μ |> _convert |> to_device
    Σ = impulse_prob.Σ
    σ_matrix = cholesky(Σ).L |> _convert |> to_device

    r = impulse_prob.interest_rate |> _convert
    h = impulse_prob.holding_costs |> _convert |> to_device
    p = impulse_prob.penalty_costs |> _convert |> to_device
    ci = impulse_prob.variable_costs |> _convert |> to_device
    c0 = impulse_prob.fixed_cost |> _convert

    # Create cost functions
    f, c = create_cost_functions(h, p, ci, c0)

    # Initialize optimizer
    opt_state = Flux.setup(Flux.Adam(learning_rates[1]), nns)

    # Steady-state sampler: (d, K) matrix — each column is a path's
    # initial state. Broadcasts initial_state across all K paths.
    x0_col = convert.(_type, initial_state)
    current_x0 = repeat(reshape(x0_col, :, 1), 1, K)

    println("Starting training with steady-state sampler:")
    tick()

    # Main training loop (Algorithm 2)
    for m in 1:M
        # Step 1: Update learning rate and penalty schedules
        current_lr = get_current_lr(m, learning_rates, decay_steps)
        current_beta = get_current_beta(m, β_penalty, penalty_steps)
        Flux.adjust!(opt_state, current_lr)

        if m in penalty_steps
            println("-----------------------------------------")
            println(
                "Epoch: $(m); Penalty β updated " *
                "to: $(current_beta)"
            )
            println("-----------------------------------------")
            flush(stdout)
        end

        # Step 2: Sample training paths (Subroutine 1)
        sim_time = @elapsed begin
            X_sample, dB_sample, dU_sample = sample_sde(
                d, K, N, current_x0, T, μ, σ_matrix,
                λ, S_guess, ν, αs, dist_type, _type
            )
            # Steady-state: ALL K terminal states → next initial
            current_x0 = to_cpu(X_sample[:, end, :])
        end

        # Step 3: Compute metrics BEFORE gradient update (testmode)
        # Single forward_pass for both training_obj and α_train
        Flux.testmode!(nns)

        val_time = @elapsed begin
            V0_current = V_net(
                nns, zeros_device(_type, d)
            ) / κ

            H_x0, penalties = forward_pass(
                nns, X_sample, dB_sample, dU_sample,
                T, N, r, f, c, device_fn, _type
            )
            training_obj = sum(H_x0) / K
            α_train = 100.0 * (
                1.0 - count(x -> x > 0, penalties) / K
            )
        end

        # Step 4: Gradient update (trainmode)
        Flux.trainmode!(nns)

        grad_time = @elapsed begin
            loss_val, grads = Flux.withgradient(nns) do model
                lagrangian_loss(
                    model, X_sample, dB_sample, dU_sample,
                    current_beta, K, T, N, r, f, c,
                    device_fn, _type;
                    mode=:training
                )
            end
            Flux.update!(opt_state, nns, grads[1])
        end

        # Verbose output
        if m == 1 || m % verbose_rate == 0
            elapsed = peektimer()
            println(
                "Epoch: $(m); Computation time: " *
                "$(elapsed); Current value of V(0): " *
                "$(V0_current)"
            )
            println(
                "Lagrangian objective: $(loss_val); " *
                "Average H(x0): $(training_obj); " *
                "Quantile-hedge α(β): $(α_train)"
            )
            println(
                "Simulation time per epoch: " *
                "$(sim_time); Gradient computation " *
                "time: $(grad_time); Testing time: " *
                "$(val_time)"
            )
            flush(stdout)
        end
    end

    elapsed_total = tok()
    println(
        "\nTraining completed in " *
        "$(elapsed_total) seconds"
    )
    println(
        "Final V(0): " *
        "$(V_net(nns, zeros_device(_type, d)) / κ)"
    )
    flush(stdout)

    return nns, elapsed_total
end


"""
    save_neural_networks(nns, filepath)

Save trained neural networks to BSON file.
"""
function save_neural_networks(nns::NeuralNetworks, filepath::String)
    mkpath(dirname(filepath))
    BSON.@save filepath nns
    println("Neural networks saved to: $(filepath)")
end


"""
    load_neural_networks(filepath)

Load neural networks from BSON file.
"""
function load_neural_networks(filepath::String)
    BSON.@load filepath nns
    return nns
end
