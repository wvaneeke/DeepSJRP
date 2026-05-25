# neural_networks.jl - Neural network creation and functor definitions

using Flux


"""
    create_flux_chain(input_dim, output_dim; kwargs...)

Create a fully-connected neural network chain with the specified architecture.
"""
function create_flux_chain(
    input_dim::Int,
    output_dim::Int;
    num_hidden::Int=3,
    hidden_size::Int=1000,
    activation=elu,
    output_activation=identity,
    bias::Bool=true
)
    # Build hidden layers
    hidden_layers = [
        Dense(hidden_size, hidden_size, activation; init=Flux.kaiming_uniform(gain=√2))
        for _ in 2:num_hidden
    ]

    return Chain(
        Dense(input_dim, hidden_size, activation; init=Flux.kaiming_uniform(gain=√2)),
        hidden_layers...,
        Dense(hidden_size, output_dim, output_activation; init=Flux.kaiming_uniform(gain=√2), bias=bias)
    )
end


"""
    NeuralNetworks

Struct containing the gradient network G and value function network H.
Implements the functor interface for computing the loss components.
"""
struct NeuralNetworks{G,H}
    G::G  # Gradient network: ∇V
    H::H  # Value function network: V
end

Flux.@layer NeuralNetworks


"""
    create_neural_networks(dim, nn_params, device_fn)

Create the neural network pair (G, H) for the impulse control problem.
"""
function create_neural_networks(dim::Int, nn_params::NeuralNetworkParameters, device_fn)
    activation = eval(nn_params.activation_name)
    
    # Gradient network: maps R^d → R^d
    G = create_flux_chain(
        dim, dim;
        num_hidden=nn_params.num_hidden_layers,
        hidden_size=nn_params.nn_width,
        activation=activation,
        bias=true
    ) |> device_fn
    
    # Value function network: maps R^d → R
    H = create_flux_chain(
        dim, 1;
        num_hidden=nn_params.num_hidden_layers,
        hidden_size=nn_params.nn_width,
        activation=activation,
        bias=true
    ) |> device_fn
    
    return NeuralNetworks(G, H)
end


"""
    (nns::NeuralNetworks)(Xs, dB, dU, T, N, r, f, c, device_fn, _type)

Forward pass computing the loss components for the Lagrangian.

Arguments:
- Xs: Sample paths of the inventory state process, shape (d, N+1, K)
- dB: Brownian increments, shape (d, N, K)
- dU: Compound Poisson increments, shape (d, N, K)
- T: Time horizon
- N: Number of time intervals
- r: Interest rate
- f: Inventory cost function
- c: Ordering cost function
- device_fn: Device transfer function
- _type: Float type (Float32 or Float64)

Returns:
- H_x0: Value function at initial states, shape (1, K)
- penalties: Violation penalties, shape (1, K)
"""
function forward_pass(
    nns::NeuralNetworks,
    Xs::AbstractArray,
    dB::AbstractArray,
    dU::AbstractArray,
    T::Real,
    N::Int,
    r::Real,
    f::Function,
    c::Function,
    device_fn,
    _type::Type
)
    dt = T / N
    
    # Time indices and discount factors
    tns = reshape(collect(1:N), 1, N)
    tns = convert.(eltype(Xs), tns)
    tns = device_fn(tns)
    exponentials = exp.(-r .* tns .* dt)
    
    # States at times t_0, ..., t_{N-1}
    x_tn = Xs[:, 1:end-1, :]
    
    # Compute loss components
    # ∫₀ᵀ e^{-rt} ∇V(X̃(t))ᵀ σ dB(t)
    loss_dB = sum(exponentials .* sum(nns.G(x_tn) .* dB, dims=1), dims=2)
    
    # ∫₀ᵀ e^{-rt} f(X̃(t)) dt
    loss_df = sum(exponentials .* f(x_tn) .* dt, dims=2)
    
    # Σⱼ e^{-rτⱼ} c(Yⱼ)
    loss_dU = sum(exponentials .* c(dU), dims=2)
    
    # Remove singleton dimensions
    loss_dB = dropdims(loss_dB, dims=2)
    loss_df = dropdims(loss_df, dims=2)
    loss_dU = dropdims(loss_dU, dims=2)
    
    # Value function at initial and terminal states
    H_x0 = nns.H(Xs[:, 1, :])
    H_xT = nns.H(Xs[:, end, :])
    
    # Penalty: max(V(x₀) - e^{-rT}V(X̃(T)) + stochastic integral - costs, 0)
    penalties = max.(H_x0 .- exp(-r * T) .* H_xT .+ loss_dB .- loss_df .- loss_dU, 0)
    
    # Device verification (runs once, ignored by Zygote during backward pass)
    Zygote.ignore() do
        verify_devices(
            x_tn=x_tn, dB=dB, dU=dU,
            exponentials=exponentials,
            loss_dB=loss_dB, loss_df=loss_df, loss_dU=loss_dU,
            H_x0=H_x0, H_xT=H_xT, penalties=penalties
        )
    end

    return H_x0, penalties
end


"""
    V_net(nns, x)

Evaluate the value function network at state x.
"""
function V_net(nns::NeuralNetworks, x::AbstractVector)
    return nns.H(x) |> Flux.cpu |> only
end


"""
    Z_net(nns, x)

Evaluate the gradient network at state x.
"""
function Z_net(nns::NeuralNetworks, x::AbstractVector)
    return nns.G(x)
end
