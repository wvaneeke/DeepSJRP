# sampling.jl - SDE sampling functions (Subroutine 1: Euler-Maruyama discretization)


"""
    sample_dB(d, num_samples, num_intervals, dt, σ_matrix, _type)

Generate sample increments of the Wiener process.

Arguments:
- d: Dimension
- num_samples: Number of sample paths (K)
- num_intervals: Number of time intervals (N)
- dt: Time step size
- σ_matrix: Cholesky factor of covariance matrix Σ
- _type: Float type

Returns:
- dB_sample: Brownian increments, shape (d, N, K)
"""
function sample_dB(
    d::Int,
    num_samples::Int,
    num_intervals::Int,
    dt::Real,
    σ_matrix::AbstractMatrix,
    _type::Type
)
    # Sample standard normal increments
    dW_sample = randn_device(_type, d, num_intervals * num_samples)
    
    # Apply covariance structure: σ * dW
    dW_sample = σ_matrix * dW_sample
    
    # Reshape to (d, N, K)
    dW_sample = reshape(dW_sample, d, num_intervals, num_samples)
    
    # Scale by √dt
    return sqrt(dt) .* dW_sample
end


"""
    sample_dU(d, num_samples, num_intervals, dt, λ, S_guess, ν, αs, dist_type, _type)

Generate sample Poisson increments (order-up-to jumps).

Arguments:
- d: Dimension
- num_samples: Number of sample paths (K)
- num_intervals: Number of time intervals (N)
- dt: Time step size
- λ: Poisson rate
- S_guess: Order-up-to vector guess (single vector for :normal/:lognormal)
- ν: Radius/variance parameter
- αs: Additional jump factors (for α_factor > 0)
- dist_type: Distribution type (:lognormal, :normal, or :uniform)
- _type: Float type

Returns:
- poisson_sample: Indicator of jump occurrences, shape (d, N, K)
- jump_sizes: Order-up-to vectors at jump times, shape (d, N, K)
"""
function sample_dU(
    d::Int,
    num_samples::Int,
    num_intervals::Int,
    dt::Real,
    λ::Real,
    S_guess::Vector{Vector{Float64}},
    ν::Real,
    αs::AbstractVector,
    dist_type::Symbol,
    _type::Type
)
    # Sample Poisson jumps via Bernoulli approximation
    poisson_sample = rand_device(_type, 1, num_intervals, num_samples) .<= λ * dt
    
    # Replicate across dimensions
    poisson_sample = repeat(poisson_sample, d, 1, 1)
    
    # Initialize jump sizes array
    jump_sizes = zeros_device(_type, d, num_intervals, num_samples)
    
    if dist_type == :uniform
        # For uniform distribution, S_guess must provide lower and upper bounds
        if length(S_guess) != 2
            error("For :uniform dist_type, S_guess must contain two arrays for lower and upper bounds.")
        end
        S_l = to_device(convert.(eltype(jump_sizes), S_guess[1]))
        S_u = to_device(convert.(eltype(jump_sizes), S_guess[2]))
        
        # Generate uniform jump sizes in [S_l, S_u] componentwise
        U = rand_device(_type, d, num_intervals, num_samples)
        @. jump_sizes = (S_u - S_l) * U + S_l
        
    elseif dist_type == :normal
        # For normal distribution, S_guess provides a single mean vector
        if length(S_guess) != 1
            error("For :normal dist_type, S_guess must contain exactly one array for the mean vector.")
        end
        S = to_device(convert.(eltype(jump_sizes), S_guess[1]))
        
        mu_vec = S
        sigma_vec = ν .* S
        
        normals = randn_device(_type, d, num_intervals, num_samples)
        
        # Affine transformation: μ + σ * Z
        @. jump_sizes = mu_vec + sigma_vec * normals
        
    elseif dist_type == :lognormal
        # For lognormal distribution, S_guess provides the desired mean
        if length(S_guess) != 1
            error("For :lognormal dist_type, S_guess must contain exactly one array.")
        end
        S = to_device(convert.(eltype(jump_sizes), S_guess[1]))
        
        # Compute lognormal parameters from desired mean and variance
        var_coef = (ν * S) .^ 2
        sigma_vec = sqrt.(log.(1 .+ var_coef ./ (S .^ 2)))
        mu_vec = log.(S) .- 0.5 .* (sigma_vec .^ 2)
        
        normals = randn_device(_type, d, num_intervals, num_samples)
        
        # Generate lognormal samples: exp(μ_log + σ_log * Z)
        @. jump_sizes = exp(mu_vec + sigma_vec * normals)
        
    else
        error("Unsupported dist_type: $(dist_type). Use :lognormal, :normal, or :uniform.")
    end
    
    return poisson_sample, jump_sizes
end


"""
    sample_sde(d, num_samples, num_intervals, x0, T, μ, σ_matrix, λ, S_guess, ν, αs, dist_type, _type)

Sample discretized paths of the reference process using Euler-Maruyama scheme.
This implements Subroutine 1 from the paper.

Arguments:
- d: Dimension
- num_samples: Number of sample paths (K)
- num_intervals: Number of time intervals (N)
- x0: Initial state, vector (d,) or matrix (d, K) for steady-state sampling
- T: Time horizon
- μ: Drift vector
- σ_matrix: Cholesky factor of covariance matrix
- λ: Poisson rate
- S_guess: Order-up-to vector guess
- ν: Radius/variance parameter
- αs: Additional jump factors
- dist_type: Distribution type
- _type: Float type

Returns:
- X_sample: Sample paths, shape (d, N+1, K)
- dB_sample: Brownian increments, shape (d, N, K)
- dU_sample: Jump increments, shape (d, N, K)
"""
function sample_sde(
    d::Int,
    num_samples::Int,
    num_intervals::Int,
    x0::AbstractArray,
    T::Real,
    μ::AbstractVector,
    σ_matrix::AbstractMatrix,
    λ::Real,
    S_guess::Vector{Vector{Float64}},
    ν::Real,
    αs::AbstractVector,
    dist_type::Symbol,
    _type::Type
)
    dt = T / num_intervals

    # Convert σ_matrix to device (matches μ_device conversion below)
    σ_matrix_device = to_device(convert.(_type, σ_matrix))

    # Sample Brownian increments
    dB_sample = sample_dB(d, num_samples, num_intervals, dt, σ_matrix_device, _type)
    
    # Sample Poisson jumps
    poisson_indicator, order_upto = sample_dU(
        d, num_samples, num_intervals, dt, λ, S_guess, ν, αs, dist_type, _type
    )
    
    # Initialize arrays
    dU_sample = zeros_device(_type, d, num_intervals, num_samples)
    X_sample = zeros_device(_type, d, num_intervals + 1, num_samples)
    
    # Convert and transfer initial state
    x0_device = to_device(convert.(_type, x0))
    
    # Set initial state for all paths
    X_sample[:, 1, :] .= x0_device
    
    # Sample additional exponential jump sizes if α > 0
    if any(!=(0.0), αs)
        U = rand_device(_type, 1, num_intervals, num_samples)
        exp_mean = to_device(convert.(_type, αs))
        αs_sample = -exp_mean .* log.(U)
    else
        αs_sample = zeros_device(_type, d, num_intervals, num_samples)
    end
    
    # Convert μ to device
    μ_device = to_device(convert.(_type, μ))
    
    # Euler-Maruyama integration
    @inbounds for n in 1:num_intervals
        # Jump size: max(Z - X, α_exponential) if Poisson event occurs
        dU_sample[:, n, :] .= poisson_indicator[:, n, :] .* max.(
            order_upto[:, n, :] .- X_sample[:, n, :],
            αs_sample[:, n, :]
        )
        
        # State update: X_{n+1} = X_n - μ*dt + dB_n + dU_n
        X_sample[:, n+1, :] .= X_sample[:, n, :] .- μ_device .* dt .+ 
                                dB_sample[:, n, :] .+ dU_sample[:, n, :]
    end
    
    return X_sample, dB_sample, dU_sample
end
