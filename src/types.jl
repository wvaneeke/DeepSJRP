# types.jl - Struct definitions for the SJRP solver

"""
    ImpulseControlProblem

Represents the continuous-time impulse control approximation of the inventory problem.
"""
struct ImpulseControlProblem
    dim::Int
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    interest_rate::Float64
    holding_costs::Vector{Float64}
    penalty_costs::Vector{Float64}
    variable_costs::Vector{Float64}
    fixed_cost::Float64

    function ImpulseControlProblem(
        dim::Int,
        μ::Vector{Float64},
        Σ::Matrix{Float64},
        interest_rate::Float64,
        holding_costs::Vector{Float64},
        penalty_costs::Vector{Float64},
        variable_costs::Vector{Float64},
        fixed_cost::Float64
    )
        length(μ) == dim || error("Dimension of μ must be equal to dim")
        size(Σ) == (dim, dim) || error("Dimensions of Σ must be dim × dim")
        length(holding_costs) == dim || error("Dimension of holding_costs must be equal to dim")
        length(penalty_costs) == dim || error("Dimension of penalty_costs must be equal to dim")
        length(variable_costs) == dim || error("Dimension of variable_costs must be equal to dim")

        new(dim, μ, Σ, interest_rate, holding_costs, penalty_costs, variable_costs, fixed_cost)
    end
end


"""
    InventoryProblem

Represents the discrete-time inventory control problem for performance simulation.
"""
struct InventoryProblem
    dim::Int
    variable_costs::Vector{Float64}
    holding_costs::Vector{Float64}
    backlog_costs::Vector{Float64}
    fixed_cost::Float64
    nb_demand::Vector  # Vector of Distribution objects
    discount_factor::Float64
end


"""
    NeuralNetworkParameters

Parameters for the neural network architecture.
"""
Base.@kwdef struct NeuralNetworkParameters
    num_hidden_layers::Int = 3
    nn_width::Int = 1000
    activation_name::Symbol = :elu
end


"""
    AlgorithmHyperParameters

Hyperparameters for Algorithm 2 (training the neural networks).
"""
Base.@kwdef struct AlgorithmHyperParameters{T<:AbstractFloat}
    precision::Type{T} = Float32
    batch_size::Int = 5000
    num_intervals::Int = 100
    T_horizon::Float64 = 0.1
    num_iterations::Int = 40000
    learning_rates::Vector{Float64} = [1e-3, 2e-4, 1e-4, 1e-5, 1e-6]
    decay_steps::Vector{Int} = [1, 5000, 10000, 20000, 30000]
    β_penalty::Vector{Float64} = [1e0, 1e1, 1e2, 1e3, 1e4, 1e4, 1e6]
    penalty_steps::Vector{Int} = [1, 2500, 5000, 7500, 10000, 15000, 20000]
    λ_rate::Float64 = 4.0  # Will be computed as 52 / average_weeks_between_orders
    S_guess::Vector{Vector{Float64}} = [[1.0]]  # Order-up-to vector guess
    ν_radius::Float64 = 0.2
    S_distribution::Symbol = :lognormal
    αjs::Vector{Float64} = [0.0]  # Jump size parameters
    cost_scaling::Float64 = 0.01  # κ (multiply to scale, divide to unscale)
end


"""
    OptimizationParameters

Parameters for finding the order-up-to vector.
"""
Base.@kwdef struct OptimizationParameters
    start_factor::Float64 = 1.0
    lower_factor::Float64 = 0.5
    upper_factor::Float64 = 1.5
    epsilon_factors::Vector{Float64} = [
        -1.25, -2.5, -5.0, -7.5, -10.0, -12.5, -15.0,
        -20.0, -25.0, -30.0, -40.0, -50.0, -75.0, -100.0
    ]
    num_simulation_runs::Int = 8
    simulation_horizon::Int = 5000
    # 2D-only NN grid bounds; ignored for d ≠ 2.
    nn_grid_min_inv::Int = -25
    nn_grid_max_inv::Int = 75
end


# =====================================================================
# Policy types for benchmark and NN policy dispatch
# =====================================================================

"""
    AbstractPolicy

Abstract supertype for all inventory control policies.
All subtypes must support `compute_order_quantity!` dispatch.
"""
abstract type AbstractPolicy end

"""
    AbstractStatelessPolicy <: AbstractPolicy

Policy whose order decision depends only on current
inventory level and period index.
"""
abstract type AbstractStatelessPolicy <: AbstractPolicy end

"""
    AbstractStatefulPolicy <: AbstractPolicy

Policy whose order decision depends on mutable state
(e.g., cumulative demand since last order).
"""
abstract type AbstractStatefulPolicy <: AbstractPolicy end

"""
    IndividualSSPolicy <: AbstractStatelessPolicy

Per-item (sᵢ, Sᵢ) policy. Orders item i up to Sᵢ
whenever inventory_i ≤ sᵢ.
"""
struct IndividualSSPolicy <: AbstractStatelessPolicy
    s_levels::Vector{Int}
    S_levels::Vector{Int}
    alpha::Float64
end

"""
    PeriodicRSPolicy <: AbstractStatelessPolicy

Fixed review interval R, order-up-to S.
Orders all items up to Sᵢ every R periods.
"""
struct PeriodicRSPolicy <: AbstractStatelessPolicy
    S_levels::Vector{Int}
    R::Int
end

"""
    AggregateDemandQSPolicy <: AbstractStatefulPolicy

Orders when cumulative aggregate demand ≥ Q.
After ordering, cumulative demand resets to 0.
"""
mutable struct AggregateDemandQSPolicy <: AbstractStatefulPolicy
    const S_levels::Vector{Int}
    const Q::Int
    cumulative_demand::Int
end

function AggregateDemandQSPolicy(S_levels::Vector{Int}, Q::Int)
    return AggregateDemandQSPolicy(S_levels, Q, 0)
end

"""
    HybridRQSPolicy <: AbstractStatefulPolicy

Hybrid (R,Q,S) policy: orders when cumulative aggregate
demand ≥ Q OR R periods have elapsed since the last order,
whichever comes first. After ordering, both counters reset.
"""
mutable struct HybridRQSPolicy <: AbstractStatefulPolicy
    const S_levels::Vector{Int}
    const R::Int
    const Q::Int
    cumulative_demand::Int
    periods_since_order::Int
end

function HybridRQSPolicy(
    S_levels::Vector{Int}, R::Int, Q::Int
)
    return HybridRQSPolicy(S_levels, R, Q, 0, 0)
end

"""
    CanOrderPolicy <: AbstractStatelessPolicy

Can-order policy: when any item hits its reorder point,
other items can order if below their can-order level.
oᵢ = ω * sᵢ + (1 - ω) * Sᵢ
"""
struct CanOrderPolicy <: AbstractStatelessPolicy
    s_levels::Vector{Int}
    S_levels::Vector{Int}
    omega::Float64
    alpha::Float64
end

"""
    MDPPolicy <: AbstractStatelessPolicy

Exact MDP optimal policy for 2D instances. Maps each
inventory state (s₁, s₂) to optimal order quantities
(a₁, a₂). States outside the MDP state space are clamped
to the nearest boundary.
"""
struct MDPPolicy <: AbstractStatelessPolicy
    policy::Dict{Tuple{Int16, Int16}, Tuple{Int16, Int16}}
    min_inv::Int16
    max_inv::Int16
end

function MDPPolicy(
    policy::Dict{Tuple{Int16, Int16}, Tuple{Int16, Int16}}
)
    keys_iter = keys(policy)
    min_inv = minimum(k -> min(k[1], k[2]), keys_iter)
    max_inv = maximum(k -> max(k[1], k[2]), keys_iter)
    return MDPPolicy(policy, min_inv, max_inv)
end

"""
    NNGridPolicy <: AbstractStatelessPolicy

Pre-baked NN policy for 2D instances. Same shape as
`MDPPolicy`: a dict mapping inventory state (x₁, x₂)
to order quantity (a₁, a₂), derived from the trained
NN at one (ε, z*) pair. `z_star` and `epsilon` are kept
as metadata for logging and downstream reporting.
"""
struct NNGridPolicy <: AbstractStatelessPolicy
    policy::Dict{Tuple{Int16, Int16}, Tuple{Int16, Int16}}
    min_inv::Int16
    max_inv::Int16
    z_star::Vector{Int}
    epsilon::Float64
end

"""
    NeuralNetworkPolicy{T} <: AbstractStatelessPolicy

Policy based on trained neural networks. Uses V_operator
for the no-action condition and pre-computed z* as
order-up-to vector. Parametric on the neural network type
to avoid forward reference to NeuralNetworks.
"""
struct NeuralNetworkPolicy{T} <: AbstractStatelessPolicy
    nns::T
    z_star::Vector{Int}
    epsilon::Float64
    impulse_prob::ImpulseControlProblem
    # 2D-only hard rules; sentinels disable them.
    min_inv::Int
    max_inv::Int
end

function NeuralNetworkPolicy(
    nns, z_star::Vector{Int}, epsilon::Float64,
    impulse_prob::ImpulseControlProblem
)
    return NeuralNetworkPolicy(
        nns, z_star, epsilon, impulse_prob,
        typemin(Int), typemax(Int)
    )
end

"""
    ZhengFedergruenParameters

Parameters for a single-item (s,S) computation using
the Zheng-Federgruen (1991) average cost algorithm.
"""
struct ZhengFedergruenParameters
    K::Float64
    h::Float64
    p::Float64
    c::Float64
    γ::Float64
    demand_dist::Any  # Distribution object
    demand_support_max::Int
end

"""
    BenchmarkSearchResult

Container for results of a benchmark parameter search.
"""
struct BenchmarkSearchResult
    policy::AbstractPolicy
    cost::Float64
    std_error::Float64
    ordering_frequency::Float64
    analytical_cost::Union{Float64, Nothing}
    search_params::Dict{Symbol, Any}
end

# Function stubs for policy dispatch (methods defined in
# simulation.jl and benchmark_policies.jl)
function compute_order_quantity! end
function reset_state! end
function update_state_after_demand! end


