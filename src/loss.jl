# loss.jl - Lagrangian loss function and cost functions


"""
    inventory_cost(x, h, p)

Compute the inventory holding and backlog cost f(x).
Corresponds to equation (1) in the paper.

Arguments:
- x: Inventory state, shape (d,) or (d, N, K)
- h: Holding cost vector
- p: Penalty (backlog) cost vector

Returns:
- Total inventory cost, summed across dimensions
"""
function inventory_cost(x::AbstractArray, h::AbstractVector, p::AbstractVector)
    return sum(h .* max.(x, 0) .+ p .* max.(-x, 0), dims=1)
end


"""
    ordering_cost(ξ, ci, c0)

Compute the ordering cost c(y).
Includes fixed cost c0 if any order quantity is positive.

Arguments:
- ξ: Order quantity, shape (d,) or (d, N, K)
- ci: Variable cost vector
- c0: Fixed cost

Returns:
- Total ordering cost
"""
function ordering_cost(ξ::AbstractArray, ci::AbstractVector, c0::Real)
    variable = sum(ξ .* ci, dims=1)
    fixed = any(x -> x > 0, ξ, dims=1) .* c0
    return variable .+ fixed
end


"""
    create_cost_functions(h, p, ci, c0)

Create closure cost functions for the inventory problem.

Returns:
- f: Inventory cost function
- c: Ordering cost function
"""
function create_cost_functions(
    h::AbstractVector,
    p::AbstractVector,
    ci::AbstractVector,
    c0::Real
)
    f(x) = inventory_cost(x, h, p)
    c(ξ) = ordering_cost(ξ, ci, c0)
    return f, c
end


"""
    lagrangian_loss(nns, X_samplepaths, delta_B, delta_U, β_m, K, T, N, r, f, c, device_fn, _type; mode)

Compute the Lagrangian loss function from equation (17) in the paper.

Arguments:
- nns: Neural networks (G, H)
- X_samplepaths: Sample paths of inventory state
- delta_B: Brownian increments
- delta_U: Jump increments
- β_m: Current penalty parameter
- K: Batch size
- T: Time horizon
- N: Number of intervals
- r: Interest rate
- f: Inventory cost function
- c: Ordering cost function
- device_fn: Device transfer function
- _type: Float type
- mode: Loss computation mode

Modes:
- :training - Full Lagrangian: -E[H(x₀)] + β·E[penalty²]
- :αtraining - Percentage of paths without violations
- :validation - Just E[H(x₀)]
"""
function lagrangian_loss(
    nns,
    X_samplepaths::AbstractArray,
    delta_B::AbstractArray,
    delta_U::AbstractArray,
    β_m::Real,
    K::Int,
    T::Real,
    N::Int,
    r::Real,
    f::Function,
    c::Function,
    device_fn,
    _type::Type;
    mode::Symbol=:training
)
    H_x0, penalties = forward_pass(nns, X_samplepaths, delta_B, delta_U, T, N, r, f, c, device_fn, _type)
    
    if mode == :training
        # Equation (17): Lβ(θ,ϑ) = -E[H(x₀)] + β·E[l(penalty)]
        # where l(x) = max(x, 0)²
        return (-sum(H_x0) + β_m * sum(penalties .^ 2)) / K
        
    elseif mode == :αtraining
        # Return percentage of paths without violations (α(β))
        return 100.0 * (1.0 - count(x -> x > 0, penalties) / K)
        
    elseif mode == :validation
        # Just return average value function estimate
        return sum(H_x0) / K
        
    else
        error("Unknown mode: $(mode). Use :training, :αtraining, or :validation")
    end
end
