# nn_grid.jl - Grid materialization of NN policies for 2D instances

using Base.Threads


"""
    compute_nn_value_grid(nns, impulse_prob, min_inv, max_inv)

Enumerate the 2D inventory grid [min_inv, max_inv]² and
evaluate `V_operator` (the no-action value N(x) = LV + rV - f)
at every state. Returns a dict keyed by `(Int16, Int16)`
inventory states.
"""
function compute_nn_value_grid(
    nns::NeuralNetworks,
    impulse_prob::ImpulseControlProblem,
    min_inv::Integer,
    max_inv::Integer
)
    @assert impulse_prob.dim == 2 "nn grid is 2D only"

    states = [
        (Int16(x1), Int16(x2))
        for x1 in min_inv:max_inv, x2 in min_inv:max_inv
    ]
    n = length(states)
    values = Vector{Float64}(undef, n)

    @threads for i in 1:n
        x1, x2 = states[i]
        values[i] = V_operator(
            nns, Float64[x1, x2], impulse_prob
        )
    end

    return Dict{Tuple{Int16, Int16}, Float64}(
        states[i] => values[i] for i in 1:n
    )
end


"""
    bake_grid_policy(values, z_star, epsilon, min_inv, max_inv)

Derive an `NNGridPolicy` from the precomputed value grid.
For each state x in the grid, the order quantity is:

    a = max.(z* − x, 0)   if  values[x] < ε   (order)
    a = (0, 0)            otherwise           (no order)

This mirrors `compute_order_quantity!(::NeuralNetworkPolicy)`.
The hard rules (never-order when x ≥ 0, always-order outside
the grid) are enforced at simulation time by the dispatch,
not in this function — the dict is intentionally a faithful
representation of the NN's raw decision over the grid.
"""
function bake_grid_policy(
    values::Dict{Tuple{Int16, Int16}, Float64},
    z_star::Vector{Int},
    epsilon::Float64,
    min_inv::Integer,
    max_inv::Integer
)
    @assert length(z_star) == 2 "nn grid is 2D only"
    z1, z2 = z_star[1], z_star[2]

    policy = Dict{Tuple{Int16, Int16}, Tuple{Int16, Int16}}()
    sizehint!(policy, length(values))

    for (state, v) in values
        if v < epsilon
            a1 = Int16(max(z1 - state[1], 0))
            a2 = Int16(max(z2 - state[2], 0))
            policy[state] = (a1, a2)
        else
            policy[state] = (Int16(0), Int16(0))
        end
    end

    return NNGridPolicy(
        policy, Int16(min_inv), Int16(max_inv),
        z_star, epsilon
    )
end
