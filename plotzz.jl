using JLD2
using Plots
using LaTeXStrings

pgfplotsx()


"""
    load_mdp_policy(path)

Load an MDP policy from JLD2. Returns the raw dict and the
(min_inv, max_inv) bounds derived from the dict keys.
"""
function load_mdp_policy(path::String)
    data = JLD2.load(path)
    policy = data["mdp_policy"]
    min_inv = minimum(k -> min(k[1], k[2]), keys(policy))
    max_inv = maximum(k -> max(k[1], k[2]), keys(policy))
    return policy, Int(min_inv), Int(max_inv)
end


"""
    load_nn_grid_policy(path)

Load an NN grid policy from JLD2. Returns dict + bounds + z* + ε.
"""
function load_nn_grid_policy(path::String)
    data = JLD2.load(path)
    return (
        policy=data["policy"],
        min_inv=Int(data["min_inv"]),
        max_inv=Int(data["max_inv"]),
        z_star=Int.(data["z_star"]),
        epsilon=Float64(data["epsilon"])
    )
end


function raw_dict_orders(
    dict, min_inv::Int, max_inv::Int, x1::Int, x2::Int
)
    s1 = clamp(Int16(x1), Int16(min_inv), Int16(max_inv))
    s2 = clamp(Int16(x2), Int16(min_inv), Int16(max_inv))
    a = get(dict, (s1, s2), (Int16(0), Int16(0)))
    return (max(Int(a[1]), 0), max(Int(a[2]), 0))
end


function mdp_effective_orders(
    dict, min_inv::Int, max_inv::Int, x1::Int, x2::Int
)
    if x1 >= 0 && x2 >= 0
        return (0, 0)
    end
    s1 = clamp(Int16(x1), Int16(min_inv), Int16(max_inv))
    s2 = clamp(Int16(x2), Int16(min_inv), Int16(max_inv))
    a = get(dict, (s1, s2), (Int16(0), Int16(0)))
    if x1 < min_inv || x2 < min_inv
        t1 = Int(s1) + Int(a[1])
        t2 = Int(s2) + Int(a[2])
        return (max(t1 - x1, 0), max(t2 - x2, 0))
    end
    return (max(Int(a[1]), 0), max(Int(a[2]), 0))
end


function nn_effective_orders(
    dict, min_inv::Int, max_inv::Int,
    z_star::Vector{Int}, x1::Int, x2::Int
)
    if x1 >= 0 && x2 >= 0
        return (0, 0)
    end
    if x1 < min_inv || x2 < min_inv
        return (max(z_star[1] - x1, 0), max(z_star[2] - x2, 0))
    end
    s1 = clamp(Int16(x1), Int16(min_inv), Int16(max_inv))
    s2 = clamp(Int16(x2), Int16(min_inv), Int16(max_inv))
    a = get(dict, (s1, s2), (Int16(0), Int16(0)))
    return (max(Int(a[1]), 0), max(Int(a[2]), 0))
end


"""
    plot_policy_comparison(mdp_dict, mdp_min, mdp_max,
                            nn_dict, nn_min, nn_max, nn_z,
                            instance_name; kwargs...)

Plot MDP vs NN ordering regions over a grid of inventory states.
"""
function plot_policy_comparison(
    mdp_dict, mdp_min::Int, mdp_max::Int,
    nn_dict, nn_min::Int, nn_max::Int, nn_z::Vector{Int},
    instance_name::String;
    s1_range::AbstractRange=-15:1:55,
    s2_range::AbstractRange=-15:1:30,
    output_filename::String="policy_comparison_$(replace(instance_name, "Case" => "")).pdf",
    apply_hard_rules::Bool=false
)
    mdp_a0 = mdp_effective_orders(
        mdp_dict, mdp_min, mdp_max, mdp_min, mdp_min
    )
    mdp_S = (mdp_min + mdp_a0[1], mdp_min + mdp_a0[2])
    nn_S  = (Int(nn_z[1]), Int(nn_z[2]))

    println("-"^50)
    println("Instance:         $(instance_name)")
    println("MDP S-vector:     $(mdp_S)")
    println("NN  S-vector:     $(nn_S)")
    println("Plot range:       x1=$(s1_range), x2=$(s2_range)")
    println("apply_hard_rules: $(apply_hard_rules)")
    println("-"^50)

    mdp_only = Tuple{Int, Int}[]
    nn_only  = Tuple{Int, Int}[]
    both     = Tuple{Int, Int}[]

    mdp_fn = apply_hard_rules ?
        (s1, s2) -> mdp_effective_orders(
            mdp_dict, mdp_min, mdp_max, s1, s2
        ) :
        (s1, s2) -> raw_dict_orders(
            mdp_dict, mdp_min, mdp_max, s1, s2
        )
    nn_fn = apply_hard_rules ?
        (s1, s2) -> nn_effective_orders(
            nn_dict, nn_min, nn_max, nn_z, s1, s2
        ) :
        (s1, s2) -> raw_dict_orders(
            nn_dict, nn_min, nn_max, s1, s2
        )

    for s1 in s1_range, s2 in s2_range
        mdp_a = mdp_fn(s1, s2)
        nn_a  = nn_fn(s1, s2)
        mdp_orders = sum(mdp_a) > 0
        nn_orders  = sum(nn_a)  > 0
        if mdp_orders && nn_orders
            push!(both, (s1, s2))
        elseif mdp_orders
            push!(mdp_only, (s1, s2))
        elseif nn_orders
            push!(nn_only, (s1, s2))
        end
    end

    function scatter_dots!(states, color, label)
        isempty(states) && return
        x = [s[1] for s in states]
        y = [s[2] for s in states]
        scatter!(
            x, y; color=color, marker=:circle,
            label=label, markersize=2.5, markerstrokewidth=0
        )
    end

    function scatter_S!(point, color, label)
        scatter!(
            [point[1]], [point[2]],
            label="", marker=:circle, markersize=3,
            markerstrokecolor=color, markerstrokewidth=3.0,
            fillalpha=0.0
        )
        scatter!(
            [point[1]], [point[2]],
            label=label, marker=:xcross, markersize=3,
            markerstrokecolor=:black, markerstrokewidth=1.0
        )
    end

    plot(
        xlabel=L"x_1",
        ylabel=L"x_2",
        legend=:outertopright,
        legendfont=font(halign=:left),
        aspect_ratio=:equal,
        xlims=(minimum(s1_range), maximum(s1_range)),
        ylims=(minimum(s2_range), maximum(s2_range))
    )

    scatter_dots!(mdp_only, :red,    "MDP policy")
    scatter_dots!(nn_only,  :blue,   "NN policy")
    scatter_dots!(both,     :purple, "Policies agree")

    scatter_S!(mdp_S, :red,  "")
    scatter_S!(nn_S,  :blue, "")

    hline!([0], color=:grey, style=:dash, label=nothing)
    vline!([0], color=:grey, style=:dash, label=nothing)

    savefig(output_filename)
    println("Saved: $(output_filename)")
end


# =====================================================================
# Main
# =====================================================================

instance_name = "2D_CaseHMM"
nn_method     = "find_stationary"  # or "solve_inf" "find_stationary" 

project_dir = dirname(@__DIR__)
mdp_path = joinpath(
    project_dir, "output", "2d_mdp",
    "mdppolicy_$(instance_name).jld2"
)
nn_path = joinpath(
    project_dir, "policies", "policies2D",
    "nn_grid_$(instance_name)_$(nn_method).jld2"
)

println("Loading MDP policy: $(mdp_path)")
mdp_dict, mdp_min, mdp_max = load_mdp_policy(mdp_path)

println("Loading NN grid policy: $(nn_path)")
nn = load_nn_grid_policy(nn_path)

println("NN grid bounds: [$(nn.min_inv), $(nn.max_inv)]")
println("NN z*: $(nn.z_star), ε = $(nn.epsilon)")

plot_policy_comparison(
    mdp_dict, mdp_min, mdp_max,
    nn.policy, nn.min_inv, nn.max_inv, nn.z_star,
    instance_name,
    apply_hard_rules=true
)
