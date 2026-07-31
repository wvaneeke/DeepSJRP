using Plots
using JSON
using LaTeXStrings
using Random

# --- 1. Set the Plotting Backend for High-Quality LaTeX Output ---
pgfplotsx() 

"""
    plot_policy_comparison(mdp_policy, nn_policy, mdp_S, nn_S, instance_name)

Generates and saves a comparison plot for MDP and NN inventory policies.

# Arguments
- `mdp_policy::Dict`: A dictionary mapping state tuples to order quantity tuples for the MDP policy.
- `nn_policy::Dict`: A dictionary mapping state tuples to order quantity tuples for the NN policy.
- `mdp_S::Tuple`: The S-vector (s1_max, s2_max) for the MDP policy.
- `nn_S::Tuple`: The S-vector (s1_max, s2_max) for the NN policy.
- `instance_name::String`: The name of the instance, used for the output filename.
"""
function plot_policy_comparison(mdp_policy, nn_policy, instance_name)

    # NOTE: You must define your S-vectors here. Using placeholder values.
    # These should be loaded or defined based on your problem instance.
    mdp_S = mdp_policy[-50, -50] .- 50
    nn_S  = nn_policy[-50, -50] .- 50

    # >>> ADD THESE LINES TO SEE THE COORDINATES <<<
    println("-----------------------------------------")
    println("Attempting to plot MDP S-vector at: ", mdp_S)
    println("Attempting to plot NN S-vector at:  ", nn_S)
    println("Plot axis limits are roughly [-15, 50].")
    println("-----------------------------------------")
    
    # --- 2. Define the Plotting Range ---
    # Adjust these values to zoom in or out on the relevant area of the state space
    s1_range = -15:1:55
    s2_range = -15:1:30

    # --- 3. Process Policies to Categorize States and Targets ---
    mdp_order_states = Tuple{Int, Int}[]
    nn_order_states = Tuple{Int, Int}[]
    agreement_states = Tuple{Int, Int}[]

    for s1 in s1_range, s2 in s2_range
        state = (s1, s2)
        
        # This logic correctly checks if a state is within the S-vector bounds before considering it
        mdp_orders = (s1 <= mdp_S[1] && s2 <= mdp_S[2]) && (sum(get(mdp_policy, state, (0, 0))) > 0)
        nn_orders = (s1 <= nn_S[1] && s2 <= nn_S[2]) && (sum(get(nn_policy, state, (0, 0))) > 0)
        
        if mdp_orders && nn_orders
            push!(agreement_states, state)
        elseif mdp_orders
            push!(mdp_order_states, state)
        elseif nn_orders
            push!(nn_order_states, state)
        end
    end

    # --- 3. Generate the Plot ---
    function scatter_plot!(states, color, marker, label)
        if !isempty(states)
            x = [s[1] for s in states]
            y = [s[2] for s in states]
            markersize = (marker == :xcross ? 6 : 2.5) # Made crosses bigger for visibility
            scatter!(x, y; color=color, marker=marker, label=label, markersize=markersize, markerstrokewidth= (marker == :xcross ? 1.5 : 0) )
        end
    end

    # New helper function for the "circle with a cross" effect
    function scatter_overlay!(states, color, label)
        if !isempty(states)
            x = [s[1] for s in states]
            y = [s[2] for s in states]
            # Plot an open circle
            scatter!(x, y, label="", marker=:circle, markersize=3, 
                     markerstrokecolor=color, markerstrokewidth=3.0, fillalpha=.0)
            # Plot a cross on top
            scatter!(x, y, label=label, marker=:xcross, markersize=3, 
                     markerstrokecolor=:black, markerstrokewidth=1.0)
        end
    end

    plot(
        xlabel=L"x_1",
        ylabel=L"x_2",
        legend=:outertopright, # "false" completely removes the legend
        legendfont=font(halign=:left), # This line left-aligns the legend text
        aspect_ratio=:equal,
        xlims=(minimum(s1_range), maximum(s1_range)),
        ylims=(minimum(s2_range), maximum(s2_range))
    )

    # Plot the ordering regions (dots)
    scatter_plot!(mdp_order_states, :red, :circle, "MDP policy")
    scatter_plot!(nn_order_states, :blue, :circle, "NN policy")
    scatter_plot!(agreement_states, :purple, :circle, "Policies agree")

    # --- Plot the S-vectors with the new overlay marker ---
    scatter_overlay!([mdp_S], :red, "")
    scatter_overlay!([nn_S], :blue, "")
    
    # --- ALTERNATIVE MARKERS (if you don't like the overlay) ---
    # scatter_plot!([mdp_S], :red, :star, "MDP S-vector") # 5-pointed star
    # scatter_plot!([nn_S], :blue, :square, "NN S-vector") # A square

    hline!([0], color=:grey, style=:dash, label=nothing)
    vline!([0], color=:grey, style=:dash, label=nothing)

    output_filename = "policy_comparison_$(instance_name).pdf"
    savefig(output_filename)
    println("✅ Plot successfully saved to: $(output_filename)")
end



# # --- Main Execution Block ---

# # Define the instance name
instance_name = "2D_MMM"


# --- 2. Load Policies from JSON Files ---
println("Loading policies for instance: $(instance_name)")
try
    # Define filenames
    mdp_filename = "policies/policies2D/mdppolicy_$(instance_name).json"
    nn_filename = "policies/policies2D/nnpolicy_$(instance_name).json"

    # Load and reconstruct MDP policy
    mdp_array = JSON.parsefile(mdp_filename);
    mdp_policy = Dict(Tuple(item[1]) => Tuple(item[2]) for item in mdp_array);

    # Load and reconstruct NN policy
    nn_array = JSON.parsefile(nn_filename);
    nn_policy = Dict(Tuple(item[1]) => Tuple(item[2]) for item in nn_array);

    println("Successfully loaded both policies from JSON files.")

    # --- 3. Call the Main Plotting Function ---
    plot_policy_comparison(mdp_policy, nn_policy, instance_name)

catch e
    println("\n❌ ERROR: Could not load or plot the policy files.")
    println("Please ensure these files exist and are valid JSON:")
    println("  - policies/policies2D/mdppolicy_$(instance_name).json")
    println("  - policies/policies2D/nnpolicy_$(instance_name).json")
    println("\nError details: ", e)
end

