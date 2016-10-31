module DeterministicPolicyGradient
using Distributions, Plots
export Batch, DPGopts, DPGfuns, DPGstate, dpg, meshgrid, meshgrid2, get_centers_multi, quadform, Qplot, Qplot!, autoscale, update_plot!

include("experience_replay.jl")
include("utilities.jl")
include("dpg_tensorflow.jl")

end # module
