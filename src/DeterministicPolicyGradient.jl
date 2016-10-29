module DeterministicPolicyGradient
using Distributions, TensorFlow, Plots
export Batch, DPGopts, DPGfuns, DPGstate, dpg, meshgrid, meshgrid2, get_centers_multi, quadform, Qplot, Qplot!

include("experience_replay.jl")
include("utilities.jl")
include("dpg_tensorflow.jl")

end # module
