module DeterministicPolicyGradient
using Distributions, TensorFlow
export DPGopts, DPGfuns, DPGstate, dpg, meshgrid, meshgrid2, get_centers_multi, quadform

include("utilities.jl")
include("dpg_tensorflow.jl")
include("experience_replay.jl")

end # module
