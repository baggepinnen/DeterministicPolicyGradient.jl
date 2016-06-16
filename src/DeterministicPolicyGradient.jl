module DeterministicPolicyGradient
using Distributions
export DPGopts, DPGfuns, DPGstate, dpg, meshgrid, meshgrid2, get_centers_multi, quadform

include("utilities.jl")
include("dpg.jl")
include("experience_replay.jl")

end # module
