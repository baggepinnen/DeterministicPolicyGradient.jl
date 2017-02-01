export SignNoise, MarkovNoise, ismarkov

"""SignNoise(change_probability,initial_sign=1)

`rand(sn::SignNoise)` produces either 1 or -1, with a change since last call depending on the probability `change_probability`
"""
type SignNoise
    change_prob::Float64
    last_sign::Int
    SignNoise(c,l=1) = new(c,l)
end
function Base.rand(sn::SignNoise)
    if rand() < sn.change_prob
        sn.last_sign *= -1
    end
    sn.last_sign
end


ismarkov(P) = all(sum(P,1) .== 1)

"""MarkovNoise(outputs::AbstractVector, transition_matrix::AbstractMatrix, initial_state::Int = 1)
Noise defined by a vector of possible `outputs` and the transition probability matrix of probabilities of switching between them. `transition_matrix` must be square and have columns that sum to one, columns for efficiency reasons.
"""
type MarkovNoise
    P::Matrix{Float64}
    state::Int
    outputs::Vector{Float64}

    function MarkovNoise(outputs::AbstractVector, transition_matrix::AbstractMatrix, initial_state::Int = 1)
        @assert ismarkov(transition_matrix) "The columns (columns for for efficiency reaons) of the transition matrix must sum to 1"
        @assert reduce(==,size(transition_matrix)) "transition_matrix must be square"
        new(transition_matrix, initial_state, outputs)
    end
end


function Base.rand(mn::MarkovNoise)
    cumulative_probability = mn.P[1,mn.state]
    r = rand()
    for i = 1:length(mn.outputs)
        if r < cumulative_probability
            mn.state = i
            break
        end
        cumulative_probability += mn.P[i+1,mn.state]
    end
    return mn.outputs[mn.state]
end

function Base.rand(mn::MarkovNoise,n)
    [rand(mn) for i = 1:n]
end
