r5(x) = round(x,5)


import Base.Collections: heapify, heapify!, heappush!, heappop!
import Base: isless, zero, push!, show, display

export Transition, ReplayMemory, sample!, push!

type Transition
    s::Vector{Float64}
    s1::Vector{Float64}
    a::Vector{Float64}
    r::Float64
    δ::Float64
end

display(t::Transition) = println("s: ", string(r5(t.s)), " s1: ", string(r5(t.s1)), " a: ", string(r5(t.a)), " r: ", string(r5(t.r)), " δ: ", string(r5(t.δ)))
show(t::Transition) = display(t)

isless(t1::Transition,t2::Transition) = isless(abs(t1.δ),abs(t2.δ))
zero(Transition) = Transition([0.],[0.],[0.],0.,0.)


type ReplayMemory
    mem::Vector{Transition}
    s::Int
    ReplayMemory(s::Int) = new(zeros(Transition,s),s)
end

function display(mem::ReplayMemory)
    map(display,mem.mem)
    nothing
end

function push!(mem::ReplayMemory, t::Transition)
    heappop!(mem.mem)
    heappush!(mem.mem, t)
    nothing
end

function sample!(mem::ReplayMemory)
    heappop!(mem.mem, ) # TODO: check ordering
end
