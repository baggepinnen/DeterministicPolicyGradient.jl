r5(x) = round(x,5)
const b = Beta(1,5)

import Base.Collections: heapify, heapify!, heappush!, heappop!
import Base: isless, zero, push!, show, display, getindex, setindex!, sort!, length, start, next, done

export Transition, ReplayMemory,SequentialReplayMemory,SortedReplayMemory, sample_greedy!, sample_beta!, sample_uniform!, push!, delete_and_push!, sort!, setindex!

type Transition
    s::Vector{Float64}
    s1::Vector{Float64}
    a::Vector{Float64}
    r::Float64
    δ::Float64
    t::Union{Int,Float64}
    i::Int
end

display(t::Transition) = println("s: ", string(r5(t.s)), " s1: ", string(r5(t.s1)), " a: ", string(r5(t.a)), " r: ", string(r5(t.r)), " δ: ", string(r5(t.δ)))
show(t::Transition) = display(t)

isless(t1::Transition,t2::Transition) = isless(abs(t1.δ),abs(t2.δ))
zero(Transition) = Transition([0.],[0.],[0.],0.,0,0,0)


abstract ReplayMemory

type SortedReplayMemory <: ReplayMemory
    mem::Vector{Transition}
    s::Int
    rs::Float64
    SortedReplayMemory(s::Int) = new(Vector{Transition}(),s, sum(1./(1:s)))
end

type SequentialReplayMemory <: ReplayMemory
    mem::Vector{Transition}
    s::Int
    _i::Int
    _filled::Bool
    _last_sampled::Vector{Int}
    SequentialReplayMemory(s::Int) = new(Vector{Transition}(s),s,0, false,Int[])
end

start(mem::ReplayMemory) = start(mem.mem)
next(mem::ReplayMemory,i) = next(mem.mem,i)
done(mem::SortedReplayMemory,i) = done(mem.mem,i)
done(mem::SequentialReplayMemory,i) = (mem._filled ? mem.s : mem._i) < i



display(mem::SortedReplayMemory) = map(display,mem.mem)
display(mem::SequentialReplayMemory) = mem._filled ? map(display,mem.mem) : [display(mem[i]) for i = 1:mem._i]
show(t::Transition) = map(display,mem.mem)

getindex(mem::ReplayMemory, ind) = mem.mem[ind]
setindex!(mem::ReplayMemory, trans::Transition, ind) = setindex!(mem.mem, trans, ind)
sort!(mem::SortedReplayMemory) = sort!(mem.mem,rev=true)
sort!(mem::SequentialReplayMemory) = nothing

function display(mem::ReplayMemory)
    map(display,mem.mem)
    nothing
end

length(mem::SortedReplayMemory) = length(mem.mem)
length(mem::SequentialReplayMemory) = mem._filled ? mem.s : mem._i

function push!(mem::SortedReplayMemory, t::Transition)
    if length(mem) >= mem.s
        return delete_and_push!(mem,t)
    end
    heappush!(mem.mem, t, Base.Order.Reverse)
    nothing
end

function push!(mem::ReplayMemory, t::Vector{Transition})
    for ti in t
        push!(mem,ti)
    end
end

function push!(mem::SequentialReplayMemory, t::Transition)
    if length(mem._last_sampled) > 0
        ind = pop!(mem._last_sampled)
        mem[ind] = t
        return
    end
    mem._i = (mem._i % mem.s) + 1
    mem[mem._i] = t
end


function delete_and_push!(mem::SortedReplayMemory, t::Transition)
    l2 = length(mem) ÷ 2
    deleteat!(mem.mem,l2+rand(1:(l2)))
    heappush!(mem.mem, t, Base.Order.Reverse)
    nothing
end

function sample_greedy!(mem::SortedReplayMemory)
    heappop!(mem.mem, Base.Order.Reverse)
    # mem[1]
end


function sample_beta!(mem::SortedReplayMemory)
    ind = ceil(Int,length(mem)*rand(b))
    retval = mem[ind]
    deleteat!(mem.mem,ind)
    retval
end

function sample_uniform!(mem::SortedReplayMemory)
    ind = rand(1:length(mem))
    retval = mem[ind]
    deleteat!(mem.mem,ind)
    retval
end

function sample_uniform!(mem::SequentialReplayMemory)
        lastind = mem._filled ? mem.s : mem._i
        ind = rand(1:lastind)
        push!(mem._last_sampled, ind)
        retval = mem[ind]
end

function sample_many!(mem::ReplayMemory,n,sample_fun)
    retvec = Vector{Transition}(n)
    for i = 1:n
        retvec[i] = sample_fun(mem)
    end
    retvec
end


sample_greedy!(mem::SortedReplayMemory,n) = sample_many!(mem::SortedReplayMemory,n,sample_greedy!)
sample_beta!(mem::SortedReplayMemory,n) = sample_many!(mem::SortedReplayMemory,n,sample_beta!)
sample_uniform!(mem::ReplayMemory,n) = sample_many!(mem::ReplayMemory,n,sample_uniform!)
