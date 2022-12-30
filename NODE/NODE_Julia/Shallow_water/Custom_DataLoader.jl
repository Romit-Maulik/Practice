# Adapted from Knet's src/data.jl (author: Deniz Yuret)
using Random: AbstractRNG, shuffle!, GLOBAL_RNG
using Base: @propagate_inbounds

struct MyDataLoader{D,R<:AbstractRNG}
    data::D
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    s_indices::Vector{Int}
    shuffle::Bool
    rng::R
end

_nobs(data::AbstractArray) = size(data)[end]
_nsim(data::AbstractArray) = size(data)[1]
_getobs(data::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_getobs, i), data)


function MyDataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))

    n = _nobs(data)

    if ndims(data[1]) > 2 
        nsim = size(data[1])[1]
    else
        nsim = 1    
    end

    if n < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
        batchsize = n
    end
    imax = partial ? n : n - batchsize + 1

    MyDataLoader(data, batchsize, n, partial, imax, [1:n;], [1:nsim;], shuffle, rng)
end

@propagate_inbounds function Base.iterate(d::MyDataLoader, i=0)     # returns data in d.indices[i+1:i+batchsize]
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.rng, d.indices)
        shuffle!(d.rng, d.s_indices)
    end

    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i+1:nexti]
    next_sim = d.s_indices[1]

    batch = d.data[1][next_sim,:, ids]
    time_batch = d.data[2][next_sim,1, ids]

    return ((batch, time_batch), nexti)

end

function Base.length(d::MyDataLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

function _nobs(data::Union{Tuple, NamedTuple})
    length(data) > 0 || throw(ArgumentError("Need at least one data input"))
    n = _nobs(data[1])
    for i in keys(data)
        ni = _nobs(data[i])
        n == ni || throw(DimensionMismatch("All data inputs should have the same number of observations, i.e. size in the last dimension. " * 
            "But data[$(repr(first(keys(data))))] ($(summary(data[1]))) has $n, while data[$(repr(i))] ($(summary(data[i]))) has $ni."))
    end
    return n
end


Base.eltype(::MyDataLoader{D}) where D = D
