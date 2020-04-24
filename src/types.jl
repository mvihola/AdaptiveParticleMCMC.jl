# Augment a null progress meter which disables it
import ProgressMeter.next!
struct NullProgress <: ProgressMeter.AbstractProgress
    NullProgress()=new()
end
next!(::NullProgress) = nothing

struct SMCState{ioT<:SMCIO, modelT<:SMCModel,
    lMT<:Union{Function,Nothing}, setT<:Function}
    model::modelT
    io::ioT
    lM::lMT
    set_param!::setT
end

"""
   state = SMCState(T, N, ParticleType, ScratchType, lG, M!, [lM=nothing]; kwargs...)

Data structures for Sequential Monte Carlo.

# Arguments:
- `T`: Length of the data record (or the hidden Markov model).
- `N`: Number of particles.
- `ParticleType`: Data type of particles.
- `ScratchType`: Data type of scratch (auxiliary data) object.
- `set_param!`: Function which sets the parameters of the model:
  `set_param!(scratch, par)` sets `scratch` to match parameters `par`.
- `lG`: Log-potential function: `lG(k, x, scratch)` returns value of potential
  at time `k` for particle `x` given `scratch`.
- `M!`: Function which simulates from the transition density:
  `M!(x, rng, k, x_, scratch)` evolves `x_` from time `k-1` to `x` at time `k`
  given random number generator `rng` and `scratch`.
- `lM`: Function which returns transition density values:
  `lM(k, x_, x, scratch)` where arguments are as above. (Only required for PG)

# Keyword arguments:
- `nthreads`: Number of threads used within the particle filter; default `1`.
- `essThreshold`: Effective sample size threshold for resampling; default `2.0`.
"""
function SMCState(T::Int, N::Int, ParticleType, ScratchType,
    set_param!::Function, lG::Function, M!::Function,
    lM::Union{Function,Nothing}=nothing; nthreads=1, essThreshold=2.0)
    model = SMCModel(M!, lG, T, ParticleType, ScratchType)
    io = SMCIO{ParticleType,ScratchType}(N, T, nthreads, true, essThreshold)
    SMCState(model, io, lM, set_param!)
end

# Function which is essentially min(1, exp(p_ - p)),  but handles
# exceptional cases gracefully
@inline function _accept_prob(p, p_, FT)
    if !isfinite(p_) # p_ is not finite => force reject
        return zero(FT)
    elseif !isfinite(p) # p_ is finite but p not => force accept
        return one(FT)
    else # both finite, the standard case:
        return (p_ >= p) ? one(FT) : exp(p_ - p)
    end
end
