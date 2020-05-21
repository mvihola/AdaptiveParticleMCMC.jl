struct SMCState{ioT<:SMCIO, modelT<:SMCModel,
    lMT<:Union{Function,Nothing}, setT<:Function, refT}
    model::modelT
    io::ioT
    lM::lMT
    set_param!::setT
    ref::refT
end

"""
   state = SMCState(T, N, ParticleType, ScratchType, set_param!, lG, M!,
                    [lM=nothing]; kwargs...)

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

For details, see documentation of `SequentialMonteCarlo.jl`
"""
function SMCState(T::Int, N::Int, ParticleType, ScratchType,
    set_param!::Function, lG::Function, M!::Function,
    lM::Union{Function,Nothing}=nothing; nthreads=1, essThreshold=2.0)
    model = SMCModel(M!, lG, T, ParticleType, ScratchType)
    io = SMCIO{ParticleType,ScratchType}(N, T, nthreads, true, essThreshold)
    ref = [model.particle() for i=1:T]
    SMCState(model, io, lM, set_param!, ref)
end

# Helper which calculates model log-likelihood for given reference
@inline function _reference_log_likelihood(state::SMCState)
    ref = state.ref; lG = state.model.lG; lM = state.lM;
    pScratch = state.io.internal.particleScratch
    L = 0.0
    @inbounds for k=1:length(ref)
        k_ = max(1, k-1); x = ref[k]; x_ = ref[k_]
        L += lG(k, x, pScratch)
        L += lM(k, x_, x, pScratch)
    end
    L
end

# Wrappers to SequentialMonteCarlo interface
@inline function _set_model_param!(state::SMCState, theta)
    state.set_param!(state.io.internal.particleScratch, theta)
    nothing
end
@inline function _run_smc!(state::SMCState)
    smc!(state.model, state.io)
end
@inline function _log_likelihood(state::SMCState)
    state.io.logZhats[end]
end
@inline function _pick_particle!(state::SMCState)
    SequentialMonteCarlo.pickParticle!(state.ref, state.io)
end
@inline _save_reference!(state::SMCState) = nothing
@inline function _init_path_storage(state::SMCState, nsim)
    [[state.model.particle() for i=1:state.io.n] for j=1:nsim]
end
@inline function _copy_reference!(out, state::SMCState)
    SequentialMonteCarlo._copyParticles!(out, state.ref)
end
@inline function _run_csmc!(state::SMCState, backward_sampling::Bool)
    csmc!(state.model, state.io, state.ref, state.ref)
    if backward_sampling
        SequentialMonteCarlo.pickParticleBS!(state.ref, state.io, state.lM)
    end
    nothing
end
