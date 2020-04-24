# Helper which calculates model log-likelihood for given reference
@inline function _reference_logLik(ref, lG::lGF, lM::lMF,
    pScratch) where {lGF <: Function, lMF <: Function}
    L = 0.0
    @inbounds for k=1:length(ref)
        k_ = max(1, k-1); x = ref[k]; x_ = ref[k_]
        L += lG(k, x, pScratch)
        L += lM(k, x_, x, pScratch)
    end
    L
end

"""
   out = adaptive_pg(theta0, prior, state, n; kwargs...)

Particle Gibbs (PG) with adaptive proposal on the parameters.

# Arguments
- `theta0`: Initial parameter vector
- `prior`: Function returning prior log density values for parameters
- `state`: `SMCState` data structure
- `n`: Number of iterations

# Optional keyword arguments
- `b`: Burn-in length; default `0.1n`
- `thin`: Thinning interval; default `1` (no thinning)
- `save_paths`: Whether to save (samples of) latent trajectories; default `false`
- `backward_sampling`: Whether to perform backward sampling; default `true`
- `show_progress`: Show progress every `show_progress` seconds (default: `false`).
- `algorithm`: The adaptation algorithm; either Robust Adaptive Metropolis
   `:ram` (default) or Adaptive Scaling within Adaptive Metropolis `:aswam`.

The hidden Markov model is defined in `model`, and the number of
particles (among other particle filter parameters) in `io`; see
the documentation of `SequentialMonteCarlo.jl`

The output `out` is a `NamedTuple` containing the simulation output: `out.acc`
is the mean acceptance rate, `out.Theta` contains the simulated parameter
values (each column is a parameter vector). If requested (by `save_paths=true`),
 `out.X[i][k]` contains the simulated state corresponding to `out.Theta[:,i]`
 at time `k`.
"""
function adaptive_pg(theta0::ParamT, prior::Function,
    state::SMCState, n::Int; b::Int=Int(ceil(0.1n)), thin::Int=1,
    save_paths::Bool=false, acc_opt::FT=FT(0.234), backward_sampling::Bool=true,
    show_progress::Real=false, algorithm=:ram) where {FT <: AbstractFloat, ParamT<:AbstractVector{FT}}

    if isnothing(state.lM)
        error("SMC state `state` must supply log-transition density!")
    end
    io = state.io; model = state.model; lM = state.lM
    set_param! = state.set_param!

    ref = [model.particle() for i=1:io.n]
    # Run the particle filter for HMM w/ param theta0:
    set_param!(io.internal.particleScratch, theta0)
    smc!(model, io)
    # ... and initialise reference trajectory to a sample from it:
    SequentialMonteCarlo.pickParticle!(ref, io)

    # Initialise random walk Metropolis state
    r = RWMState(theta0)
    # Initialise adaptation
    if algorithm == :ram
        s = RobustAdaptiveMetropolis(theta0, acc_opt)
    elseif algorithm == :aswam
        s = AdaptiveScalingWithinAdaptiveMetropolis(theta0, acc_opt)
    else
        error("Unknown algorithm: ", algorithm)
    end
    # Allocate space for simulated parameters
    nsim = Int(floor((n-b)/thin))
    Theta = zeros(FT, length(theta0), nsim)
    acc = zero(FT)
    if save_paths
        X = [[model.particle() for i=1:io.n] for j=1:nsim]
    else
        X = Missing
    end
    if show_progress > 0
        progress = Progress(n, Float64(show_progress))
    else
        progress = NullProgress()
    end
    for k = 1:n
        # Propose parameter update:
        ###########################
        draw!(r, s); pr_ = prior(r.y)
        if pr_ > -Inf
            p = prior(r.x) + _reference_logLik(ref, model.lG, lM, io.internal.particleScratch)
            set_param!(io.internal.particleScratch, r.y)
            p_ = pr_ + _reference_logLik(ref, model.lG, lM, io.internal.particleScratch)
            #alpha = min(one(FT), exp(p_ - p))
            alpha = _accept_prob(p, p_, FT)
            if rand() <= alpha
                accept!(r)
                acc += 1
            else
                set_param!(io.internal.particleScratch, r.x)
            end
        else # Early rejection if prior fails:
            alpha = zero(FT)
        end
        adapt!(s, r, alpha, k)

        # State update with CPF (& optional backward sampling)
        ############################
        csmc!(model, io, ref, ref)
        if backward_sampling
            SequentialMonteCarlo.pickParticleBS!(ref, io, lM)
        end

        @inbounds if k>b && rem(k-b, thin)==0
            i = Int((k-b)/thin)
            Theta[:,i] = r.x
            if save_paths
                SequentialMonteCarlo._copyParticles!(X[i], ref)
            end
        end
        next!(progress)
    end
    (Theta=Theta, acc=acc/n, X=X, adapt_state=s, theta0=theta0)
end
