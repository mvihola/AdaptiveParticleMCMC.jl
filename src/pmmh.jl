"""
   out = pmmh_am(theta0, prior, set!, model, io, n; kwargs...)

Generic particle marginal Metropolis-Hastings (PMMH) with adaptive
Metropolis proposal on the parameters.

# Arguments
- `theta0`: Initial parameter vector
- `prior`: Function returning prior log density values for parameters
- `set_param!`: Function which updates the parameter value of the model
- `model`: `SequentialMonteCarlo.SMCModel` data structure
- `io`:  `SequentialMonteCarlo.SMCIO` data structure
- `n`: Number of iterations

# Optional keyword arguments
- `b`: Burn-in length; default `0.1n`
- `thin`: Thinning interval; default `1` (no thinning)
- `save_paths`: Whether to save (samples of) latent trajectories; default `false`
- `show_progress`: Show progress every `show_progress` seconds (default: `false`).

The hidden Markov model is defined in `model`, and the number of
particles (among other particle filter parameters) in `io`; see
the documentation of `SequentialMonteCarlo.jl`

The output `out` is a `NamedTuple` containing the simulation output: `out.acc`
is the mean acceptance rate, `out.Theta` contains the simulated parameter
values (each column is a parameter vector). If requested (by `save_paths=true`),
`out.X[i][k]` contains the simulated state corresponding to `out.Theta[:,i]`
at time `k`.
"""
function pmmh_am(theta0::ParamT, prior::Function,
    set_param!::Function, model::SMCModel, io::SMCIO, n::Int;
    b::Int=Int(ceil(0.1n)), thin::Int=1, save_paths::Bool=false,
    show_progress::Real=false) where {FT<:AbstractFloat, ParamT<:AbstractVector{FT}}

    # Run the particle filter for HMM w/ param theta0:
    set_param!(io.internal.particleScratch, theta0); smc!(model, io)
    # The initial log-target density:
    p = io.logZhats[end] + prior(theta0)

    # Initialise random walk Metropolis state
    r = RWMState(theta0)
    # Initialise adaptive Metropolis state
    s = AdaptiveMetropolis(theta0)
    # Allocate space for simulated parameters
    nsim = Int(floor((n-b)/thin))
    Theta = zeros(FT, length(theta0), nsim)
    D = zeros(nsim)
    acc = zero(FT)
    ref = [model.particle() for i=1:io.n]
    SequentialMonteCarlo.pickParticle!(ref, io)
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
        # PMMH: Draw proposal for parameter, run SMC:
        draw!(r, s)
        set_param!(io.internal.particleScratch, r.y); smc!(model, io)
        p_ = io.logZhats[end] + prior(r.y)
        alpha = min(one(FT), exp(p_ - p))
        if rand() <= alpha
            accept!(r)
            p = p_
            acc += 1
            SequentialMonteCarlo.pickParticle!(ref, io)
        end
        adapt!(s, r, alpha, k)

        # Save output:
        @inbounds if k>b && rem(k-b, thin)==0
            i = Int((k-b)/thin)
            Theta[:,i] = r.x
            D[i] = p
            if save_paths
                SequentialMonteCarlo._copyParticles!(X[i], ref)
            end
        end
        next!(progress)
    end
    (Theta=Theta, D=D, acc=acc/n, X=X, am=s, theta0=theta0)
end
