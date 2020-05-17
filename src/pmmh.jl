"""
   out = adaptive_pmmh(theta0, prior, state, n; kwargs...)

Generic particle marginal Metropolis-Hastings (PMMH) with adaptive
Metropolis proposal on the parameters.

# Arguments
- `theta0`: Initial parameter vector
- `prior`: Function returning prior log density values for parameters
- `state`: `SMCState` data structure.
- `n`: Number of iterations

# Optional keyword arguments
- `b`: Burn-in length; default `0.1n`
- `thin`: Thinning interval; default `1` (no thinning)
- `save_paths`: Whether to save (samples of) latent trajectories; default `false`
- `show_progress`: Show progress every `show_progress` seconds (default: `false`).

The hidden Markov model and SMC state is defined in `state`; see `?SMCState`.

The output `out` is a `NamedTuple` containing the simulation output: `out.acc`
is the mean acceptance rate, `out.Theta` contains the simulated parameter
values (each column is a parameter vector). If requested (by `save_paths=true`),
`out.X[i][k]` contains the simulated state corresponding to `out.Theta[:,i]`
at time `k`.
"""
function adaptive_pmmh(theta0::ParamT, prior::Function, state, n::Int;
    b::Int=Int(ceil(0.1n)), thin::Int=1, save_paths::Bool=false,
    show_progress::Real=false) where {FT<:AbstractFloat, ParamT<:AbstractVector{FT}}

    # Run the particle filter for HMM w/ param theta0:
    _set_model_param!(state, theta0)
    _run_smc!(state)
    # The initial log-target density:
    p = _log_likelihood(state) + prior(theta0)

    # Initialise random walk Metropolis state
    r = RWMState(theta0)
    # Initialise adaptive Metropolis state
    s = AdaptiveMetropolis(theta0)
    # Allocate space for simulated parameters
    nsim = Int(floor((n-b)/thin))
    Theta = zeros(FT, length(theta0), nsim)
    D = zeros(nsim)
    acc = zero(FT)
    if save_paths
        _pick_particle!(state)
        X = _init_path_storage(state, nsim)
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
        draw!(r, s); pr_ = prior(r.y)
        if pr_ > -Inf
            _set_model_param!(state, r.y)
            _run_smc!(state)
            p_ = _log_likelihood(state) + pr_
            #alpha = min(one(FT), exp(p_ - p))
            alpha = _accept_prob(p, p_, FT)
            if rand() <= alpha
                accept!(r)
                p = p_
                acc += 1
                save_paths && _pick_particle!(state)
            end
        else
            alpha = 0.0
        end
        adapt!(s, r, alpha, k)

        # Save output:
        @inbounds if k>b && rem(k-b, thin)==0
            i = Int((k-b)/thin)
            Theta[:,i] = r.x
            D[i] = p
            if save_paths
                _copy_reference!(X[i], state)
            end
        end
        next!(progress)
    end
    (Theta=Theta, D=D, acc=acc/n, X=X, am=s, theta0=theta0)
end
