# AdaptiveParticleMCMC.jl

Simple implementation of adaptive proposals within particle Markov chain Monte Carlo [(Andrieu, Doucet and Holenstein, 2010)](https://doi.org/10.1111/j.1467-9868.2009.00736.x), based on the [AdaptiveMCMC.jl](https://github.com/mvihola/AdaptiveMCMC.jl) and [SequentialMonteCarlo.jl](https://github.com/awllee/SequentialMonteCarlo.jl) packages.

The package implemnents the following combinations:

* Particle marginal Metropolis-Hastings with
Adaptive Metropolis covariance adaptation ([Haario, Saksman and Tamminen, 2001](https://projecteuclid.org/euclid.bj/1080222083), and [Andrieu and Moulines, 2006](http://dx.doi.org/10.1214/105051606000000286))
* Particle Gibbs with Robust adaptive Metropolis acceptance rate based
shape adaptation [(Vihola, 2012)](http://dx.doi.org/10.1007/s11222-011-9269-5) and
Adaptive scaling within Adaptive Metropolis [(e.g. Andrieu and Thoms,
2008)](https://doi.org/10.1007/s11222-008-9110-y)

These choices are discussed in
* M. Vihola. Ergonomic and reliable Bayesian inference with adaptive Markov chain Monte Carlo. In W. W. Piegrorsch, R. Levine,  H. H. Zhang, and T. C. M. Lee, editors, *Handbook of Computational Statistics and Data Science*, Wiley, to appear.

If you use this package in your work, please cite the publication above.

## Getting the package

```julia
using Pkg
Pkg.add(PackageSpec(url="https://github.com/mvihola/AdaptiveParticleMCMC.jl.git"))
```

## Quick start

```julia
using AdaptiveParticleMCMC
# 'Particle' and 'ParticleScratch' data types for SequentialMonteCarlo0
mutable struct MyParticle
    s::Float64
    MyParticle() = new(0.0)
end
mutable struct MyParam
    μ::Float64
    MyParam() = new(0.0)
end
# SequentialMonteCarlo M!, lM, lG:
M!(x, rng, k, x_prev, par) = (x.s = par.μ + randn(rng); nothing)
lM(k, x_prev, x, par) = -.5(x.s-par.μ)^2
lG(k, x, scratch) = -.5*x.s^2
# (Trivial) prior & parameter update function
test_prior(theta) = 1
set_param!(par, theta) = (par.μ = theta[1]; nothing)
# Test run: N particles, n iterations, time series length T
N=16; T=10; n=10000
# SequentialMonteCarlo data structures
state = SMCState(T, N, MyParticle, MyParam, set_param!, lG, M!, lM)
# Run the algorithms
out_pmmh = adaptive_pmmh([0.0], test_prior, state, n)
out_pg = adaptive_pg([0.0], test_prior, state, n)
```

## Simple stochastic volatility model

```julia
# This example requires that also the packages Distributions, LabelledArrays,
# and CSV are installed; install by
# using Pkg; Pkg.add("Statistics"); Pkg.add("CSV"); Pkg.add("Distributions"); Pkg.add("LabelledArrays"); Pkg.add("DataFrames")
using AdaptiveParticleMCMC, Statistics, LabelledArrays, Distributions, CSV, DataFrames

# Define the particle type for the model (here, latent is univariate AR(1))
mutable struct SVParticle
    s::Float64
    SVParticle() = new(0.0) # Void constructor required!
end

# Model parameters:
mutable struct SVParam
    ρ::Float64     # Latent AR(1) coefficient
    σ::Float64     # Latent AR(1) noise sd
    β::Float64     # Latent mean
    σ_s1::Float64  # Latent AR(1) stationary sd
end

# Monthly S&P 500 data (from https://datahub.io/core/s-and-p-500):
data = CSV.read(download("https://raw.githubusercontent.com/mvihola/AdaptiveParticleMCMC.jl/master/examples/sp500post2000.csv"), DataFrame)
sp500_data = diff(log.(data.SP500)) # Monthly log-returns
sp500_data .-= mean(sp500_data)     # Remove trend
# Initialise parameters
sv_par = SVParam(0.9,1.0,0.0,1.0)

# This will be the SequentialMonteCarlo "particle scratch", which
# will contain both model data & parameters, and which will be
# the 'scratch' argument of M_ar1!, lM_ar1, lG_sv
struct SVScratch
    par::SVParam
    y::Vector{Float64}
    SVScratch() = new(sv_par, sp500_data)
end

# Transition *simulator* of stationary zero-mean AR(1) with
# parameters (ρ, σ)
function M_ar1!(x, rng, k, x_prev, scratch)
    if k == 1
        x.s = rand(rng, Normal(0.0, scratch.par.σ_s1))
    else
        x.s = rand(rng, Normal(scratch.par.ρ*x_prev.s, scratch.par.σ))
    end
end

# Log transition *density* of stationary zero-mean AR(1) with parameters (ρ, σ)
# (only used with Particle Gibbs, not with Particle marginal Metropolis-Hastings)
function lM_ar1(k, x_prev, x, scratch)
    if k == 1
        return logpdf(Normal(0.0, scratch.par.σ_s1), x.s)
    else
        return logpdf(Normal(scratch.par.ρ*x_prev.s, scratch.par.σ), x.s)
    end
end

# Potential (stochastic volatility observation log-density)
function lG_sv(k, x, scratch)
    s = exp(.5*(scratch.par.β + x.s))        # Observation sd
    logpdf(Normal(0.0, s), scratch.y[k])   # y[k] ~ N(0, s^2)
end

# We are sampling transformed parameters (logit(ρ), log(σ), β)
# The function set_param! sets the model parameters, that is, transforms
# parameters to the 'actual' model parameter values.
inv_logit(x) = 2.0/(1.0+exp(-x)) - 1.0 # (-∞,∞) → (-1,1)
function set_param!(scratch, θ)
    scratch.par.ρ = inv_logit(θ.logit_̢rho)
    scratch.par.σ = exp(θ.log_sigma)
    scratch.par.β = θ.beta
    scratch.par.σ_s1 = scratch.par.σ/sqrt(1.0 - scratch.par.ρ^2) # Stationary variance
end

# Normal prior for the transformed parameters
function prior(theta)
    (logpdf(Normal(6.0,5.0), theta.logit_̢rho)
    +logpdf(Normal(-1.0,5.0), theta.log_sigma)
    +logpdf(Normal(-6.0,5.0), theta.beta))
end

# Create data structures for SequentialMonteCarlo:
##################################################
T = length(sp500_data)
N = 64     # Number of particles
n = 40_000 # Number of PMCMC iterations

# Set up SMC state:
state = SMCState(T, N, SVParticle, SVScratch, set_param!, lG_sv, M_ar1!, lM_ar1)

# Initial (transformed) parameter vector
theta0 = LVector(logit_̢rho=0.0, log_sigma=0.0, beta=0.0)
# Particle marginal Metropolis-Hastings with Adaptive Metropolis
out_pmmh = adaptive_pmmh(theta0, prior, state, n;
  thin=100, show_progress=2, save_paths=true);

# Particle Gibbs with Robust Adaptive Metropolis
out_pg = adaptive_pg(theta0, prior, state, n;
  thin=100, show_progress=2, save_paths=true);

# The visualisation requires "StatsPlots" package; install by
# using Pkg; Pkg.add("StatsPlots")
using StatsPlots
function quantile_plot!(plt, S, p=0.95; x=1:size(S)[1])
    alpha = (1-p)/2
    qs = [alpha, 0.5, 1-alpha]
    Q = mapslices(x->quantile(x, qs), S, dims=2)
    plot!(plt, x, Q[:,2], ribbon=(Q[:,2]-Q[:,1], Q[:,3]-Q[:,2]), color=:black,
    legend=:false, fillalpha=0.2)
end
function show_out(out; title="")
    labels = [string.(typeof(out.theta0).parameters[4])...]
    p_theta = corrplot(out.Theta', size=(600,600), label=labels,
    title="Parameter posterior$title")
    # The volatilities:
    S = [out.X[j][i].s+out.Theta[3,j] for i=1:length(out.X[1]), j=1:length(out.X)]
    p_paths = plot(xlabel="Time", size=(600,800), ylabel="Log-volatility",
    legend=false, title="Latent posterior median, 50% and 95% credible intervals")
    quantile_plot!(p_paths, S, 0.95; x=data.Date[2:end])
    quantile_plot!(p_paths, S, 0.5; x=data.Date[2:end])
    p_data = plot(data.Date, log.(data.SP500), size=(600,800), ylabel="Log SP500", legend=false)
    plot(p_theta, p_paths, p_data, layout=grid(3,1, heights=[0.7,0.15,0.15]))
end
show_out(out_pg)
```
