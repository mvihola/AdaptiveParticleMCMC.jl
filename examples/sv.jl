# This example requires that also the packages Distributions, LabelledArrays,
# CSV and Statistics are installed; install by
# using Pkg; Pkg.add("CSV"); Pkg.add("Distributions"); Pkg.add("LabelledArrays"); Pkg.add("Statistics"); Pkg.add("DataFrames")
using AdaptiveParticleMCMC, LabelledArrays, Distributions, CSV, Statistics, DataFrames

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
data = CSV.read(joinpath(@__DIR__,"sp500post2000.csv"), DataFrame)
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
# This function sets the model parameters based on sampled (transformed)
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

# Set up the SMC state
state = SMCState(T, N, SVParticle, SVScratch, set_param!, lG_sv, M_ar1!, lM_ar1)

# Initial (transformed) parameter vector
theta0 = LVector(logit_̢rho=0.0, log_sigma=0.0, beta=0.0)
# Particle marginal Metropolis-Hastings with Adaptive Metropolis
out_pmmh = adaptive_pmmh(theta0, prior, state, n;
  thin=100, show_progress=2, save_paths=true);

# Particle Gibbs with Robust Adaptive Metropolis
out_pg = adaptive_pg(theta0, prior, state, n;
  thin=100, show_progress=2, save_paths=true);

using StatsPlots, Statistics
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
