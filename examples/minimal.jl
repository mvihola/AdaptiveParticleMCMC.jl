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
# SequentialMonteCarlo data types
state = SMCState(T, N, MyParticle, MyParam, set_param!, lG, M!, lM)
# Run the algorithms
out_pmmh = adaptive_pmmh([0.0], test_prior, state, n)
out_pg = adaptive_pg([0.0], test_prior, state, n)
out_pg_aswam = adaptive_pg([0.0], test_prior, state, n; algorithm=:aswam)
