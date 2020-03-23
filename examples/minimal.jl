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
test_model = AdaptiveParticleMCMC.SMCModel(M!, lG, T, MyParticle, MyParam)
test_io = AdaptiveParticleMCMC.SMCIO{MyParticle,MyParam}(N, T, 1, true)
# Run the algorithms
out_pmmh = pmmh_am([0.0]), test_prior, set_param!, test_model, test_io, n)
out_pg = pg_ram([0.0], test_prior, set_param!, test_model, test_io, lM, n)
