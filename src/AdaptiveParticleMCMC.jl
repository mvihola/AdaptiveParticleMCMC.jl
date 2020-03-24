# Very simple implementation of some adaptive MCMC algorithms.
module AdaptiveParticleMCMC

export adaptive_pg, adaptive_pmmh, SMCState

using Random, SequentialMonteCarlo, ProgressMeter, AdaptiveMCMC

include("types.jl") # Data types
include("pmmh.jl") # Particle marginal Metropolis-Hastings
include("pg.jl")   # Particle Gibbs

end # module
