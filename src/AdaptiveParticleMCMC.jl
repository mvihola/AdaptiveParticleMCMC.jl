# Very simple implementation of some adaptive MCMC algorithms.
module AdaptiveParticleMCMC

export adaptive_pg, adaptive_pmmh, SMCState

using Random, ProgressMeter, AdaptiveMCMC

include("common.jl") # Data types
include("smc_interface.jl") # Null interface to SMC backend
include("pmmh.jl") # Particle marginal Metropolis-Hastings
include("pg.jl")   # Particle Gibbs

end # module
