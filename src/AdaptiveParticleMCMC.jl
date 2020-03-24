# Very simple implementation of some adaptive MCMC algorithms.
module AdaptiveParticleMCMC

export adaptive_pg, adaptive_pmmh

using Random, SequentialMonteCarlo, ProgressMeter, AdaptiveMCMC

# Augment a null progress meter which disables it
import ProgressMeter.next!
struct NullProgress <: ProgressMeter.AbstractProgress
    NullProgress()=new()
end
next!(::NullProgress) = nothing

include("pmmh.jl") # Particle marginal Metropolis-Hastings
include("pg.jl")   # Particle Gibbs

end # module
