# Very simple implementation of some adaptive MCMC algorithms.
module AdaptiveParticleMCMC

export pg_ram, pmmh_am

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
