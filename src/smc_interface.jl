# The default (void) interface to SMC backend

# This is the exported symbol for SMC state
function SMCState end

# Return (conditional) log-likelihood of the reference path 
# (used with particle Gibbs)
_reference_log_likelihood(state) = error("SMC state type $(typeof(state)) must have method _reference_log_likelihood")

# Change model parameters to correspond theta
_set_model_param!(state, theta) = error("SMC state type $(typeof(state)) must have method _set_model_param!")

# Run one sweep of SMC
_run_smc!(state) = error("SMC state type $(typeof(state)) must have method _run_smc!")

# Model log-likelihood estimate with current parameters
# (used with particle marginal Metropolis-Hastings)
_log_likelihood(state) = error("SMC state type $(typeof(state)) must have method _log_likelihood!")

# Pick one path from SMC using ancestor tracing
_pick_particle!(state) = error("SMC state type $(typeof(state)) must have method _pick_particle!")

# Save reference state
_save_reference!(state) = error("SMC state type $(typeof(state)) must have method _save_reference!")

# Initialise storage for paths
_init_path_storage(state, nsim) = error("SMC state type $(typeof(state)) must have method _init_path_storage")

# In-place copy reference path to output
_copy_reference!(out, state) = error("SMC state type $(typeof(state)) must have method _copy_reference!")

# Run one iteration of conditional SMC with ancestor tracing or backward sampling
_run_csmc!(state, backward_sampling) = error("SMC state type $(typeof(state)) must have method _run_csmc!")

# Custom action performed after each iteration
_post_iteration_hook!(state, iteration) = error("SMC state type $(typeof(state)) must have method _set_model_param!")
