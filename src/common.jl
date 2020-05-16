# Augment a null progress meter which disables it
import ProgressMeter.next!
struct NullProgress <: ProgressMeter.AbstractProgress
    NullProgress()=new()
end
next!(::NullProgress) = nothing


# Function which is essentially min(1, exp(p_ - p)),  but handles
# exceptional cases gracefully
@inline function _accept_prob(p, p_, FT)
    if !isfinite(p_) # p_ is not finite => force reject
        return zero(FT)
    elseif !isfinite(p) # p_ is finite but p not => force accept
        return one(FT)
    else # both finite, the standard case:
        return (p_ >= p) ? one(FT) : exp(p_ - p)
    end
end
