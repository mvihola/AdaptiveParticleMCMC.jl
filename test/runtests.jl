using Test, Random, RNGPool, Statistics, AdaptiveParticleMCMC, SequentialMonteCarlo

Random.seed!(1234) # Adaptive RWM uses these
setRNGs(1234)      # SequentialMonteCarlo uses these

mutable struct TestParticle s::Float64 end
TestParticle() = TestParticle(0.0)
mutable struct TestParam σ::Float64 end
TestParam() = TestParam(1.0)
function M!(x, rng, k, x_prev, scratch)
    if k == 1
        x.s = randn(rng)*scratch.θ.σ
    else
        x.s = x_prev.s + randn(rng)*scratch.θ.σ
    end
end
function lM(k, x_prev, x, scratch)
    if k == 1
        return -log(scratch.θ.σ) - .5(x.s/scratch.θ.σ)^2
    else
        return -log(scratch.θ.σ) - .5((x.s-x_prev.s)/scratch.θ.σ)^2
    end
end
lG(k, x, scratch) = -log(scratch.θ.σ) - .5(x.s/scratch.θ.σ)^2

test_prior(theta) = -.5*mapreduce(t->t^2, +, theta)
struct TestScratch θ::TestParam end
function set_param!(scratch, theta_) scratch.θ.σ = exp(theta_[1]); nothing end                  # (-∞,∞) → (0,∞)end

test_parameters = TestParam()
TestScratch() = TestScratch(test_parameters)
N=16; T=10; n=10000
test_model = SMCModel(M!, lG, T, TestParticle, TestScratch)
test_io = SMCIO{TestParticle,TestScratch}(N, T, 1, true)
update!(theta_) = set_param!(test_parameters, theta_)
theta0 = zeros(1)
out_pmmh = pmmh_am(theta0, test_prior, set_param!, test_model, test_io, n; save_paths=true);
out_pg = pg_ram(theta0, test_prior, set_param!, test_model, test_io, lM, n; save_paths=true);
function test_stats(out)
    X_ = [out.X[i][j].s for i=1:length(out.X), j=1:T]
    mX = mapslices(mean, X_, dims=1)
    mT = mean(out.Theta)
    sT = std(out.Theta)
    mX, mT, sT
end
function test_output(out, tolT=0.2, tolX=0.001)
    mX, mT, sT = test_stats(out)
    abs(mT + 10) < tolT &&
    abs(sT - 0.99) < tolT &&
    all(abs.(mX) .< tolX)
end
@test test_output(out_pmmh)
@test test_output(out_pg)
