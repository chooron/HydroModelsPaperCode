
using GlobalSensitivity
using QuasiMonteCarlo

function ishi(X)
    A = 7
    B = 0.1
    sin(X[1]) + A * sin(X[2])^2 + B * X[3]^4 * sin(X[1])
end

n = 600000
lb = -ones(4) * π
ub = ones(4) * π
sampler = SobolSample()
A, B = QuasiMonteCarlo.generate_design_matrices(n, lb, ub, sampler)

res1 = gsa(ishi, Sobol(order = [0, 1, 2]), A, B)

res1.S1
res1.S2
res1.S3
res1.ST
