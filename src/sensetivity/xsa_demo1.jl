# cite https://hess.copernicus.org/articles/24/5835/2020/
using HydroModels
using Distributions
using GlobalSensitivity
using QuasiMonteCarlo
using Plots

@variables A B C X
@variables A1 A2 B1 B2 B3 C1 C2
@parameters x1 x2 x3 x4 x5 x6 x7
@parameters a1 a2 b1 b2 b3 c1 c2

bucket_1 = @hydrobucket :disjoint begin
    fluxes = begin
        @hydroflux A1 ~ sin(x1)
        @hydroflux A2 ~ 1.0 + x1
        @hydroflux B1 ~ 1.0 + 2 * x2^4
        @hydroflux B2 ~ 1.0 + x3^2
        @hydroflux B3 ~ x5 + x4
        @hydroflux C1 ~ sin(x6)^2
        @hydroflux C2 ~ 1 + x7^4
        @hydroflux A ~ A1 * a1 + A2 * a2
        @hydroflux B ~ B1 * b1 + B2 * b2 + B3 * b3
        @hydroflux C ~ C1 * c1 + C2 * c2
        @hydroflux X ~ A * B + C
    end
end

params_ranges = [[-π, π] for _ in 1:7]
random_param_values = map(params_ranges) do params_range
    rand(Uniform(params_range[1], params_range[2]))
end
randam_init_params = NamedTuple{(:x1, :x2, :x3, :x4, :x5, :x6, :x7)}(random_param_values)
random_init_weights = NamedTuple{(:a1, :a2, :b1, :b2, :b3, :c1, :c2)}(rand(7))
params = ComponentVector(merge(randam_init_params, random_init_weights))
ps_axes = getaxes(ComponentVector(params=params))
output = bucket_1(rand(14, 1), ComponentVector(params=params))

function temp_func(p)
    output = bucket_1(rand(1, 1), ComponentVector(p, ps_axes))[end]
    return output
end

merge_params_ranges = vcat(params_ranges, [[0.0, 1.0] for _ in 1:7])
sampler = SobolSample()
design_matrices = QuasiMonteCarlo.generate_design_matrices(
    1000, [p[1] for p in merge_params_ranges],
    [p[2] for p in merge_params_ranges], sampler
)
Sobol(order=[0, 1, 2, 3, 4, 5])
design_matrices[1]
reg_sens = gsa(temp_func, RegressionGSA(true), merge_params_ranges; samples=1000)
partial_corr = reg_sens.partial_correlation
stand_regr = reg_sens.standard_regression

plot(partial_corr[1, :])
plot!(stand_regr[1, :])


