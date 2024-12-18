using CSV
using DataFrames
using ComponentArrays
using BenchmarkTools
using Optimization
using OptimizationBBO
using JLD2
using Plots
using HydroModels
using HydroModelTools

include("../models/HBV.jl")
TOTAL_AREA = 15230.3
meteo_df = CSV.read("src/distributed/data/meteo_mean.csv", DataFrame)
flow_df = CSV.read("src/distributed/data/flow.csv", DataFrame)
flow_vec = @. flow_df[:, "FLOW"] * 24 * 3600 / (TOTAL_AREA * 1e6) * 1e3
input = (prcp=meteo_df[:, "mean_prec"], pet=meteo_df[:, "mean_pet"], temp=meteo_df[:, "mean_temp"])
input_mat = Matrix(reduce(hcat, [input[nm] for nm in HydroModels.get_input_names(hbv_model)])')
target = (q=flow_vec,)

params = ComponentVector(TT=0.0, CFMAX=5.0, CWH=0.1, CFR=0.05, FC=200.0, LP=0.6, BETA=3.0, k0=0.06, k1=0.2, k2=0.1, PPERC=2, UZL=10)
init_states = ComponentVector(suz=0.0, slz=0.0, soilwater=0.0, meltwater=0.0, snowpack=0.0)
pas = ComponentVector(params=params, initstates=init_states)

# hbv_optimizer = HydroOptimizer(component=hbv_model, solve_alg=BBO_adaptive_de_rand_1_bin_radiuslimited(), maxiters=1000, warmup=100)

# tunable_pas = ComponentVector(params=params)
# const_pas = ComponentVector(initstates=init_states)
# lower_bounds = [-1.5, 1, 0.0, 0.0, 50.0, 0.3, 1.0, 0.05, 0.01, 0.001, 0.0, 0.0]
# upper_bounds = [1.2, 8.0, 0.2, 0.1, 500.0, 1.0, 6.0, 0.5, 0.3, 0.15, 3.0, 70.0]
# hbv_opt_params, loss_df = hbv_optimizer(
#     [input],
#     [target],
#     tunable_pas=tunable_pas,
#     const_pas=const_pas,
#     lb=lower_bounds,
#     ub=upper_bounds,
#     return_loss_df=true
# )
# save("tutorials/distributed/cache/hbv_opt.jld2", "opt_params", hbv_opt_params)

hbv_opt_params = load("src/distributed/cache/hbv_opt.jld2", "opt_params")
new_output = hbv_model(input_mat, hbv_opt_params, config=(solver=ODESolver(),))
plot(new_output[end,:], label="simulated")
plot!(target.q, label="observed")