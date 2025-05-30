using CSV, DataFrames, JLD2, Dates
using Lux, StableRNGs, ComponentArrays
using DataInterpolations, SciMLSensitivity, OrdinaryDiffEq
using Statistics
using Plots
using HydroModels
using BenchmarkTools
using Zygote, ForwardDiff, Enzyme
using DifferentiationInterface

include("../../models/exphydro.jl")
# load data
file_path = "data/exphydro/01013500.csv"
data = CSV.File(file_path);
df = DataFrame(data);
ts = collect(1:1000)
input = (lday=df[ts, "dayl(day)"], temp=df[ts, "tmean(C)"], prcp=df[ts, "prcp(mm/day)"])
input_arr = stack(input, dims=1)
q_vec = df[ts, "flow(mm)"]


pas = ComponentVector(params=ComponentVector(
    f=0.01674478, Smax=1709.461015, Qmax=18.46996175,
    Df=2.674548848, Tmax=0.175739196, Tmin=-2.092959084
))
initstates = ComponentVector(snowpack=0.0, soilwater=1300.0)
config = (solver=HydroModels.ODESolver(sensealg=ForwardDiffSensitivity()), interp=LinearInterpolation)

@btime output = exphydro_model(input_arr, pas, initstates=initstates, config=config)

f1 = p -> exphydro_model(input_arr, p, initstates=initstates, config=config)[end,:] |> sum
@btime value_and_gradient(f1, AutoForwardDiff(), pas)
@btime value_and_gradient(f1, AutoZygote(), pas)
# exphydro_grad_opt_params, loss_df = model_grad_opt(
#     [input], [(flow=q_vec,)],
#     tunable_pas=tunable_pas,
#     const_pas=const_pas,
#     config=[config],
#     return_loss_df=true
# )