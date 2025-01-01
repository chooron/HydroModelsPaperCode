using CSV
using DataFrames
using Lux
using StableRNGs
using ComponentArrays
using DataInterpolations
using OrdinaryDiffEq
using Statistics
using BenchmarkTools
using Plots
using JLD2
using SciMLSensitivity
using HydroModelTools
using Zygote
using ModelingToolkit


include("E:\\JlCode\\HydroModels\\src\\HydroModels.jl")
HydroFlux = HydroModels.HydroFlux
StateFlux = HydroModels.StateFlux
NeuralFlux = HydroModels.NeuralFlux
HydroBucket = HydroModels.HydroBucket
HydroModel = HydroModels.HydroModel
include("../models/m50v2.jl")

# load data
data = CSV.read("data/exphydro/01013500.csv", DataFrame)
ts = collect(1:10000)
input = (lday=data[ts, "dayl(day)"], temp=data[ts, "tmean(C)"], prcp=data[ts, "prcp(mm/day)"])
flow_vec = data[ts, "flow(mm)"]
config = (solver=ODESolver(sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP()), 
kwargs=Dict(:dt=>1.0, :reltol=>1e-3, :abstol=>1e-3, :save_everystep=>false)), 
interp=LinearInterpolation, alg=BS3())

m50_opt_params = load("E:\\JlCode\\HydroModelsPaperCode\\src\\implements\\save\\m50_opt.jld2")["opt_params"]
input_mat = Matrix(reduce(hcat, collect(input[HydroModels.get_input_names(m50_model)]))')

new_params = ComponentVector(
    params=m50_opt_params.params[HydroModels.get_param_names(m50_model)],
    nns=m50_opt_params.nn[HydroModels.get_nn_names(m50_model)],
    initstates=m50_opt_params.initstates[HydroModels.get_state_names(m50_model)],
)
@btime m50_output = m50_ele(input_mat, new_params, config=config)
# m50_outputv2 = m50_model(input_mat, m50_opt_params, config=config)

function loss1(p)
    m50_output = m50_ele(input_mat, p, config=config)
    sum(m50_output[end,:] .- flow_vec)
end
# @btime gradient(loss1, new_params)
