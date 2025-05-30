using CSV, DataFrames, JLD2, Dates
using Lux, StableRNGs, ComponentArrays
using DataInterpolations, SciMLSensitivity, OrdinaryDiffEq
using Statistics
using Plots
using HydroModels
using BenchmarkTools
using Zygote
using ForwardDiff

include("../../models/gr4j.jl")
# load data
file_path = "data/gr4j/sample.csv"
data = CSV.File(file_path);
df = DataFrame(data);
ts = collect(1:3600)
input = (Ep=df[ts, "pet"], P=df[ts, "prec"])
input_arr = stack(input[HydroModels.get_input_names(gr4j_model)], dims=1)
q_vec = df[ts, "qobs"]

pas = ComponentVector(params=ComponentVector(
    x1=50.0,
    x2=0.1,
    x3=20.0,
    x4=3.5
))
initstates = ComponentVector(S=0.0, R=0.0)
config = (solver=HydroModels.DiscreteSolver(), interp=LinearInterpolation)
@btime output = gr4j_model(input_arr, pas, initstates=initstates, config=config)