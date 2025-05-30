using CSV, DataFrames, JLD2, Dates
using Lux, StableRNGs, ComponentArrays
using DataInterpolations, SciMLSensitivity, OrdinaryDiffEq
using OptimizationBBO, Optimization
using Statistics
using Plots
using HydroModels
using Zygote

include("../../models/HBV.jl")
include("../../models/loss_func.jl")
# build model based on HBV and rapid module
HRU_AREA = (0.1 * 111)^2
TOTAL_AREA = 15230.3

@variables s_river q_routed
@parameters lag
grid_basin_info = load("data/hanjiang/grid_basin_info.jld2")
flwdir_matrix = grid_basin_info["flwdir_matrix"]
index_info = collect.(grid_basin_info["index_info"])
hru_num = length(index_info)

grid_route = @hydroroute begin
    fluxes = begin
        @hydroflux q_routed ~ s_river / (1 + lag)
    end
    dfluxes = begin
        @stateflux s_river ~ q - q_routed
    end
    aggr_func = HydroModels.build_aggr_func(flwdir_matrix, index_info)
end

hbv_grid = HydroModel(name=:rapid_hbv, components=[hbv_model.components..., grid_route])
# load data
select_idx = 1:14610
time_idx = collect(1:length(select_idx))
flow_df = CSV.read("data/hanjiang/flow.csv", DataFrame)
flow_vec = flow_df[select_idx, "FLOW"] * 24 * 3600 / (TOTAL_AREA * 1e6) * 1e3
grid_meto = load("data/hanjiang/grid_meto.jld2")
subbasin_pet_df = grid_meto["pet"][select_idx, :]
subbasin_prcp_df = grid_meto["prec"][select_idx, :]
subbasin_temp_df = grid_meto["temp"][select_idx, :]
base_pas = load("src/distributed/cache/hbv_opt.jld2", "opt_params")
input = map(2:length(names(subbasin_pet_df))) do id
    (prcp=subbasin_prcp_df[:, id], temp=subbasin_temp_df[:, id], pet=subbasin_pet_df[:, id])
end
input_arr = stack([stack(i, dims=1) for i in input], dims=2)
# generate multi-types parameters and initstates
merge_params = ComponentVector(base_pas[:params]; lag=0.1)
node_params = ComponentVector(params=NamedTuple{keys(merge_params)}(
    [repeat([merge_params[k]], hru_num) for k in keys(merge_params)]
))
merge_initstates = ComponentVector(base_pas[:initstates]; s_river=0.0)
node_initstates = ComponentVector(NamedTuple{keys(merge_initstates)}(
    [repeat([merge_initstates[k]], hru_num) for k in keys(merge_initstates)]
))

config = (timeidx=time_idx, solver=HydroModels.ODESolver())
hbv_grid_output = hbv_grid(input_arr, node_params, initstates=node_initstates, config=config)
output_105 = hbv_grid_output[:, 105, :]

