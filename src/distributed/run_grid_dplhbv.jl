using CSV
using DataFrames
using ComponentArrays
using BenchmarkTools
using NamedTupleTools
using Optimization
using OptimizationOptimisers
using JLD2
using Plots
using StableRNGs
using HydroModels
include("../models/dplHBV.jl")


# build model based on HBV and rapid module
HRU_AREA = (0.1 * 111)^2
TOTAL_AREA = 15230.3
@variables q s_river q_routed
@parameters lag
grid_basin_info = load("tutorials/distributed/data/grid_basin_info.jld2")
flwdir_matrix = grid_basin_info["flwdir_matrix"]
index_info = grid_basin_info["index_info"]
hru_names = [Symbol("subbasin_$i") for i in eachindex(index_info)]
rflux = HydroFlux([q, s_river] => [q_routed], [lag], exprs=[s_river / (1 + lag) + q])
grid_route = GridRoute(rfunc=rflux, rstate=s_river, flwdir=flwdir_matrix, positions=index_info)
dpl_hbv_grid = HydroModel(name=:rapid_hbv, components=[dpl_hbv_model, grid_route])
# load data
select_idx = 2000:4000
time_idx = collect(1:length(select_idx))
flow_df = CSV.read("tutorials/distributed/data/flow.csv", DataFrame)
flow_vec = flow_df[select_idx, "FLOW"] * 24 * 3600 / (TOTAL_AREA * 1e6) * 1e3
grid_meto = load("tutorials/distributed/data/grid_meto.jld2")
subbasin_pet_df = grid_meto["pet"][select_idx, :]
subbasin_prcp_df = grid_meto["prec"][select_idx, :]
subbasin_temp_df = grid_meto["temp"][select_idx, :]
base_pas = load("tutorials/implements/save/dplHBV_opt.jld2", "opt_params")
input = map(2:length(names(subbasin_pet_df))) do id
    (prcp=subbasin_prcp_df[:, id], temp=subbasin_temp_df[:, id], pet=subbasin_pet_df[:, id])
end

# generate multi-types parameters and initstates
psnn_ps, psnn_st = Lux.setup(StableRNG(123), params_nn)
psnn_ps_ca = ComponentVector(psnn_ps)
psnn_ps_vec = Vector(ComponentVector(psnn_ps))

node_params = NamedTuple{Tuple(hru_names)}(repeat([ComponentVector(base_pas[:params]; lag=0.1)], length(hru_names)))
node_initstates = NamedTuple{Tuple(hru_names)}(repeat([ComponentVector(base_pas[:initstates]; s_river=0.0)], length(hru_names)))
node_pas = ComponentVector(params=node_params, initstates=node_initstates, nn=(pnn=psnn_ps_vec,))

# input_arr = permutedims(reduce((c1, c2) -> cat(c1, c2, dims=3), [reduce(hcat, inp)' for inp in input]), (1, 3, 2))
hbv_grid_output = dpl_hbv_grid(input, node_pas, convert_to_ntp=true, config=(ptypes=hru_names, stypes=hru_names, timeidx=time_idx))
output_105 = hbv_grid_output[105]
plot(output_105.q_routed[1:100], label="sim")
plot!(flow_vec[1:100], label="obs")

# outlet_wrapper = HydroModels.ComputeComponentOutlet(hbv_rapid, 1)
# hbv_rapid_opt = GradOptimizer(component=outlet_wrapper, maxiters=100, adtype=AutoForwardDiff(), solve_alg=Adam(1e-2))
# tunable_pas = ComponentVector(params=node_params)
# const_pas = ComponentVector(initstates=node_initstates, params=convert_params)
# opt_params, loss_df = hbv_rapid_opt(
#     [input],
#     [(flow_routed=flow_vec,)],
#     tunable_pas=tunable_pas,
#     const_pas=const_pas,
#     config=[configs],
#     return_loss_df=true
# )
