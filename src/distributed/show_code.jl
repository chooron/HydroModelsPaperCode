#* build UH-based Model
@variables q q_routed
@parameters lag
uhfunc = UHFunction(:UH_1_HALF)
uh_route = UnitHydrograph([q]=>[q_routed], lag; uhfunc=uhfunc)
dpl_hbv_uh = HydroModel(name=:rapid_hbv, components=[dpl_hbv_model.components..., uh_route])
sum_dpl_hbv_uh = WeightSumComponentOutlet(dpl_hbv_uh, [q_routed])

#* build Grid-Based Model
HRU_AREA = (0.1 * 111)^2
@variables q s_river q_routed
@parameters lag
grid_basin_info = load("tutorials/distributed/data/grid_basin_info.jld2")
flwdir_matrix = grid_basin_info["flwdir_matrix"]
index_info = grid_basin_info["index_info"]
rflux = HydroFlux([q, s_river] => [q_routed], [lag], exprs=[s_river / (1 + lag) + q])
grid_route = GridRoute(rfunc=rflux, rstate=s_river, flwdir=flwdir_matrix, positions=index_info)
dpl_hbv_grid = HydroModel(name=:rapid_hbv, components=[dpl_hbv_model.components..., grid_route])

#* build Rapid-Based Model
@variables q flow flow_routed
@parameters subarea
vector_basin_info = load("tutorials/distributed/data/vector_basin_info.jld2")
vector_dgraph = vector_basin_info["dg8"]
subbasin_areas = vector_basin_info["subbasin_area"]
subbasin_rivlen = vector_basin_info["riv_len_df"]
hybas_id = [Symbol("subbasin_$i") for i in vector_basin_info["hybas_id"]]
flow_convert_flux = HydroFlux([q] => [flow], [subarea], exprs=[q * 24 * 3600 / (subarea * 1e6) * 1e3])
rapid_route = RapidRoute([flow] => [flow_routed], network=vector_dgraph)
dpl_hbv_rapid = HydroModel(name=:rapid_hbv, components=[dpl_hbv_model.components..., flow_convert_flux, rapid_route])

