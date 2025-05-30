#* build UH-based Model
@variables q q_routed
@parameters lag

uh1 = HydroModels.@unithydro :maxbas_uh begin
    uh_func = begin
        lag => (t / lag)^2.5
    end
    uh_vars = [q]
    configs = (solvetype=:SPARSE, outputs=[q_routed])
end

dpl_hbv_uh = HydroModel(name=:rapid_hbv, components=[dpl_hbv_model.components..., uh1])
sum_dpl_hbv_uh = HydroWrapper(dpl_hbv_uh, post=(output) -> sum(output, dims=2))

#* build Grid-Based Model
@variables s_river q_routed
@parameters lag
grid_basin_info = load("data/hanjiang/grid_basin_info.jld2")
flwdir_matrix = grid_basin_info["flwdir_matrix"]
index_info = collect.(grid_basin_info["index_info"])
grid_route = @hydroroute begin
    fluxes = begin
        @hydroflux q_routed ~ s_river / (1 + lag)
    end
    dfluxes = begin
        @stateflux s_river ~ q - q_routed
    end
    aggr_func = HydroModels.build_aggr_func(flwdir_matrix, index_info)
end
dpl_hbv_grid = HydroModel(name=:grid_hbv, components=[dpl_hbv_model.components..., grid_route])

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

