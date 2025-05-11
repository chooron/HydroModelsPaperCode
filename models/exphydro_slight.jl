using HydroModels
using HydroModels: step_func, @variables, @parameters
# define variables and parameters
@variables temp lday pet prcp snowfall rainfall snowpack melt
@variables soilwater pet evap baseflow surfaceflow flow rainfall
@parameters Tmin Tmax Df Smax Qmax f

# define model components
bucket = @hydrobucket :exphydro begin
    fluxes = begin
        @hydroflux snowfall ~ step_func(Tmin - temp) * prcp
        @hydroflux rainfall ~ step_func(temp - Tmin) * prcp
        @hydroflux melt ~ step_func(temp - Tmax) * step_func(snowpack) * min(snowpack, Df * (temp - Tmax))
        @hydroflux pet ~ 29.8 * lday * 24 * 0.611 * exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
        @hydroflux evap ~ step_func(soilwater) * pet * min(1.0, soilwater / Smax)
        @hydroflux baseflow ~ step_func(soilwater) * Qmax * exp(-f * (max(0.0, Smax - soilwater)))
        @hydroflux surfaceflow ~ max(0.0, soilwater - Smax)
        @hydroflux flow ~ baseflow + surfaceflow
    end
    dfluxes = begin
        @stateflux snowpack ~ snowfall - melt
        @stateflux soilwater ~ (rainfall + melt) - (evap + flow)
    end
end

exphydro_model = @hydromodel :exphydro begin
    bucket
end

export exphydro_model, bucket
