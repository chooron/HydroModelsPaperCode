using Lux
using HydroModels
using HydroModels: @neuralflux, @stateflux, @hydroflux, @hydromodel, @hydrobucket, step_func

function LSTMCompact(in_dims, hidden_dims, out_dims)
    lstm_cell = LSTMCell(in_dims => hidden_dims)
    classifier = Dense(hidden_dims => out_dims, sigmoid)
    return @compact(; lstm_cell, classifier) do x::AbstractArray{T,2} where {T}
        x = reshape(x, size(x)..., 1)
        x_init, x_rest = Iterators.peel(LuxOps.eachslice(x, Val(2)))
        y, carry = lstm_cell(x_init)
        output = [vec(classifier(y))]
        for x in x_rest
            y, carry = lstm_cell((x, carry))
            output = vcat(output, [vec(classifier(y))])
        end
        @return reduce(hcat, output)
    end
end

@variables soilwater snowpack meltwater suz slz
@variables prcp pet temp
@variables rainfall snowfall melt refreeze infil excess recharge evap q0 q1 q2 q perc
@variables BETA GAMMA
@parameters TT CFMAX CFR CWH LP FC PPERC UZL k0 k1 k2 kp

#* parameters estimate by LSTM
params_nn = Lux.Chain(LSTMCompact(3, 10, 2), name=:pnn)
nn_flux = @neuralflux [BETA, GAMMA] ~ params_nn([prcp, temp, pet])

split_flux = @hydroflux begin
    snowfall ~ step_func(TT - temp) * prcp
    rainfall ~ step_func(temp - TT) * prcp
end
snow_bucket = @hydrobucket begin
    fluxes = begin
        @hydroflux melt ~ min(snowpack, max(0.0, temp - TT) * CFMAX)
        @hydroflux refreeze ~ min(max((TT - temp), 0.0) * CFR * CFMAX, meltwater)
        @hydroflux infil ~ max(0.0, meltwater - snowpack * CWH)
    end
    dfluxes = begin
        @stateflux snowpack ~ snowfall + refreeze - melt
        @stateflux meltwater ~ melt - (refreeze + infil)
    end
end
soil_bucket = @hydrobucket begin
    fluxes = begin
        @hydroflux recharge ~ (rainfall + infil) * clamp(max(0.0, soilwater / FC)^(BETA * 5 + 1), 0, 1)
        @hydroflux excess ~ max(soilwater - FC, 0.0)
        @hydroflux evap ~ clamp(max(0.0, soilwater / (LP * FC))^(GAMMA + 1), 0, 1) * pet
    end
    dfluxes = begin
        @stateflux soilwater ~ (rainfall + infil) - (recharge + excess + evap)
    end
end
zone_bucket = @hydrobucket begin
    fluxes = begin
        @hydroflux perc ~ suz * PPERC
        @hydroflux q0 ~ max(0.0, suz - UZL) * k0
        @hydroflux q1 ~ suz * k1
        @hydroflux q2 ~ slz * k2
        @hydroflux q ~ q0 + q1 + q2
    end
    dfluxes = begin
        @stateflux suz ~ recharge + excess - perc - q0 - q1
        @stateflux slz ~ perc - q2
    end
end
dpl_hbv_model = @hydromodel begin
    nn_flux
    split_flux
    snow_bucket
    soil_bucket
    zone_bucket
end

