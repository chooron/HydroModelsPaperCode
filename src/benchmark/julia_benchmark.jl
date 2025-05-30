using OrdinaryDiffEq
using ForwardDiff
# using Zygote # Kept commented as in original
using CSV
using DataFrames
using Statistics
using DataInterpolations
using ComponentArrays
using BenchmarkTools
using DifferentiationInterface
using SciMLSensitivity
# 读取数据
file_path = "data/exphydro/01013500.csv" # Please ensure this path is correct
data = CSV.File(file_path)
df = DataFrame(data)
ts = collect(1:1000)  # 使用500条数据
input = (lday=df[ts, "dayl(day)"], temp=df[ts, "tmean(C)"], prcp=df[ts, "prcp(mm/day)"])
q_obs = df[ts, "flow(mm)"]

# Create interpolation objects
# These interpolations will use the time `t` from the ODE solver directly.
temp_interp = LinearInterpolation(input.temp, ts)
prcp_interp = LinearInterpolation(input.prcp, ts)
lday_interp = LinearInterpolation(input.lday, ts)

# Exphydro模型参数 (numeric part, to be optimized)
p_numeric = ComponentVector(
    f=0.01674478, Smax=1709.461015, Qmax=18.46996175,
    Df=2.674548848, Tmax=0.175739196, Tmin=-2.092959084
)

# 定义模型函数
function exphydro_model!(du, u, p, t)
    # 状态变量
    snowpack, soilwater = u
    
    # Unpack numeric parameters
    Tmin = p.Tmin
    Tmax = p.Tmax
    Df = p.Df
    Smax = p.Smax
    Qmax = p.Qmax
    f = p.f
    
    # Get interpolated input variables for the current time t
    # The solver provides `t` as a float, which DataInterpolations handles.
    temp = temp_interp(t)
    prcp = prcp_interp(t)
    lday = lday_interp(t)
    
    # 计算PET
    pet = 29.8 * lday * 24 * 0.611 * exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)
    
    # 降雪和降雨
    snowfall = (temp <= Tmin) * prcp
    rainfall = (temp > Tmin) * prcp
    
    # 融雪
    melt = (temp > Tmax) * (snowpack > 0) * min(snowpack, Df * (temp - Tmax))
    
    # 土壤水分过程
    evap = (soilwater > 0) * pet * min(1.0, soilwater / Smax) # Using unpacked Smax
    baseflow = (soilwater > 0) * Qmax * exp(-f * max(0.0, Smax - soilwater)) # Using unpacked Qmax, f, Smax
    surfaceflow = max(0.0, soilwater - Smax) # Using unpacked Smax
    
    # 状态方程
    du[1] = snowfall - melt  # 雪包变化
    du[2] = (rainfall + melt) - (evap + baseflow + surfaceflow)  # 土壤水分变化
end

# 初始条件
u0 = [0.0, 0.0]  # 初始雪包和土壤水分

# 时间范围
tspan = (Float64(ts[1]), Float64(ts[end])) # Ensure tspan is Float for solver, matching interpolation range

# 求解ODE (initial run with p_ode)
prob = ODEProblem(exphydro_model!, u0, tspan, p_numeric) # Use the combined p_ode
sol = solve(prob, Tsit5(), saveat=1, abstol=1e-3, reltol=1e-3, sensealg=ForwardDiffSensitivity()) # saveat matches the original integer time steps

# 提取结果
# Ensure sol.t aligns with ts for direct indexing if necessary, saveat=1 should achieve this.
q_sim = [sol[2, i] for i in 1:length(ts)] # 只取土壤水分作为输出

# 计算Nash-Sutcliffe效率系数
function nse(q_obs_nse, q_sim_nse)
    return 1 - sum((q_obs_nse .- q_sim_nse).^2) / sum((q_obs_nse .- mean(q_obs_nse)).^2)
end

# 计算目标函数（负NSE）
# This function takes only the numeric parameters that are being optimized
function objective(current_p_numeric)
    # u0 and tspan are from the outer scope
    prob_opt = ODEProblem(exphydro_model!, u0, tspan, current_p_numeric)
    Qmax = current_p_numeric.Qmax
    f = current_p_numeric.f
    Smax = current_p_numeric.Smax
    # Added sensealg for ForwardDiff compatibility
    sol_opt = solve(prob_opt, Tsit5(), saveat=1, sensealg=ForwardDiffSensitivity())

    soilwater_vec = sol_opt[2,:]

    baseflow_vec = @. (soilwater_vec > 0) * Qmax * exp(-f * max(0.0, Smax - soilwater_vec)) # Using unpacked Qmax, f, Smax
    surfaceflow_vec = @. max(0.0, soilwater_vec - Smax) # Using unpacked Smax
    q_sim_opt = baseflow_vec .+ surfaceflow_vec
    
    return q_sim_opt
end

# 使用ForwardDiff计算梯度
# We pass p_numeric (the optimizable part) to the objective function for gradient calculation
@btime objective(p_numeric)
@btime value_and_gradient((p) -> objective(p) |> sum, AutoForwardDiff(), p_numeric)
@btime value_and_gradient((p) -> objective(p) |> sum, AutoZygote(), p_numeric)

# # 使用Zygote计算梯度 (kept commented)
# grad_zygote = Zygote.gradient(objective, p_numeric)[1] # p_numeric is the argument to objective
