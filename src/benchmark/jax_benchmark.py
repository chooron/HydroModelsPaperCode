import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import diffrax
from diffrax import diffeqsolve, ODETerm, Tsit5, SaveAt, PIDController
import time
from typing import NamedTuple, Tuple
import numpy as np
from benchmark.utils.data_loader import load_hydro_data, get_data_path
from interpax import interp1d

# 定义模型参数
class ModelParams(NamedTuple):
    Tmin: float  # 最小温度阈值
    Tmax: float  # 最大温度阈值
    Df: float    # 融雪系数
    Smax: float  # 最大土壤含水量
    Qmax: float  # 最大基流
    f: float     # 基流衰减系数

# 定义模型状态
class ModelState(NamedTuple):
    snowpack: float    # 积雪量
    soilwater: float   # 土壤含水量

# 定义输入数据
class ModelInput(NamedTuple):
    temp: jnp.ndarray    # 温度
    lday: jnp.ndarray    # 日照时长
    prcp: jnp.ndarray    # 降水量

# 定义模型输出
class ModelOutput(NamedTuple):
    flow: float        # 总流量
    baseflow: float    # 基流
    surfaceflow: float # 地表径流
    evap: float        # 蒸发量
    melt: float        # 融雪量

def step_func(x: float) -> float:
    """阶跃函数"""
    return (jnp.tanh(5.0 * x) + 1.0) * 0.5

def calculate_pet(temp: float, lday: float) -> float:
    """计算潜在蒸散发"""
    return 29.8 * lday * 24 * 0.611 * jnp.exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)

def model_derivatives(t: float, state: ModelState, args: Tuple[ModelParams, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> ModelState:
    """模型导数函数"""
    params, temp_data, lday_data, prcp_data = args
    
    # 获取当前时间步的插值输入
    temp = interp1d(t, jnp.arange(1, len(temp_data) + 1), temp_data, method="cubic")
    lday = interp1d(t, jnp.arange(1, len(lday_data) + 1), lday_data, method="cubic")
    prcp = interp1d(t, jnp.arange(1, len(prcp_data) + 1), prcp_data, method="cubic")
    
    # 计算降雪和降雨
    snowfall = step_func(params.Tmin - temp) * prcp
    rainfall = step_func(temp - params.Tmin) * prcp
    
    # 计算融雪
    melt = step_func(temp - params.Tmax) * step_func(state.snowpack) * \
           jnp.minimum(state.snowpack, params.Df * (temp - params.Tmax))
    
    # 计算潜在蒸散发
    pet = calculate_pet(temp, lday)
    
    # 计算蒸发
    evap = step_func(state.soilwater) * pet * jnp.minimum(1.0, state.soilwater / params.Smax)
    
    # 计算基流
    baseflow = step_func(state.soilwater) * params.Qmax * \
              jnp.exp(-params.f * (jnp.maximum(0.0, params.Smax - state.soilwater)))
    
    # 计算地表径流
    surfaceflow = jnp.maximum(0.0, state.soilwater - params.Smax)
    
    # 计算总流量
    flow = baseflow + surfaceflow
    
    # 计算状态导数
    dsnowpack = jnp.maximum(snowfall - melt, -state.snowpack)
    dsoilwater = jnp.maximum((rainfall + melt) - (evap + flow), -state.soilwater)
    
    return ModelState(snowpack=dsnowpack, soilwater=dsoilwater)

def solve_model(params: ModelParams, initial_state: ModelState, 
                inputs: ModelInput, t_span: Tuple[float, float], 
                dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """求解模型"""
    # 创建ODE项
    term = ODETerm(model_derivatives)
    
    # 创建求解器
    solver = Tsit5()

    controller = PIDController(
        rtol=1e-3,
        atol=1e-3,
    )
    
    # 求解ODE
    solution = diffeqsolve(
        term,
        solver,
        t0=t_span[0],
        t1=t_span[1],
        dt0=dt,
        y0=initial_state,
        args=(params, inputs.temp, inputs.lday, inputs.prcp),
        saveat=SaveAt(ts=jnp.arange(t_span[0], t_span[1] + dt, dt)),
        stepsize_controller=controller,
        max_steps=10000  # 增加最大步数
    )
    
    return solution.ts, solution.ys

def loss_function(params: ModelParams, initial_state: ModelState, 
                 inputs: ModelInput, observed_flow: jnp.ndarray) -> float:
    """损失函数"""
    # 求解模型
    t_span = (1.0, float(len(inputs.temp)))
    dt = 1.0
    _, states = solve_model(params, initial_state, inputs, t_span, dt)
    
    # 计算预测流量
    def compute_flow(t_idx):
        snowpack_state = states.snowpack[t_idx]
        soilwater_state = states.soilwater[t_idx]
        t = t_idx + 1.0  # 转换为实际时间
        
        # 获取当前时间步的插值输入
        temp = interp1d(t, jnp.arange(1, len(inputs.temp) + 1), inputs.temp, method="cubic")
        lday = interp1d(t, jnp.arange(1, len(inputs.lday) + 1), inputs.lday, method="cubic")
        
        # 计算蒸发
        evap = step_func(soilwater_state) * calculate_pet(temp, lday) * \
               jnp.minimum(1.0, soilwater_state / params.Smax)
        
        # 计算基流
        baseflow = step_func(soilwater_state) * params.Qmax * \
                  jnp.exp(-params.f * (jnp.maximum(0.0, params.Smax - soilwater_state)))
        
        # 计算地表径流
        surfaceflow = jnp.maximum(0.0, soilwater_state - params.Smax)
        
        # 计算总流量
        return baseflow + surfaceflow
    
    # 使用vmap进行向量化计算
    predicted_flow = jax.vmap(compute_flow)(jnp.arange(len(states.snowpack)))
    return jnp.mean((predicted_flow - observed_flow) ** 2)

# 测试代码
def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据
    data_path = get_data_path()
    inputs_dict, observed_flow = load_hydro_data(data_path, data_length=10000)
    
    # 模型参数
    params = ModelParams(
        f=0.01674478, Smax=1709.461015, Qmax=18.46996175,
        Df=2.674548848, Tmax=0.175739196, Tmin=-2.092959084
    )
    
    # 初始状态
    initial_state = ModelState(snowpack=0.0, soilwater=50.0)
    
    # 输入数据
    inputs = ModelInput(
        temp=jnp.array(inputs_dict['temp']),
        lday=jnp.array(inputs_dict['lday']),
        prcp=jnp.array(inputs_dict['prcp'])
    )
    
    # 转换观测流量为JAX数组
    observed_flow = jnp.array(observed_flow)
    start_time = time.time()
    # 计算并打印损失值
    loss = loss_function(params, initial_state, inputs, observed_flow)
    print(f"\n损失值: {loss:.4f}")
    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.4f} 秒")
if __name__ == "__main__":
    main()
