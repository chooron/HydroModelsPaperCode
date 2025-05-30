import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Tuple, List
import time
from benchmark.utils.data_loader import load_hydro_data, get_data_path

@dataclass
class ModelParams:
    """模型参数"""
    Tmin: float  # 最小温度阈值
    Tmax: float  # 最大温度阈值
    Df: float    # 融雪系数
    Smax: float  # 最大土壤含水量
    Qmax: float  # 最大基流
    f: float     # 基流衰减系数

@dataclass
class ModelState:
    """模型状态"""
    snowpack: float    # 积雪量
    soilwater: float   # 土壤含水量

@dataclass
class ModelInput:
    """模型输入"""
    temp: np.ndarray   # 温度
    lday: np.ndarray   # 日照时长
    prcp: np.ndarray   # 降水量

@dataclass
class ModelOutput:
    """模型输出"""
    flow: float        # 总流量
    baseflow: float    # 基流
    surfaceflow: float # 地表径流
    evap: float        # 蒸发量
    melt: float        # 融雪量

def step_func(x: float) -> float:
    """阶跃函数"""
    return (np.tanh(5.0 * x) + 1.0) * 0.5

def calculate_pet(temp: float, lday: float) -> float:
    """计算潜在蒸散发"""
    return 29.8 * lday * 24 * 0.611 * np.exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)

def bucket_surface(state: ModelState, temp: float, lday: float, prcp: float, params: ModelParams) -> Tuple[float, float, float, float]:
    """表面bucket计算"""
    # 计算降雪和降雨
    snowfall = step_func(params.Tmin - temp) * prcp
    rainfall = step_func(temp - params.Tmin) * prcp
    
    # 计算融雪
    melt = step_func(temp - params.Tmax) * step_func(state.snowpack) * \
           min(state.snowpack, params.Df * (temp - params.Tmax))
    
    # 计算潜在蒸散发
    pet = calculate_pet(temp, lday)
    
    return snowfall, rainfall, melt, pet

def bucket_soil(state: ModelState, params: ModelParams, 
                rainfall: float, melt: float, pet: float) -> ModelOutput:
    """土壤bucket计算"""
    # 计算蒸发
    evap = step_func(state.soilwater) * pet * min(1.0, state.soilwater / params.Smax)
    
    # 计算基流
    baseflow = step_func(state.soilwater) * params.Qmax * \
               np.exp(-params.f * (max(0.0, params.Smax - state.soilwater)))
    
    # 计算地表径流
    surfaceflow = max(0.0, state.soilwater - params.Smax)
    
    # 计算总流量
    flow = baseflow + surfaceflow
    
    return ModelOutput(flow=flow, baseflow=baseflow, 
                      surfaceflow=surfaceflow, evap=evap, melt=melt)

def model_derivatives(t: float, state_array: np.ndarray, 
                     interpolators: Tuple[interp1d, interp1d, interp1d],
                     params: ModelParams) -> np.ndarray:
    """模型导数计算"""
    # 获取当前时间步的插值输入
    temp = interpolators[0](t)
    lday = interpolators[1](t)
    prcp = interpolators[2](t)
    
    state = ModelState(snowpack=state_array[0], soilwater=state_array[1])
    
    # 计算表面bucket
    snowfall, rainfall, melt, pet = bucket_surface(state, temp, lday, prcp, params)
    
    # 计算土壤bucket
    output = bucket_soil(state, params, rainfall, melt, pet)
    
    # 计算状态导数
    dsnowpack = snowfall - melt
    dsoilwater = (rainfall + melt) - (output.evap + output.flow)
    
    return np.array([dsnowpack, dsoilwater])

def solve_model(initial_state: ModelState, inputs: ModelInput, params: ModelParams, 
                t_span: Tuple[float, float], dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """求解模型"""
    # 创建时间点
    t_points = np.arange(t_span[0], t_span[1] + dt, dt)
    
    # 创建插值器
    temp_interp = interp1d(t_points, inputs.temp, kind='linear', bounds_error=False, fill_value=(inputs.temp[0], inputs.temp[-1]))
    lday_interp = interp1d(t_points, inputs.lday, kind='linear', bounds_error=False, fill_value=(inputs.lday[0], inputs.lday[-1]))
    prcp_interp = interp1d(t_points, inputs.prcp, kind='linear', bounds_error=False, fill_value=(inputs.prcp[0], inputs.prcp[-1]))
    
    # 求解ODE
    solution = solve_ivp(
        model_derivatives,
        t_span,
        np.array([initial_state.snowpack, initial_state.soilwater]),
        args=((temp_interp, lday_interp, prcp_interp), params),
        t_eval=t_points,
        method='RK45',
        rtol=1e-3,
        atol=1e-3
    )
    
    return solution.t, solution.y

def main():
    # 设置随机种子
    np.random.seed(42)
    
    # 加载数据
    data_path = get_data_path()
    inputs_dict, observed_flow = load_hydro_data(data_path, data_length=1000)  # 使用一年的数据
    
    # 模型参数
    params = ModelParams(
        f=0.01674478, Smax=1709.461015, Qmax=18.46996175,
        Df=2.674548848, Tmax=0.175739196, Tmin=-2.092959084
    )
    
    # 初始状态
    initial_state = ModelState(snowpack=0.0, soilwater=50.0)
    
    # 输入数据
    inputs = ModelInput(
        temp=inputs_dict['temp'],
        lday=inputs_dict['lday'],
        prcp=inputs_dict['prcp']
    )
    
    # 时间设置
    t_span = (0.0, len(inputs_dict['temp']) - 1)
    dt = 1.0
    
    # 求解模型
    start_time = time.time()
    t_eval, states = solve_model(initial_state, inputs, params, t_span, dt)
    end_time = time.time()
    
    print(f"模型求解时间: {end_time - start_time:.4f} 秒")
    
    # 计算预测流量
    predicted_flow = np.array([
        bucket_soil(
            ModelState(snowpack=states[0, i], soilwater=states[1, i]),
            params,
            inputs.prcp[i], 0.0,
            calculate_pet(inputs.temp[i], inputs.lday[i])
        ).flow
        for i in range(len(t_eval))
    ])
    
    # 计算损失
    loss = np.mean((predicted_flow - observed_flow) ** 2)
    print(f"\n损失值: {loss:.4f}")

if __name__ == "__main__":
    main()
