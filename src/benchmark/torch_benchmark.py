import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torchdiffeq
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
from benchmark.utils.data_loader import load_hydro_data, get_data_path
from benchmark.utils.interpolate import natural_cubic_spline_coeffs, NaturalCubicSpline

@dataclass
class ModelParams:
    """模型参数"""
    Tmin: float  # 最小温度阈值
    Tmax: float  # 最大温度阈值
    Df: float    # 融雪系数
    Smax: float  # 最大土壤含水量
    Qmax: float  # 最大基流
    f: float     # 基流衰减系数

    def to_tensor(self) -> torch.Tensor:
        """将参数转换为张量"""
        return torch.tensor([self.Tmin, self.Tmax, self.Df, 
                           self.Smax, self.Qmax, self.f], 
                          dtype=torch.float32, requires_grad=True)

@dataclass
class ModelState:
    """模型状态"""
    snowpack: float    # 积雪量
    soilwater: float   # 土壤含水量

    def to_tensor(self) -> torch.Tensor:
        """将状态转换为张量"""
        return torch.tensor([self.snowpack, self.soilwater], 
                          dtype=torch.float32)

@dataclass
class ModelInput:
    """模型输入"""
    temp: torch.Tensor   # 温度
    lday: torch.Tensor   # 日照时长
    prcp: torch.Tensor   # 降水量

    @classmethod
    def from_numpy(cls, temp: np.ndarray, lday: np.ndarray, prcp: np.ndarray) -> 'ModelInput':
        """从NumPy数组创建输入"""
        return cls(
            temp=torch.tensor(temp, dtype=torch.float32),
            lday=torch.tensor(lday, dtype=torch.float32),
            prcp=torch.tensor(prcp, dtype=torch.float32)
        )

@dataclass
class ModelOutput:
    """模型输出"""
    flow: torch.Tensor        # 总流量
    baseflow: torch.Tensor    # 基流
    surfaceflow: torch.Tensor # 地表径流
    evap: torch.Tensor        # 蒸发量
    melt: torch.Tensor        # 融雪量

def step_func(x: torch.Tensor) -> torch.Tensor:
    """阶跃函数"""
    return (torch.tanh(5.0 * x) + 1.0) * 0.5

def calculate_pet(temp: torch.Tensor, lday: torch.Tensor) -> torch.Tensor:
    """计算潜在蒸散发"""
    return 29.8 * lday * 24 * 0.611 * torch.exp((17.3 * temp) / (temp + 237.3)) / (temp + 273.2)

class HydroModel(torch.nn.Module):
    """水文模型类"""
    def __init__(self, params: ModelParams, 
                 temp_interp: NaturalCubicSpline,
                 lday_interp: NaturalCubicSpline,
                 prcp_interp: NaturalCubicSpline):
        super().__init__()
        self.params = params.to_tensor()
        self.temp_interp = temp_interp
        self.lday_interp = lday_interp
        self.prcp_interp = prcp_interp

    def bucket_surface(self, state: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """表面bucket计算"""
        # 获取当前时间步的插值输入
        temp = self.temp_interp.evaluate(t).squeeze(-1)
        lday = self.lday_interp.evaluate(t).squeeze(-1)
        prcp = self.prcp_interp.evaluate(t).squeeze(-1)
        
        # 计算降雪和降雨
        snowfall = step_func(self.params[0] - temp) * prcp
        rainfall = step_func(temp - self.params[0]) * prcp
        
        # 计算融雪
        melt = step_func(temp - self.params[1]) * step_func(state[0]) * \
               torch.minimum(state[0], self.params[2] * (temp - self.params[1]))
        
        # 计算潜在蒸散发
        pet = calculate_pet(temp, lday)
        
        return snowfall, rainfall, melt, pet

    def bucket_soil(self, state: torch.Tensor, 
                   rainfall: torch.Tensor, melt: torch.Tensor, 
                   pet: torch.Tensor) -> ModelOutput:
        """土壤bucket计算"""
        # 计算蒸发
        evap = step_func(state[1]) * pet * torch.minimum(
            torch.tensor(1.0), state[1] / self.params[3])
        
        # 计算基流
        baseflow = step_func(state[1]) * self.params[4] * \
                  torch.exp(-self.params[5] * (torch.maximum(
                      torch.tensor(0.0), self.params[3] - state[1])))
        
        # 计算地表径流
        surfaceflow = torch.maximum(torch.tensor(0.0), state[1] - self.params[3])
        
        # 计算总流量
        flow = baseflow + surfaceflow
        
        return ModelOutput(flow=flow, baseflow=baseflow, 
                         surfaceflow=surfaceflow, evap=evap, melt=melt)

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """模型前向传播"""
        # 计算表面bucket
        snowfall, rainfall, melt, pet = self.bucket_surface(state, t)
        
        # 计算土壤bucket
        output = self.bucket_soil(state, rainfall, melt, pet)

        snowpack, soilwater = state[0], state[1]
        
        # 计算状态导数
        dsnowpack = torch.maximum(snowfall - melt, -snowpack)
        dsoilwater = torch.maximum((rainfall + melt) - (output.evap + output.flow), -soilwater)
        
        return torch.stack([dsnowpack, dsoilwater])

def solve_model(model: HydroModel, initial_state: ModelState, 
                t_span: Tuple[float, float], dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """求解模型"""
    # 设置求解器参数
    t_eval = torch.arange(t_span[0], t_span[1] + dt, dt)
    
    # 求解ODE
    solution = torchdiffeq.odeint(
        model,
        initial_state.to_tensor(),
        t_eval,
        method='rk4',
        rtol=1e-3,
        atol=1e-3
    )
    
    return t_eval, solution

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 加载数据
    data_path = get_data_path()
    time_length = 1000
    inputs_dict, observed_flow = load_hydro_data(data_path, data_length=time_length)
    
    # 模型参数
    params = ModelParams(
        f=0.01674478, Smax=1709.461015, Qmax=18.46996175,
        Df=2.674548848, Tmax=0.175739196, Tmin=-2.092959084
    )
    
    # 创建时间点
    times = torch.arange(1, time_length + 1, dtype=torch.float32)
    
    # 创建插值器
    temp_interp = NaturalCubicSpline(natural_cubic_spline_coeffs(times, torch.from_numpy(inputs_dict['temp']).unsqueeze(1)))
    lday_interp = NaturalCubicSpline(natural_cubic_spline_coeffs(times, torch.from_numpy(inputs_dict['lday']).unsqueeze(1)))
    prcp_interp = NaturalCubicSpline(natural_cubic_spline_coeffs(times, torch.from_numpy(inputs_dict['prcp']).unsqueeze(1)))
    
    # 创建模型
    model = HydroModel(params, temp_interp, lday_interp, prcp_interp)
    
    # 初始状态
    initial_state = ModelState(snowpack=0.0, soilwater=50.0)
    
    # 时间设置
    t_span = (1.0, time_length)  # 使用实际的时间索引
    dt = 1.0  # 时间步长为1天
    
    # 求解模型
    start_time = time.time()
    t_eval, states = solve_model(model, initial_state, t_span, dt)
    end_time = time.time()
    
    print(f"模型求解时间: {end_time - start_time:.4f} 秒")
    
    # 计算预测流量
    predicted_flow = torch.stack([
        model.bucket_soil(
            states[i],
            prcp_interp.evaluate(t_eval[i]).squeeze(-1),
            torch.tensor(0.0),
            calculate_pet(
                temp_interp.evaluate(t_eval[i]).squeeze(-1),
                lday_interp.evaluate(t_eval[i]).squeeze(-1)
            )
        ).flow
        for i in range(len(t_eval))
    ])
    
    # 计算损失
    loss = torch.mean((predicted_flow - torch.tensor(observed_flow, dtype=torch.float32)) ** 2)
    print(f"\n损失值: {loss.item():.4f}")
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(t_eval, predicted_flow.detach().numpy(), label='Predicted Flow')
    plt.plot(t_eval, observed_flow, label='Observed Flow')
    plt.xlabel('Time')
    plt.ylabel('Flow (mm/day)')
    plt.title('Flow Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
