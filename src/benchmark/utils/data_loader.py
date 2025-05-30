import pandas as pd
import numpy as np
from typing import Tuple, Dict
from pathlib import Path

def load_hydro_data(file_path: str, data_length: int=-1) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    加载水文数据
    
    参数:
        file_path: CSV文件路径
        
    返回:
        Tuple[Dict[str, np.ndarray], np.ndarray]: 输入数据和观测流量
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取输入数据
    inputs = {
        'temp': df['tmean(C)'].values[:data_length],  # 温度
        'lday': df['dayl(day)'].values[:data_length],  # 日照时长
        'prcp': df['prcp(mm/day)'].values[:data_length],  # 降水量
    }
    
    # 提取观测流量
    observed_flow = df['flow(mm)'].values[:data_length]
    
    return inputs, observed_flow

def get_data_path() -> str:
    """获取数据文件路径"""
    return str(Path(__file__).parent.parent.parent.parent / 'data' / 'exphydro' / '01013500.csv') 