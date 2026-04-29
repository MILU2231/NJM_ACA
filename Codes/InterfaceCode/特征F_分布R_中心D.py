# -*- coding: utf-8 -*-
"""
10min级特征工程（从事件切片JSON文件） - 多进程并行版
读取列车事件和非列车事件的JSON文件，按10min窗口聚合并计算聚类指标
使用多进程加速处理，基准选择：第一个列车事件≥2的10min窗口
支持按日期文件夹过滤数据
传感器信息从数据文件中读取并统一应用

修改说明：
1. 将所有结果保存到统一目录：X:/NJM_Item\ACA_对接优化数据集\ACA特征工程结果\
2. 按特征类型分类存储
3. 保存原始29维特征
4. 按时间范围组织文件
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import warnings
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import traceback
from scipy.fftpack import fft
from scipy.stats import kurtosis

warnings.filterwarnings('ignore')

# 在文件开头添加一个自定义JSON编码器
import numpy as np
import json
from json import JSONEncoder

class NumpyEncoder(JSONEncoder):
    """自定义JSON编码器，处理numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return super(NumpyEncoder, self).default(obj)

# =========================
# 特征提取函数（来自特征工程）
# =========================
def time_statistical_compute(signal):
    """时域特征计算"""
    N = len(signal)
    p1 = np.mean(signal)
    x = signal - p1
    p2 = np.sqrt(np.sum(x ** 2) / N)  # RMS
    p3 = (np.sum(np.sqrt(np.abs(x))) / N) ** 2  # 方根幅值
    p4 = np.sum(np.abs(x)) / N  # 绝对平均值

    # P5: 原始第三动量
    p5 = np.sum(x ** 3) / N

    # P6: 原始第四动量
    p6 = np.sum(x ** 4) / N

    p7 = np.var(x, ddof=1)  # 方差
    p8 = np.max(x)
    p9 = np.min(x)
    p10 = p8 - p9

    # 增加微小量 epsilon 防止除以零
    EPS = 1e-9

    f1 = p2 / (p4 + EPS)  # 波形指标
    f2 = p8 / (p2 + EPS)  # 峰值指标
    f3 = p8 / (p4 + EPS)  # 脉冲指标
    f4 = p8 / (p3 + EPS)  # 裕度指标

    # P15 偏斜度指标 (Skewness Coefficient)
    f5 = p5 / (np.sqrt(p7) ** 3 + EPS)

    # P16 峭度指标 (Kurtosis Coefficient)
    f6 = p6 / (np.sqrt(p7) ** 4 + EPS)

    val = np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])
    factor = np.array([f1, f2, f3, f4, f5, f6])
    return np.concatenate([val, factor])


def fre_statistical_compute(frequencies, amplitudes):
    """频域特征计算"""
    fre_line_num = max(amplitudes.shape)
    p1 = np.mean(amplitudes)
    p2 = np.sum((amplitudes - p1) ** 2) / fre_line_num
    p3 = np.sum((amplitudes - p1) ** 3) / (fre_line_num * np.sqrt(p2 ** 3)) if p2 > 0 else 0
    p4 = np.sum((amplitudes - p1) ** 4) / (fre_line_num * p2 ** 2) if p2 > 0 else 0

    meanf = np.sum(frequencies * amplitudes) / np.sum(amplitudes) if np.sum(amplitudes) > 0 else 0
    sigma = np.sqrt(np.sum((frequencies - meanf) ** 2 * amplitudes) / fre_line_num)

    p5 = meanf
    p6 = sigma
    p7 = np.sqrt(np.sum(frequencies ** 2 * amplitudes) / np.sum(amplitudes)) if np.sum(amplitudes) > 0 else 0
    p8 = np.sqrt(np.sum(frequencies ** 4 * amplitudes) / np.sum(frequencies ** 2 * amplitudes)) if np.sum(
        frequencies ** 2 * amplitudes) > 0 else 0
    p9 = np.sum(frequencies ** 2 * amplitudes) / np.sqrt(
        np.sum(amplitudes) * np.sum(frequencies ** 4 * amplitudes)) if np.sum(amplitudes) * np.sum(
        frequencies ** 4 * amplitudes) > 0 else 0
    p10 = sigma / meanf if meanf > 0 else 0
    p11 = np.sum((frequencies - meanf) ** 3 * amplitudes) / (sigma ** 3 * fre_line_num) if sigma > 0 else 0
    p12 = np.sum((frequencies - meanf) ** 4 * amplitudes) / (sigma ** 4 * fre_line_num) if sigma > 0 else 0
    p13 = np.sum(np.sqrt(np.abs(frequencies - meanf)) * amplitudes) / (
            np.sqrt(sigma) * fre_line_num) if sigma > 0 else 0

    return np.array([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13])


def extract_29_features_teacher(signal, fs=50):
    """提取29维特征"""
    L = len(signal)
    nfft = 2 ** int(np.ceil(np.log2(L)))
    y_f = fs * np.arange(0, nfft / 2) / nfft
    y_ft = fft(signal, n=nfft)
    amplitudes = 2 * np.abs(y_ft[:nfft // 2]) / L

    time_features = time_statistical_compute(signal)
    fre_features = fre_statistical_compute(y_f, amplitudes)

    return np.concatenate([time_features, fre_features])


def extract_features_from_json_data(json_data):
    """从JSON数据中提取29维特征（不含标签）"""
    try:
        # 提取信号数据
        signal = np.array(json_data['data'])

        # 提取29维特征
        features_29 = extract_29_features_teacher(signal)

        return features_29.tolist()
    except Exception as e:
        print(f"提取特征失败: {e}")
        return None


# =========================
# 增强的字体设置函数
# =========================
def 设置中文字体():
    """设置中文字体，确保中文、字符和角标正常显示"""
    try:
        # 获取系统可用字体
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        # 优先选择的中文字体列表
        中文字体优先级 = [
            'SimHei',  # Windows黑体
            'Microsoft YaHei',  # Windows微软雅黑
            'SimSun',  # Windows宋体
            'NSimSun',  # Windows新宋体
            'FangSong',  # Windows仿宋
            'KaiTi',  # Windows楷体
            'STSong',  # Mac/Linux宋体
            'STHeiti',  # Mac/Linux黑体
            'STKaiti',  # Mac/Linux楷体
            'LiSu',  # 隶书
            'YouYuan',  # 幼圆
            'Arial Unicode MS',  # 跨平台Unicode字体
            'DejaVu Sans'  # Linux通用字体
        ]

        # 查找可用的中文字体
        可用中文字体 = []
        for font_name in 中文字体优先级:
            if font_name in available_fonts:
                可用中文字体.append(font_name)
                print(f"✅ 找到可用字体: {font_name}")

        if 可用中文字体:
            # 设置字体
            rcParams['font.sans-serif'] = 可用中文字体 + ['DejaVu Sans', 'Arial']
            rcParams['axes.unicode_minus'] = False  # 正确显示负号

            # 设置数学字体
            if 'STIXGeneral' in available_fonts:
                rcParams['mathtext.fontset'] = 'stix'
                rcParams['mathtext.default'] = 'regular'
                print("✅ 使用STIX数学字体")
            else:
                rcParams['mathtext.fontset'] = 'cm'
                rcParams['mathtext.default'] = 'regular'
                print("✅ 使用Computer Modern数学字体")

            print(f"✅ 字体设置成功，使用字体: {可用中文字体[0]}")
        else:
            # 如果没有中文字体，使用默认字体并设置unicode_minus
            rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
            rcParams['axes.unicode_minus'] = False
            print("⚠️ 未找到中文字体，使用默认英文字体")

        # 设置其他绘图参数
        rcParams['figure.dpi'] = 100
        rcParams['savefig.dpi'] = 300
        rcParams['savefig.bbox'] = 'tight'
        rcParams['savefig.pad_inches'] = 0.1

    except Exception as e:
        print(f"⚠️ 字体设置失败: {e}")
        # 设置安全的默认值
        rcParams['axes.unicode_minus'] = False
        rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']


# 初始化字体设置
设置中文字体()

# =========================
# 路径配置 - 修改为统一存储目录
# =========================

# 事件切片根目录（保持不变）
事件切片根目录 = r"X:/NJM_Item\ACA_对接优化数据集\C13_ACA列车切片数据\C13_02列车事件切片"

# 统一特征工程结果目录
特征工程结果根目录 = r"X:/NJM_Item\ACA_对接优化数据集\ACA特征工程结果"

# 创建主目录结构
os.makedirs(特征工程结果根目录, exist_ok=True)

# 在根目录下创建子目录
# 子目录结构 = {
#     '10min级特征': '10分钟级聚类特征',
#     '1_raw_features': '10分钟级原始特征',
#     '2_time_series': '10分钟级时间序列',
#     '3_reports': '10分钟级分析报告',
#     '4_benchmark': '基准信息',
# }
#
# # 创建所有子目录
# for 子目录, 描述 in 子目录结构.items():
#     路径 = os.path.join(特征工程结果根目录, 子目录)
#     os.makedirs(路径, exist_ok=True)
#     print(f"✅ 创建目录: {子目录} - {描述}")

# # 创建带时间戳的详细结果目录（可选，用于每次运行）
# 时间戳目录 = os.path.join(
#     特征工程结果根目录,
#     f"2_10min级特征_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
# )
# os.makedirs(时间戳目录, exist_ok=True)

# print(f"📁 主要结果将保存在: {特征工程结果根目录}")
# print(f"📁 本次运行详细结果将保存在: {时间戳目录}")

# 29维特征名称
特征名称 = [
    "P1 均值",
    "P2 均方根值(RMS)",
    "P3 方根幅值",
    "P4 绝对平均值",
    "P5 偏斜度(原始三阶矩)",
    "P6 峭度(原始四阶矩)",
    "P7 方差",
    "P8 最大值",
    "P9 最小值",
    "P10 峰峰值",
    "P11 波形指标",
    "P12 峰值指标",
    "P13 脉冲指标",
    "P14 裕度指标",
    "P15 偏斜度指标(标准化)",
    "P16 峭度指标(标准化)",
    "P17 频谱均值",
    "P18 频谱方差",
    "P19 频谱偏斜度",
    "P20 频谱峭度",
    "P21 频率中心",
    "P22 频率标准差",
    "P23 均方根频率",
    "P24 主频带位置比",
    "P25 主频带位置反比",
    "P26 频率变异系数",
    "P27 频率偏斜度(频域)",
    "P28 频率峭度(频域)",
    "P29 频率绝对偏差加权"
]

# 全局变量存储从文件中读取的传感器信息
全局传感器ID = None
全局传感器名称 = None


# =========================
# 新增函数：保存原始窗口特征
# =========================
def 保存原始窗口特征(窗口事件字典, 窗口列表, 时间戳目录, 开始日期文件夹, 结束日期文件夹):
    """保存每个窗口内的原始29维特征"""
    print("💾 保存原始窗口特征...")

    原始特征目录 = os.path.join(特征工程结果根目录, "2_1_raw_features")
    时间范围 = f"{开始日期文件夹}_to_{结束日期文件夹}"
    时间范围目录 = os.path.join(原始特征目录, 时间范围)
    os.makedirs(时间范围目录, exist_ok=True)

    总样本数 = 0
    保存的窗口数 = 0

    for 窗口开始时间 in 窗口列表:
        窗口事件 = 窗口事件字典[窗口开始时间]

        if len(窗口事件) > 0:
            # 收集当前窗口的原始特征
            窗口特征数据 = {
                'window_start_time': 窗口开始时间.strftime('%Y-%m-%d %H:%M:%S'),
                'num_events': len(窗口事件),
                'train_events': int(sum(1 for event in 窗口事件 if event.get('事件类型') == '列车事件')),  # 转为int
                'non_train_events': int(
                    len(窗口事件) - sum(1 for event in 窗口事件 if event.get('事件类型') == '列车事件')),  # 转为int
                'samples': []
            }

            for event in 窗口事件:
                if 'features' in event and len(event['features']) >= 29:
                    # 转换numpy数组为Python列表
                    features_29 = event['features'][:29]
                    if isinstance(features_29, np.ndarray):
                        features_29 = features_29.tolist()

                    样本数据 = {
                        'event_type': event.get('事件类型', 'unknown'),
                        'start_time': event['start_time'].strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_seconds': float(event['duration_seconds']) if isinstance(event['duration_seconds'],
                                                                                           np.floating) else event[
                            'duration_seconds'],
                        'label': int(event.get('label', 0)),  # 转为int
                        'features_29': features_29
                    }
                    窗口特征数据['samples'].append(样本数据)
                    总样本数 += 1

            # 保存单个窗口的特征
            if 窗口特征数据['samples']:
                文件名 = f"{窗口开始时间.strftime('%Y%m%d_%H%M')}_raw_features.json"
                文件路径 = os.path.join(时间范围目录, 文件名)
                with open(文件路径, 'w', encoding='utf-8') as f:
                    json.dump(窗口特征数据, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                保存的窗口数 += 1

    print(f"✅ 原始窗口特征已保存到: {时间范围目录}")
    print(f"  总窗口数: {保存的窗口数}")
    print(f"  总样本数: {总样本数}")

    # 创建汇总文件
    汇总数据 = {
        '时间范围': 时间范围,
        '保存时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '总窗口数': int(len(窗口列表)),  # 转为int
        '有事件窗口数': int(sum(1 for 窗口 in 窗口列表 if len(窗口事件字典[窗口]) > 0)),  # 转为int
        '保存的窗口数': int(保存的窗口数),  # 转为int
        '总样本数': int(总样本数),  # 转为int
        '目录位置': 时间范围目录
    }

    汇总路径 = os.path.join(时间范围目录, "summary.json")
    with open(汇总路径, 'w', encoding='utf-8') as f:
        json.dump(汇总数据, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # 同时保存到时间戳目录
    时间戳原始目录 = os.path.join(时间戳目录, "raw_window_features")
    os.makedirs(时间戳原始目录, exist_ok=True)

    # 保存汇总版本到时间戳目录
    时间戳汇总路径 = os.path.join(时间戳原始目录, "raw_features_summary.json")
    with open(时间戳汇总路径, 'w', encoding='utf-8') as f:
        json.dump(汇总数据, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return 原始特征目录

# =========================
# 新增函数：保存时间序列数据
# =========================
def 保存时间序列数据(df, 时间戳目录, 开始日期文件夹, 结束日期文件夹):
    """保存时间序列数据到统一目录"""
    print("💾 保存时间序列数据...")

    # 1. 保存到统一时间序列目录
    时间序列目录 = os.path.join(特征工程结果根目录, "2_2_time_series")
    时间范围 = f"{开始日期文件夹}_to_{结束日期文件夹}"
    时间范围目录 = os.path.join(时间序列目录, 时间范围)
    os.makedirs(时间范围目录, exist_ok=True)

    # 保存CSV
    csv_path = os.path.join(时间范围目录, "10min_time_series.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 保存JSON简化版 - 先转换为Python原生类型
    json_path = os.path.join(时间范围目录, "10min_time_series.json")
    json_data = df.to_dict(orient='records')

    # 转换所有numpy类型为Python类型
    可序列化数据 = []
    for record in json_data:
        可序列化记录 = {}
        for key, value in record.items():
            if pd.isna(value):
                可序列化记录[key] = None
            elif isinstance(value, (np.integer, np.int64, np.int32)):
                可序列化记录[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                可序列化记录[key] = float(value)
            elif isinstance(value, np.ndarray):
                可序列化记录[key] = value.tolist()
            elif isinstance(value, np.bool_):
                可序列化记录[key] = bool(value)
            elif isinstance(value, pd.Timestamp):
                可序列化记录[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            else:
                可序列化记录[key] = value
        可序列化数据.append(可序列化记录)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(可序列化数据, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # 2. 同时保存到时间戳目录（详细版）
    时间戳csv路径 = os.path.join(时间戳目录, "10min_time_series_detailed.csv")
    df.to_csv(时间戳csv路径, index=False, encoding='utf-8-sig')

    print(f"✅ 时间序列数据已保存到:")
    print(f"  统一目录: {csv_path}")
    print(f"  时间戳目录: {时间戳csv路径}")

    # 生成数据统计信息
    数据统计 = {
        '时间范围': 时间范围,
        '保存时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '总窗口数': int(len(df)),
        '有事件窗口数': int((df['事件数量'] > 0).sum()),
        '时间范围_start': df['窗口开始时间'].min().strftime('%Y-%m-%d %H:%M:%S'),
        '时间范围_end': df['窗口开始时间'].max().strftime('%Y-%m-%d %H:%M:%S'),
        '总事件数': int(df['事件数量'].sum()),
        '总列车事件数': int(df['列车事件数量'].sum()),
        '总非列车事件数': int(df['非列车事件数量'].sum()),
        '文件大小_CSV': f"{os.path.getsize(csv_path) / 1024:.2f} KB"
    }

    统计路径 = os.path.join(时间范围目录, "time_series_stats.json")
    with open(统计路径, 'w', encoding='utf-8') as f:
        json.dump(数据统计, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    return csv_path

# =========================
# 新增函数：保存基准信息
# =========================
def 保存基准信息(基准信息, 时间戳目录, 开始日期文件夹, 结束日期文件夹):
    """保存基准信息到统一目录"""
    print("💾 保存基准信息...")

    # 1. 保存到统一基准目录
    基准目录 = os.path.join(特征工程结果根目录, "2_4_benchmark")
    os.makedirs(基准目录, exist_ok=True)

    # 文件名包含时间范围
    文件名 = f"benchmark_{开始日期文件夹}_to_{结束日期文件夹}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    基准路径 = os.path.join(基准目录, 文件名)

    with open(基准路径, 'w', encoding='utf-8') as f:
        json.dump(基准信息, f, ensure_ascii=False, indent=2)

    # 2. 同时保存到时间戳目录
    时间戳基准路径 = os.path.join(时间戳目录, "benchmark_info.json")
    with open(时间戳基准路径, 'w', encoding='utf-8') as f:
        json.dump(基准信息, f, ensure_ascii=False, indent=2)

    print(f"✅ 基准信息已保存到:")
    print(f"  统一目录: {基准路径}")
    print(f"  时间戳目录: {时间戳基准路径}")

    return 基准路径


# =========================
# 新增函数：生成汇总报告
# =========================
def 生成汇总报告(df, 基准信息, 时间戳目录, 开始日期文件夹, 结束日期文件夹):
    """生成处理汇总报告"""
    print("📋 生成汇总报告...")

    报告目录 = os.path.join(特征工程结果根目录, "2_3_reports")
    时间范围 = f"{开始日期文件夹}_to_{结束日期文件夹}"
    时间范围目录 = os.path.join(报告目录, 时间范围)
    os.makedirs(时间范围目录, exist_ok=True)

    # 创建报告时确保使用Python原生类型
    报告内容 = [
        "=" * 80,
        "10分钟级特征工程处理汇总报告",
        "=" * 80,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"处理日期范围: {开始日期文件夹} 到 {结束日期文件夹}",
        f"数据存储位置: {特征工程结果根目录}",
        "",
        "1. 目录结构:",
        f"   2_1_raw_features/{时间范围}/ - 原始窗口特征",
        f"   2_2_time_series/{时间范围}/ - 时间序列数据",
        f"   2_3_reports/{时间范围}/ - 分析报告",
        f"   2_4_benchmark/ - 基准信息",
        "",
        "2. 数据统计:",
        f"   总窗口数: {int(len(df))}",
        f"   时间范围: {df['窗口开始时间'].min()} 到 {df['窗口开始时间'].max()}",
        f"   有事件窗口数: {int((df['事件数量'] > 0).sum())} ({int((df['事件数量'] > 0).sum()) / int(len(df)) * 100:.1f}%)",
        f"   总事件数: {int(df['事件数量'].sum())}",
        f"   总列车事件数: {int(df['列车事件数量'].sum())}",
        f"   总非列车事件数: {int(df['非列车事件数量'].sum())}",
        f"   标签均值（列车事件比例）: {float(df['标签均值'].mean()):.3f}",
        "",
        "3. 基准信息:",
        f"   基准窗口: {基准信息['基准窗口']}",
        f"   基准窗口列车事件数: {int(基准信息['基准窗口列车事件数'])}",
        f"   基准窗口总事件数: {int(基准信息['基准窗口总事件数'])}",
        f"   基准特征样本数: {int(基准信息['基准特征样本数'])}",
        f"   传感器信息: {基准信息['传感器信息']['sensor_id']}_{基准信息['传感器信息']['sensor_name']}",
        f"   选择理由: {基准信息['选择理由']}",
        "",
        "4. 关键特征统计:",
    ]

    # 添加关键特征统计
    关键特征 = ['中心偏移距离', '分布半径', '聚类密度', 'P1均值', 'P5偏斜度', 'P16峭度指标']
    for feat in 关键特征:
        if feat in df.columns:
            非空值 = df[feat].dropna()
            if len(非空值) > 0:
                报告内容.append(f"   {feat}:")
                报告内容.append(
                    f"      有效值数量: {int(len(非空值))}/{int(len(df))} ({int(len(非空值)) / int(len(df)) * 100:.1f}%)")
                报告内容.append(f"      均值: {float(非空值.mean()):.4f}")
                报告内容.append(f"      标准差: {float(非空值.std()):.4f}")
                报告内容.append(f"      最小值: {float(非空值.min()):.4f}")
                报告内容.append(f"      最大值: {float(非空值.max()):.4f}")

    报告内容.extend([
        "",
        "5. 文件清单:",
        f"   - 时间序列数据: 2_2_time_series/{时间范围}/10min_time_series.csv",
        f"   - 原始特征数据: 2_1_raw_features/{时间范围}/*.json",
        f"   - 基准信息: 2_4_benchmark/benchmark_*.json",
        f"   - 标准化器: 2_4_benchmark/scaler_*.pkl",
        f"   - 基准中心: 2_4_benchmark/benchmark_center_*.npy",
        f"   - 分析报告: 2_3_reports/{时间范围}/",
        f"   - 本次运行详细结果: {时间戳目录}",
        "",
        "6. 处理状态: ✅ 完成",
        "",
        "=" * 80,
        "处理完成!"
    ])

    报告路径 = os.path.join(时间范围目录, "处理汇总报告.txt")
    with open(报告路径, 'w', encoding='utf-8') as f:
        f.write('\n'.join(报告内容))

    print(f"✅ 汇总报告已保存到: {报告路径}")

    # 同时保存到时间戳目录
    时间戳报告路径 = os.path.join(时间戳目录, "处理汇总报告.txt")
    with open(时间戳报告路径, 'w', encoding='utf-8') as f:
        f.write('\n'.join(报告内容))

    return 报告路径

# =========================
# 原有工具函数（保持不变）
# =========================
def 获取10min窗口ID(时间戳):
    """将时间戳转换为10min窗口的开始时间"""
    if isinstance(时间戳, str):
        时间戳 = datetime.strptime(时间戳, "%Y-%m-%d %H:%M:%S")

    # 向下取整到最近的10分钟
    分钟 = 时间戳.minute
    十分钟块 = (分钟 // 10) * 10
    窗口开始时间 = datetime(时间戳.year, 时间戳.month, 时间戳.day,
                            时间戳.hour, 十分钟块, 0)
    return 窗口开始时间


def 读取事件JSON文件(文件路径):
    """读取单个事件JSON文件并提取29维特征（不含标签）"""
    try:
        with open(文件路径, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 从data中提取29维特征
        features_29 = extract_29_features_teacher(np.array(data['data']))
        if features_29 is None:
            return None

        # 根据事件类型判断标签
        event_type = data.get('type', 'unknown')
        if event_type in ['single', 'convergence']:
            label = 1  # 列车事件
        else:
            label = 0  # 非列车事件

        # 从数据文件中读取传感器信息
        sensor_id = data.get('sensor_id', 'NJM-ACA-C13-02')  # 默认值
        sensor_name = data.get('sensor_name', '下游侧NMC27索索力')  # 默认值

        # 更新全局传感器信息（只更新一次）
        global 全局传感器ID, 全局传感器名称
        if 全局传感器ID is None:
            全局传感器ID = sensor_id
        if 全局传感器名称 is None:
            全局传感器名称 = sensor_name

        # 提取事件信息（使用从文件中读取的传感器信息）
        事件信息 = {
            'start_time': datetime.strptime(data['start_time'], "%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.strptime(data['end_time'], "%Y-%m-%d %H:%M:%S"),
            'duration_seconds': data['duration_seconds'],
            'event_id': data.get('event_id', 0),
            'train_index': data.get('train_index', 0),
            'type': event_type,
            'is_borrowed': data.get('is_borrowed', False),
            'sampling_rate': data.get('sampling_rate', 50),
            'features': features_29,  # 29维特征（不含标签）
            'label': label,
            'sensor_id': sensor_id,  # 从JSON文件中读取的传感器ID
            'sensor_name': sensor_name,  # 从JSON文件中读取的传感器名称
            'data': data.get('data', [])[:50]  # 保存前50个数据点用于验证
        }

        return 事件信息
    except Exception as e:
        print(f"❌ 读取JSON文件失败 {文件路径}: {e}")
        return None


def 获取日期文件夹列表(根目录, 开始日期文件夹=None, 结束日期文件夹=None):
    """获取指定范围内的日期文件夹列表"""
    所有日期文件夹 = []

    # 遍历根目录下的所有子目录
    for item in os.listdir(根目录):
        item_path = os.path.join(根目录, item)
        if os.path.isdir(item_path):
            # 尝试解析文件夹名为日期
            try:
                folder_date = datetime.strptime(item, "%Y-%m-%d")
                所有日期文件夹.append((item, folder_date))
            except ValueError:
                # 如果不是日期格式的文件夹，跳过
                continue

    # 按日期排序
    所有日期文件夹.sort(key=lambda x: x[1])

    # 如果指定了开始和结束日期，进行过滤
    开始日期 = None
    结束日期 = None

    if 开始日期文件夹:
        开始日期 = datetime.strptime(开始日期文件夹, "%Y-%m-%d")
    if 结束日期文件夹:
        结束日期 = datetime.strptime(结束日期文件夹, "%Y-%m-%d")

    过滤后文件夹 = []
    for folder_name, folder_date in 所有日期文件夹:
        if 开始日期 and folder_date < 开始日期:
            continue
        if 结束日期 and folder_date > 结束日期:
            continue
        过滤后文件夹.append(folder_name)

    return 过滤后文件夹


def 扫描目录JSON文件_按日期过滤(目录路径, 开始日期文件夹=None, 结束日期文件夹=None):
    """扫描指定目录下的所有JSON文件（支持按日期文件夹过滤）"""
    json_files = []

    # 先获取日期文件夹列表
    日期文件夹列表 = 获取日期文件夹列表(目录路径, 开始日期文件夹, 结束日期文件夹)

    if not 日期文件夹列表:
        print(f"⚠️ 在 {目录路径} 中未找到日期格式的文件夹")
        return json_files

    print(f"📅 处理日期文件夹: {日期文件夹列表[:5]}... (共{len(日期文件夹列表)}个)")

    # 遍历每个日期文件夹
    for date_folder in 日期文件夹列表:
        date_folder_path = os.path.join(目录路径, date_folder)
        if os.path.exists(date_folder_path):
            # 扫描该日期文件夹下的所有JSON文件
            for root, dirs, files in os.walk(date_folder_path):
                for file in files:
                    if file.endswith('.json'):
                        json_files.append(os.path.join(root, file))

    return json_files


def 并行读取事件文件(json_files, 事件类型):
    """并行读取多个JSON文件"""
    print(f"📂 并行读取{事件类型}JSON文件... 使用{cpu_count()}个CPU核心")

    所有事件 = []
    读取失败数 = 0

    # 使用ProcessPoolExecutor进行并行读取
    with ProcessPoolExecutor(max_workers=min(cpu_count(), 8)) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(读取事件JSON文件, file): file for file in json_files}

        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_file)):
            if (i + 1) % 100 == 0:
                print(f"  进度: {i + 1}/{len(json_files)}")

            file_path = future_to_file[future]
            try:
                event = future.result()
                if event:
                    event['事件类型'] = 事件类型
                    event['文件路径'] = file_path
                    所有事件.append(event)
                else:
                    读取失败数 += 1
            except Exception as e:
                print(f"⚠️ 处理文件 {file_path} 时出错: {e}")
                读取失败数 += 1

    print(f"✅ 成功读取 {len(所有事件)} 个{事件类型}事件，失败: {读取失败数}")
    return 所有事件


def 计算增强特征_无标签(X_scaled, 当前中心, 基准中心):
    """计算增强特征，不包含标签相关计算"""
    if len(X_scaled) < 3:  # 样本太少，返回NaN
        return np.full(14, np.nan)

    try:
        # 1. 核心聚类指标
        中心偏移距离 = np.linalg.norm(当前中心 - 基准中心)

        # 计算所有样本到当前中心的距离
        样本距离 = np.linalg.norm(X_scaled - 当前中心, axis=1)
        分布半径 = np.percentile(样本距离, 95) if len(样本距离) > 0 else 0

        # 2. 样本统计特征
        样本数量 = len(X_scaled)
        聚类密度 = 样本数量 / (分布半径 + 1e-10) if 分布半径 > 0 else 0
        距离均值 = 样本距离.mean() if len(样本距离) > 0 else 0
        距离标准差 = 样本距离.std() if len(样本距离) > 1 else 0

        # 3. 原始特征的统计特征
        特征均值 = X_scaled.mean(axis=0)
        特征标准差 = X_scaled.std(axis=0)

        # 4. 特征变异系数
        特征变异系数 = 特征标准差 / (np.abs(特征均值) + 1e-10)

        # 5. 主成分分析
        try:
            pca = PCA(n_components=2)
            pca.fit_transform(X_scaled)
            主成分方差比 = pca.explained_variance_ratio_
        except:
            主成分方差比 = np.zeros(2)

        # 6. 特征相关性
        try:
            特征相关性 = np.corrcoef(X_scaled.T)
            平均相关性 = np.mean(特征相关性[np.triu_indices_from(特征相关性, k=1)])
        except:
            平均相关性 = 0

        # 7. 异常点检测
        if len(样本距离) >= 4:  # 需要至少4个样本计算IQR
            Q1 = np.percentile(样本距离, 25)
            Q3 = np.percentile(样本距离, 75)
            IQR = Q3 - Q1
            异常点比例 = np.sum((样本距离 > Q3 + 1.5 * IQR) | (样本距离 < Q1 - 1.5 * IQR)) / 样本数量
        else:
            异常点比例 = 0

        # 组合所有特征（精简版，14个核心特征）
        增强特征 = np.array([
            中心偏移距离,
            分布半径,
            样本数量,
            聚类密度,
            距离均值,
            距离标准差,
            特征均值[0] if len(特征均值) > 0 else 0,  # P1 均值
            特征均值[4] if len(特征均值) > 4 else 0,  # P5 偏斜度(原始三阶矩)
            特征均值[15] if len(特征均值) > 15 else 0,  # P16 峭度指标(标准化)
            特征标准差.mean() if len(特征标准差) > 0 else 0,
            特征变异系数.mean() if len(特征变异系数) > 0 else 0,
            平均相关性,
            异常点比例,
            主成分方差比[0] if len(主成分方差比) > 0 else 0,
        ])

        return 增强特征
    except Exception as e:
        return np.full(14, np.nan)


def 处理单个10min窗口(args):
    """处理单个10min窗口的函数（用于并行处理）"""
    窗口开始时间, 窗口事件, scaler, 基准中心 = args
    global 全局传感器ID, 全局传感器名称

    if len(窗口事件) > 0:
        # 提取29维特征（不包括标签）
        特征列表 = []
        标签列表 = []  # 收集标签用于统计
        for event in 窗口事件:
            if 'features' in event and len(event['features']) >= 29:
                特征列表.append(event['features'][:29])
                标签列表.append(event.get('label', 0))

        if len(特征列表) >= 3:  # 至少需要3个样本
            X = np.vstack(特征列表)

            # 移除NaN和Inf
            X = X[~np.isnan(X).any(axis=1)]
            X = X[~np.isinf(X).any(axis=1)]

            if len(X) >= 3:
                try:
                    X_scaled = scaler.transform(X)
                    当前中心 = X_scaled.mean(axis=0)

                    # 计算增强特征（不包含标签相关）
                    增强特征 = 计算增强特征_无标签(X_scaled, 当前中心, 基准中心)

                    # 窗口统计
                    事件数量 = len(窗口事件)
                    列车事件数 = sum(1 for event in 窗口事件 if event.get('事件类型') == '列车事件')
                    非列车事件数 = 事件数量 - 列车事件数
                    平均持续时间 = np.mean([event['duration_seconds'] for event in 窗口事件]) if 事件数量 > 0 else 0

                    # 传感器信息统一使用从文件中读取的值
                    global 全局传感器ID, 全局传感器名称
                    if 全局传感器ID and 全局传感器名称:
                        主要传感器 = f"{全局传感器ID}_{全局传感器名称}"
                    else:
                        主要传感器 = "NJM-ACA-C13-02_下游侧NMC27索索力"

                    # 标签统计（仅用于分析，不用于聚类）
                    标签均值 = np.mean(标签列表) if 标签列表 else 0
                    标签标准差 = np.std(标签列表) if len(标签列表) > 1 else 0

                    窗口指标 = {
                        '窗口开始时间': 窗口开始时间,
                        '事件数量': 事件数量,
                        '列车事件数量': 列车事件数,
                        '非列车事件数量': 非列车事件数,
                        '平均持续时间': 平均持续时间,
                        '主要传感器': 主要传感器,  # 传感器信息
                        '标签均值': 标签均值,  # 仅用于分析
                        '标签标准差': 标签标准差,  # 仅用于分析
                        '中心偏移距离': 增强特征[0],
                        '分布半径': 增强特征[1],
                        '样本数量': 增强特征[2],
                        '聚类密度': 增强特征[3],
                        '距离均值': 增强特征[4],
                        '距离标准差': 增强特征[5],
                        'P1均值': 增强特征[6],
                        'P5偏斜度': 增强特征[7],
                        'P16峭度指标': 增强特征[8],
                        '特征标准差平均': 增强特征[9],
                        '变异系数平均': 增强特征[10],
                        '平均相关性': 增强特征[11],
                        '异常点比例': 增强特征[12],
                        '主成分1方差比': 增强特征[13]
                    }

                    return 窗口指标
                except Exception as e:
                    print(f"⚠️ 处理窗口 {窗口开始时间} 时出错: {e}")
                    pass

    # 如果处理失败或事件不足
    事件数量 = len(窗口事件)
    列车事件数 = sum(1 for event in 窗口事件 if event.get('事件类型') == '列车事件')
    非列车事件数 = 事件数量 - 列车事件数
    平均持续时间 = np.mean([event['duration_seconds'] for event in 窗口事件]) if 事件数量 > 0 else 0

    # 传感器信息统一使用从文件中读取的值
    if 全局传感器ID and 全局传感器名称:
        主要传感器 = f"{全局传感器ID}_{全局传感器名称}"
    else:
        主要传感器 = "NJM-ACA-C13-02_下游侧NMC27索索力"

    窗口指标 = {
        '窗口开始时间': 窗口开始时间,
        '事件数量': 事件数量,
        '列车事件数量': 列车事件数,
        '非列车事件数量': 非列车事件数,
        '平均持续时间': 平均持续时间,
        '主要传感器': 主要传感器,
        '标签均值': np.nan,
        '标签标准差': np.nan,
        '中心偏移距离': np.nan,
        '分布半径': np.nan,
        '样本数量': 0,
        '聚类密度': np.nan,
        '距离均值': np.nan,
        '距离标准差': np.nan,
        'P1均值': np.nan,
        'P5偏斜度': np.nan,
        'P16峭度指标': np.nan,
        '特征标准差平均': np.nan,
        '变异系数平均': np.nan,
        '平均相关性': np.nan,
        '异常点比例': np.nan,
        '主成分1方差比': np.nan
    }

    return 窗口指标


# =========================
# 原有主要处理函数（保持不变）
# =========================
def 加载所有事件数据_并行(开始日期文件夹=None, 结束日期文件夹=None):
    """并行加载列车事件和非列车事件数据（按日期文件夹过滤）"""
    print("📂 并行加载所有事件数据...")
    global 全局传感器ID, 全局传感器名称
    # 定义事件类型目录
    事件类型目录 = {
        '列车事件': os.path.join(事件切片根目录, "Min_列车事件"),
        '非列车事件': os.path.join(事件切片根目录, "Min_非列车事件")
    }

    所有事件 = []

    for 事件类型, 目录路径 in 事件类型目录.items():
        if os.path.exists(目录路径):
            print(f"  🔍 扫描{事件类型}目录...")

            # 按日期文件夹过滤扫描JSON文件
            json_files = 扫描目录JSON文件_按日期过滤(目录路径, 开始日期文件夹, 结束日期文件夹)
            print(f"    找到 {len(json_files)} 个JSON文件")

            if json_files:
                # 并行读取文件
                事件列表 = 并行读取事件文件(json_files, 事件类型)
                所有事件.extend(事件列表)
        else:
            print(f"⚠️ 目录不存在: {目录路径}")

    # 按时间排序
    所有事件.sort(key=lambda x: x['start_time'])

    print(f"✅ 共加载 {len(所有事件)} 个事件")

    # 显示从文件中读取的传感器信息

    print(f"📡 从数据文件中读取的传感器信息: ID={全局传感器ID}, 名称={全局传感器名称}")

    return 所有事件


def 创建10min时间窗口(所有事件, 开始时间=None, 结束时间=None):
    """创建连续的10min时间窗口"""
    print("⏱️ 创建10min时间窗口...")

    # 确定时间范围
    if 开始时间 is None:
        开始时间 = min([event['start_time'] for event in 所有事件])
    if 结束时间 is None:
        结束时间 = max([event['end_time'] for event in 所有事件])

    # 将开始时间对齐到10min边界
    开始时间_对齐 = 获取10min窗口ID(开始时间)

    # 生成所有10min窗口
    窗口列表 = []
    当前时间 = 开始时间_对齐

    while 当前时间 <= 结束时间:
        窗口列表.append(当前时间)
        当前时间 += timedelta(minutes=10)

    print(f"✅ 创建了 {len(窗口列表)} 个10min窗口")
    print(f"  时间范围: {开始时间_对齐} 到 {窗口列表[-1]}")

    return 窗口列表


def 将事件分配到时间窗口_并行(所有事件, 窗口列表):
    """将事件分配到对应的10min窗口（并行版本）"""
    print("📊 并行分配事件到时间窗口...")

    # 初始化窗口字典
    窗口事件字典 = {窗口: [] for 窗口 in 窗口列表}

    # 预处理：为每个事件计算窗口ID
    print("  预处理事件...")
    for event in 所有事件:
        事件开始时间 = event['start_time']
        事件开始窗口 = 获取10min窗口ID(事件开始时间)

        if 事件开始窗口 in 窗口事件字典:
            窗口事件字典[事件开始窗口].append(event)

    # 统计
    有事件窗口数 = sum(1 for 窗口 in 窗口列表 if len(窗口事件字典[窗口]) > 0)
    总事件数 = sum(len(events) for events in 窗口事件字典.values())

    print(f"✅ 分配完成:")
    print(f"  总事件数: {总事件数}")
    print(f"  有事件窗口数: {有事件窗口数}/{len(窗口列表)} ({有事件窗口数 / len(窗口列表) * 100:.1f}%)")

    return 窗口事件字典


def 建立基准特征_10min窗口(窗口事件字典, 窗口列表):
    """建立基准特征用于标准化 - 优先选择列车事件≥2的10min窗口"""
    print("🎯 建立基准特征（10min窗口，列车事件≥2优先策略）...")

    # 策略1: 寻找第一个列车事件数量≥2的窗口
    基准窗口 = None
    for 窗口 in 窗口列表:
        窗口事件 = 窗口事件字典[窗口]
        列车事件数 = sum(1 for event in 窗口事件 if event.get('事件类型') == '列车事件')

        if 列车事件数 >= 2:
            基准窗口 = 窗口
            print(f"📌 找到符合条件的基准窗口: {窗口} (列车事件: {列车事件数})")
            break

    # 策略2: 如果没找到，寻找列车事件最多的窗口
    if 基准窗口 is None:
        print("⚠️ 未找到列车事件≥2的窗口，寻找列车事件最多的窗口...")
        最大列车事件数 = 0
        最佳窗口 = None

        for 窗口 in 窗口列表:
            窗口事件 = 窗口事件字典[窗口]
            列车事件数 = sum(1 for event in 窗口事件 if event.get('事件类型') == '列车事件')

            if 列车事件数 > 最大列车事件数:
                最大列车事件数 = 列车事件数
                最佳窗口 = 窗口

        if 最佳窗口 and 最大列车事件数 > 0:
            基准窗口 = 最佳窗口
            print(f"📌 使用列车事件最多的窗口作为基准: {基准窗口} (列车事件: {最大列车事件数})")
        else:
            # 策略3: 使用第一个有事件的窗口
            print("⚠️ 没有找到列车事件，使用第一个有事件的窗口")
            for 窗口 in 窗口列表:
                if len(窗口事件字典[窗口]) > 0:
                    基准窗口 = 窗口
                    print(f"📌 使用第一个有事件的窗口作为基准: {窗口}")
                    break

    # 如果所有策略都失败，使用第一个窗口
    if 基准窗口 is None:
        基准窗口 = 窗口列表[0]
        print(f"📌 所有窗口都无事件，使用第一个窗口作为基准: {基准窗口}")

    # 收集基准特征（只使用29维，不包括标签）
    基准特征列表 = []
    基准窗口事件 = 窗口事件字典[基准窗口]

    print(f"  从基准窗口收集特征...")

    # 先收集列车事件的特征
    for event in 基准窗口事件:
        if event.get('事件类型') == '列车事件' and 'features' in event and len(event['features']) >= 29:
            基准特征列表.append(event['features'][:29])

    # 如果列车事件特征不足，补充非列车事件
    if len(基准特征列表) < 10:
        print(f"  基准窗口列车事件特征不足 ({len(基准特征列表)})，补充其他事件")
        for event in 基准窗口事件:
            if event.get('事件类型') != '列车事件' and 'features' in event and len(event['features']) >= 29:
                基准特征列表.append(event['features'][:29])
                if len(基准特征列表) >= 20:  # 目标收集20个样本
                    break

    # 如果仍然不足，从相邻窗口补充
    if len(基准特征列表) < 10:
        print(f"  基准窗口特征不足 ({len(基准特征列表)})，从相邻窗口补充")

        # 找到基准窗口的索引
        基准索引 = 窗口列表.index(基准窗口)

        # 向前后各检查5个窗口
        for i in range(1, 6):
            # 向前搜索
            if 基准索引 - i >= 0:
                前窗口 = 窗口列表[基准索引 - i]
                for event in 窗口事件字典[前窗口]:
                    if 'features' in event and len(event['features']) >= 29:
                        基准特征列表.append(event['features'][:29])
                        if len(基准特征列表) >= 30:
                            break

            # 向后搜索
            if 基准索引 + i < len(窗口列表) and len(基准特征列表) < 30:
                后窗口 = 窗口列表[基准索引 + i]
                for event in 窗口事件字典[后窗口]:
                    if 'features' in event and len(event['features']) >= 29:
                        基准特征列表.append(event['features'][:29])
                        if len(基准特征列表) >= 30:
                            break

            if len(基准特征列表) >= 30:
                break

    # 如果仍然非常少，创建增强特征
    if len(基准特征列表) < 5:
        print("⚠️ 可用特征非常少，创建增强基准")

        if len(基准特征列表) > 0:
            # 基于现有特征生成增强版本
            现有特征 = np.vstack(基准特征列表)
            特征均值 = np.mean(现有特征, axis=0)
            特征标准差 = np.std(现有特征, axis=0)

            # 生成增强特征（添加噪声）
            for _ in range(20 - len(基准特征列表)):
                增强特征 = 特征均值 + np.random.normal(0, 特征标准差 * 0.1, 29)
                基准特征列表.append(增强特征)
        else:
            # 创建虚拟特征
            print("   创建虚拟基准特征")
            for _ in range(20):
                虚拟特征 = np.random.normal(0, 1, 29)
                基准特征列表.append(虚拟特征)

    基准特征 = np.vstack(基准特征列表)

    # 标准化器
    scaler = StandardScaler()
    基准特征_scaled = scaler.fit_transform(基准特征)
    基准中心 = 基准特征_scaled.mean(axis=0)

    # 记录基准信息
    global 全局传感器ID, 全局传感器名称
    基准信息 = {
        '基准窗口': 基准窗口.strftime('%Y-%m-%d %H:%M:%S'),
        '基准窗口列车事件数': sum(1 for event in 基准窗口事件 if event.get('事件类型') == '列车事件'),
        '基准窗口总事件数': len(基准窗口事件),
        '基准特征样本数': len(基准特征),
        '基准中心形状': 基准中心.shape,
        '选择理由': '第一个列车事件≥2的10min窗口' if sum(
            1 for event in 基准窗口事件 if event.get('事件类型') == '列车事件') >= 2
        else '列车事件最多的10min窗口' if sum(1 for event in 基准窗口事件 if event.get('事件类型') == '列车事件') > 0
        else '第一个有事件的10min窗口',
        '特征维度': 29,
        '特征名称': 特征名称,
        '传感器信息': {
            'sensor_id': 全局传感器ID or 'NJM-ACA-C13-02',
            'sensor_name': 全局传感器名称 or '下游侧NMC27索索力',
            'source': '从JSON数据文件中读取'
        }
    }

    print(f"✅ 基准建立完成:")
    print(f"   基准窗口: {基准信息['基准窗口']}")
    print(f"   基准窗口列车事件: {基准信息['基准窗口列车事件数']}")
    print(f"   基准窗口总事件: {基准信息['基准窗口总事件数']}")
    print(f"   基准特征样本数: {基准信息['基准特征样本数']}")
    print(f"   特征维度: {基准信息['特征维度']}")
    print(f"   选择理由: {基准信息['选择理由']}")
    print(f"   传感器信息: {基准信息['传感器信息']['sensor_id']}_{基准信息['传感器信息']['sensor_name']}")

    return scaler, 基准中心, 基准信息


def 并行计算窗口聚类指标(窗口事件字典, 窗口列表, scaler, 基准中心):
    """并行计算每个窗口的聚类指标"""
    print("🧮 并行计算窗口聚类指标...")
    print(f"  使用 {min(cpu_count(), 4)} 个CPU核心")

    # 准备参数列表
    tasks = []
    for 窗口 in 窗口列表:
        tasks.append((窗口, 窗口事件字典[窗口], scaler, 基准中心))

    窗口指标列表 = []
    处理完成数 = 0

    # 使用进程池并行处理
    with Pool(processes=min(cpu_count(), 4)) as pool:
        # 使用imap_unordered提高性能
        results = pool.imap_unordered(处理单个10min窗口, tasks, chunksize=50)

        for result in results:
            窗口指标列表.append(result)
            处理完成数 += 1

            if 处理完成数 % 500 == 0:
                print(f"  进度: {处理完成数}/{len(窗口列表)}")

    # 按时间排序
    窗口指标列表.sort(key=lambda x: x['窗口开始时间'])

    print(f"✅ 完成 {len(窗口指标列表)} 个窗口的计算")
    return 窗口指标列表


def 添加时间特征(df):
    """添加时间相关特征（适配10min窗口）"""
    print("⏰ 添加时间特征...")

    df['窗口开始时间'] = pd.to_datetime(df['窗口开始时间'])

    # 基本时间特征
    df['年'] = df['窗口开始时间'].dt.year
    df['月'] = df['窗口开始时间'].dt.month
    df['日'] = df['窗口开始时间'].dt.day
    df['小时'] = df['窗口开始时间'].dt.hour
    df['十分钟块'] = (df['窗口开始时间'].dt.minute // 10) * 10  # 0, 10, 20, 30, 40, 50
    df['星期几'] = df['窗口开始时间'].dt.dayofweek  # 0=周一, 6=周日
    df['是否工作日'] = df['星期几'].apply(lambda x: 1 if x < 5 else 0)
    df['是否周末'] = df['星期几'].apply(lambda x: 1 if x >= 5 else 0)

    # 一天中的时间段（基于10min更精细）
    def 获取时间段(小时, 十分钟块):
        if 0 <= 小时 < 6:
            return '深夜'
        elif 6 <= 小时 < 12:
            if 小时 == 11 and 十分钟块 >= 30:
                return '中午'
            return '上午'
        elif 12 <= 小时 < 18:
            if 小时 == 12 and 十分钟块 < 30:
                return '中午'
            return '下午'
        else:
            return '晚上'

    df['时间段'] = df.apply(lambda row: 获取时间段(row['小时'], row['十分钟块']), axis=1)

    # 时间段编码
    时间段映射 = {'深夜': 0, '上午': 1, '中午': 2, '下午': 3, '晚上': 4}
    df['时间段编码'] = df['时间段'].map(时间段映射)

    # 是否高峰时段（更精细的10min判断）
    def 是否高峰时段(小时, 十分钟块, 星期几):
        if 星期几 < 5:  # 工作日
            # 早高峰：7:00-9:30
            if (7 <= 小时 <= 9) and not (小时 == 9 and 十分钟块 > 30):
                return 1
            # 晚高峰：17:00-19:30
            elif (17 <= 小时 <= 19) and not (小时 == 19 and 十分钟块 > 30):
                return 1
            else:
                return 0
        else:  # 周末
            # 周末高峰：10:00-12:30, 18:00-20:30
            if (10 <= 小时 <= 12) and not (小时 == 12 and 十分钟块 > 30):
                return 1
            elif (18 <= 小时 <= 20) and not (小时 == 20 and 十分钟块 > 30):
                return 1
            else:
                return 0

    df['是否高峰时段'] = df.apply(lambda row: 是否高峰时段(row['小时'], row['十分钟块'], row['星期几']), axis=1)

    # 是否夜间
    df['是否夜间'] = df['小时'].apply(lambda x: 1 if 22 <= x <= 23 or 0 <= x <= 5 else 0)

    return df


def 处理缺失值(df):
    """处理缺失值"""
    print("🔧 处理缺失值...")

    # 统计缺失值
    缺失统计 = df.isnull().sum()
    print("缺失值统计:")
    for col in df.columns:
        if 缺失统计[col] > 0:
            print(f"  {col}: {int(缺失统计[col])} 个缺失值 ({float(缺失统计[col]) / len(df) * 100:.1f}%)")

    # 确保数据按时间排序
    df = df.sort_values('窗口开始时间')

    # 对于聚类指标，使用线性插值
    聚类指标列 = ['中心偏移距离', '分布半径', '样本数量', '聚类密度', '距离均值',
                  '距离标准差', 'P1均值', 'P5偏斜度', 'P16峭度指标',
                  '特征标准差平均', '变异系数平均', '平均相关性',
                  '异常点比例', '主成分1方差比', '标签均值', '标签标准差']

    for col in 聚类指标列:
        if col in df.columns:
            # 使用线性插值
            df[col] = df[col].interpolate(method='linear', limit_direction='both', limit=10)

            # 对于两端仍然缺失的值，使用前向/后向填充
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

            # 确保数值类型正确
            if col in ['样本数量', '标签均值']:
                df[col] = df[col].astype(float)

    # 对于其他数值列，使用0填充
    for col in df.select_dtypes(include=[np.number]).columns:
        if col not in 聚类指标列 and df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(0)

    # 处理主要传感器列 - 直接设置为统一值
    if '主要传感器' in df.columns:
        global 全局传感器ID, 全局传感器名称
        if 全局传感器ID and 全局传感器名称:
            # 使用从文件中读取的传感器信息
            df['主要传感器'] = f"{全局传感器ID}_{全局传感器名称}"
        else:
            # 使用默认值
            df['主要传感器'] = "NJM-ACA-C13-02_下游侧NMC27索索力"

    # 确保所有数值列都是Python原生类型
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].dtype in [np.int64, np.int32]:
            df[col] = df[col].astype(int)
        elif df[col].dtype in [np.float64, np.float32]:
            df[col] = df[col].astype(float)

    return df

# =========================
# 原有数据质量分析函数（保持不变）
# =========================
def 数据质量分析(df, 基准信息, 时间戳目录, 开始日期文件夹, 结束日期文件夹):
    """分析数据质量并生成报告"""
    print("📊 数据质量分析...")

    # 创建分析报告
    报告内容 = []
    报告内容.append("=" * 80)
    报告内容.append("10min级特征工程数据质量分析报告（29维特征并行版）")
    报告内容.append("=" * 80)
    报告内容.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    报告内容.append(f"CPU核心数: {cpu_count()}")
    报告内容.append(f"数据总行数（窗口数）: {len(df)}")
    报告内容.append(f"数据总列数: {len(df.columns)}")
    报告内容.append("")

    # 基准信息
    报告内容.append("1. 基准信息:")
    报告内容.append(f"   基准窗口: {基准信息['基准窗口']}")
    报告内容.append(f"   基准窗口列车事件数: {基准信息['基准窗口列车事件数']}")
    报告内容.append(f"   基准窗口总事件数: {基准信息['基准窗口总事件数']}")
    报告内容.append(f"   基准特征样本数: {基准信息['基准特征样本数']}")
    报告内容.append(f"   特征维度: {基准信息['特征维度']}")
    报告内容.append(f"   选择理由: {基准信息['选择理由']}")
    报告内容.append(f"   传感器信息: {基准信息['传感器信息']['sensor_id']}_{基准信息['传感器信息']['sensor_name']}")
    报告内容.append(f"   传感器信息来源: {基准信息['传感器信息']['source']}")
    报告内容.append("")

    # 29维特征信息
    报告内容.append("2. 29维特征信息:")
    报告内容.append("   特征列表:")
    for i, name in enumerate(特征名称, 1):
        报告内容.append(f"     P{i:2d}: {name}")
    报告内容.append("")

    # 基本信息
    报告内容.append("3. 数据基本信息:")
    报告内容.append(f"   时间范围: {df['窗口开始时间'].min()} 到 {df['窗口开始时间'].max()}")
    报告内容.append(f"   时间跨度: {(df['窗口开始时间'].max() - df['窗口开始时间'].min()).days} 天")
    报告内容.append(f"   10min窗口数: {len(df)}")
    报告内容.append("")

    # 事件统计
    报告内容.append("4. 事件统计:")
    总事件数 = df['事件数量'].sum()
    有事件窗口数 = (df['事件数量'] > 0).sum()
    总列车事件数 = df['列车事件数量'].sum()
    总非列车事件数 = df['非列车事件数量'].sum()

    报告内容.append(f"   总事件数: {int(总事件数)}")
    报告内容.append(f"   总列车事件数: {int(总列车事件数)}")
    报告内容.append(f"   总非列车事件数: {int(总非列车事件数)}")
    报告内容.append(f"   有事件窗口数: {有事件窗口数} ({有事件窗口数 / len(df) * 100:.1f}%)")
    报告内容.append(f"   平均每窗口事件数: {总事件数 / len(df):.2f}")
    报告内容.append(f"   平均每窗口列车事件数: {总列车事件数 / len(df):.2f}")
    报告内容.append(f"   平均每窗口非列车事件数: {总非列车事件数 / len(df):.2f}")
    报告内容.append(f"   标签均值（列车事件比例）: {df['标签均值'].mean():.3f}")
    报告内容.append("")

    # 聚类指标统计
    报告内容.append("5. 聚类指标统计:")
    关键特征 = ['中心偏移距离', '分布半径', '聚类密度', 'P1均值', 'P5偏斜度', 'P16峭度指标']
    for feat in 关键特征:
        if feat in df.columns:
            非空值 = df[feat].dropna()
            if len(非空值) > 0:
                报告内容.append(f"   {feat}:")
                报告内容.append(f"     有效值数量: {len(非空值)} ({len(非空值) / len(df) * 100:.1f}%)")
                报告内容.append(f"     均值: {非空值.mean():.4f}")
                报告内容.append(f"     标准差: {非空值.std():.4f}")
                报告内容.append(f"     最小值: {非空值.min():.4f}")
                报告内容.append(f"     中位数: {非空值.median():.4f}")
                报告内容.append(f"     最大值: {非空值.max():.4f}")
                报告内容.append("")

    # 时间分布统计
    报告内容.append("6. 时间分布统计:")
    报告内容.append(f"   工作日比例: {df['是否工作日'].mean() * 100:.1f}%")
    报告内容.append(f"   高峰时段比例: {df['是否高峰时段'].mean() * 100:.1f}%")
    报告内容.append(f"   夜间时段比例: {df['是否夜间'].mean() * 100:.1f}%")
    报告内容.append(f"   时间段分布:")
    time_dist = df['时间段'].value_counts()
    for time_period, count in time_dist.items():
        报告内容.append(f"     {time_period}: {count} ({count / len(df) * 100:.1f}%)")

    # 保存报告到统一目录
    报告目录 = os.path.join(特征工程结果根目录, "2_3_reports")
    时间范围 = f"{开始日期文件夹}_to_{结束日期文件夹}"
    时间范围目录 = os.path.join(报告目录, 时间范围)
    os.makedirs(时间范围目录, exist_ok=True)

    报告路径 = os.path.join(时间范围目录, "数据质量分析报告.txt")
    with open(报告路径, 'w', encoding='utf-8') as f:
        f.write('\n'.join(报告内容))

    print(f"✅ 数据质量报告已保存到: {报告路径}")

    # 同时保存到时间戳目录
    时间戳报告路径 = os.path.join(时间戳目录, "数据质量分析报告.txt")
    with open(时间戳报告路径, 'w', encoding='utf-8') as f:
        f.write('\n'.join(报告内容))

    # 生成可视化图表
    生成数据质量图表(df, 基准信息, 时间戳目录, 开始日期文件夹, 结束日期文件夹)

    return 报告内容


def 生成数据质量图表(df, 基准信息, 时间戳目录, 开始日期文件夹, 结束日期文件夹):
    """生成数据质量可视化图表"""
    print("📈 生成数据质量图表...")

    # 设置图表样式
    plt.style.use('seaborn-v0_8-darkgrid')

    # 确保字体已正确设置
    设置中文字体()

    # 创建图表保存目录
    报告目录 = os.path.join(特征工程结果根目录, "2_3_reports")
    时间范围 = f"{开始日期文件夹}_to_{结束日期文件夹}"
    时间范围目录 = os.path.join(报告目录, 时间范围)
    os.makedirs(时间范围目录, exist_ok=True)

    # 1. 时间序列图
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))

    # 子图1: 事件数量趋势
    axes[0, 0].plot(df['窗口开始时间'], df['事件数量'], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('事件数量时间序列 (10min窗口)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('时间')
    axes[0, 0].set_ylabel('事件数量')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 子图2: 标签均值趋势
    有效标签均值 = df['标签均值'].dropna()
    if len(有效标签均值) > 0:
        axes[0, 1].plot(df['窗口开始时间'], df['标签均值'], 'r-', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('标签均值时间序列 (列车事件比例)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('时间')
        axes[0, 1].set_ylabel('标签均值 (0-1)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)

    # 子图3: 中心偏移距离
    有效中心偏移 = df['中心偏移距离'].dropna()
    if len(有效中心偏移) > 0:
        axes[1, 0].plot(df['窗口开始时间'], df['中心偏移距离'], 'g-', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('中心偏移距离时间序列', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('时间')
        axes[1, 0].set_ylabel('中心偏移距离')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)

    # 子图4: 分布半径
    有效分布半径 = df['分布半径'].dropna()
    if len(有效分布半径) > 0:
        axes[1, 1].plot(df['窗口开始时间'], df['分布半径'], 'orange', alpha=0.7, linewidth=1)
        axes[1, 1].set_title('分布半径时间序列', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('时间')
        axes[1, 1].set_ylabel('分布半径')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)

    # 子图5: P1均值特征趋势
    有效P1均值 = df['P1均值'].dropna()
    if len(有效P1均值) > 0:
        axes[2, 0].plot(df['窗口开始时间'], df['P1均值'], 'purple', alpha=0.7, linewidth=1)
        axes[2, 0].set_title('P1均值（信号均值）时间序列', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('时间')
        axes[2, 0].set_ylabel('P1均值')
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].tick_params(axis='x', rotation=45)

    # 子图6: P5偏斜度特征趋势
    有效P5偏斜度 = df['P5偏斜度'].dropna()
    if len(有效P5偏斜度) > 0:
        axes[2, 1].plot(df['窗口开始时间'], df['P5偏斜度'], 'brown', alpha=0.7, linewidth=1)
        axes[2, 1].set_title('P5偏斜度（原始三阶矩）时间序列', fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('时间')
        axes[2, 1].set_ylabel('P5偏斜度')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].tick_params(axis='x', rotation=45)

    plt.suptitle(
        f'10min级特征工程数据质量分析（29维特征）\n基准窗口: {基准信息["基准窗口"]}\n设备: {基准信息["传感器信息"]["sensor_id"]}_{基准信息["传感器信息"]["sensor_name"]}',
        fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    图表路径1 = os.path.join(时间范围目录, "数据质量分析图_时间序列.png")
    plt.savefig(图表路径1, dpi=300, bbox_inches='tight')

    # 同时保存到时间戳目录
    时间戳图表路径1 = os.path.join(时间戳目录, "数据质量分析图_时间序列.png")
    plt.savefig(时间戳图表路径1, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 分布和相关性图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 子图1: 29维特征分布直方图
    for i, col in enumerate(['P1均值', 'P5偏斜度', 'P16峭度指标', '标签均值']):
        if col in df.columns:
            非空值 = df[col].dropna()
            if len(非空值) > 0:
                row, col_idx = divmod(i, 2)
                axes[row, col_idx].hist(非空值, bins=50, alpha=0.7, color=['blue', 'red', 'green', 'purple'][i])
                axes[row, col_idx].set_title(f'{col}分布', fontsize=12, fontweight='bold')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('频次')
                axes[row, col_idx].grid(True, alpha=0.3)

    plt.suptitle(f'29维关键特征分布\n特征维度: {基准信息["特征维度"]}', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    图表路径2 = os.path.join(时间范围目录, "数据质量分析图_特征分布.png")
    plt.savefig(图表路径2, dpi=300, bbox_inches='tight')

    # 同时保存到时间戳目录
    时间戳图表路径2 = os.path.join(时间戳目录, "数据质量分析图_特征分布.png")
    plt.savefig(时间戳图表路径2, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 数据质量图表已保存到: {图表路径1}, {图表路径2}")


# =========================
# 主程序
# =========================
# =========================
# 主程序
# =========================
# 在主程序开始处添加
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 10min级特征工程（29维特征并行版，统一存储目录）")
    print(f"🎯 CPU核心数: {cpu_count()}")
    print(f"📁 数据源目录: {事件切片根目录}")
    print(f"📁 统一存储目录: {特征工程结果根目录}")

    # 用户可配置的开始和结束日期文件夹
    开始日期文件夹 = "2023-03-23"  # 例如: "2023-03-23"
    结束日期文件夹 = "2023-04-23"  # 例如: "2023-04-23"

    print("=" * 80)

    开始总时间 = time.time()

    try:
        # 步骤1: 并行加载所有事件数据（按日期文件夹过滤）
        print("\n📂 步骤1: 并行加载所有事件数据")
        所有事件 = 加载所有事件数据_并行(开始日期文件夹, 结束日期文件夹)

        if len(所有事件) == 0:
            print("❌ 未加载到任何事件数据")
            exit(1)

        # ✅ 延迟创建时间戳目录：确保有数据后再创建
        时间戳目录 = os.path.join(
            特征工程结果根目录,
            f"2_10min级特征_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(时间戳目录, exist_ok=True)
        print(f"📁 本次运行目录: {时间戳目录}")

        if len(所有事件) == 0:
            print("❌ 未加载到任何事件数据")
            exit(1)

        # 步骤2: 创建10min时间窗口
        print("\n⏱️ 步骤2: 创建10min时间窗口")
        窗口列表 = 创建10min时间窗口(所有事件)

        # 步骤3: 分配事件到窗口
        print("\n📊 步骤3: 将事件分配到时间窗口")
        窗口事件字典 = 将事件分配到时间窗口_并行(所有事件, 窗口列表)

        # 步骤4: 建立基准特征（列车事件≥2优先）
        print("\n🎯 步骤4: 建立基准特征（列车事件≥2优先策略）")
        scaler, 基准中心, 基准信息 = 建立基准特征_10min窗口(窗口事件字典, 窗口列表)

        # 步骤5: 并行计算窗口聚类指标
        print("\n🧮 步骤5: 并行计算窗口聚类指标")
        窗口指标列表 = 并行计算窗口聚类指标(窗口事件字典, 窗口列表, scaler, 基准中心)

        # 步骤6: 保存原始窗口特征
        print("\n💾 步骤6: 保存原始窗口特征")
        原始特征目录 = 保存原始窗口特征(窗口事件字典, 窗口列表, 时间戳目录, 开始日期文件夹, 结束日期文件夹)

        # 步骤7: 创建DataFrame
        print("\n📋 步骤7: 创建DataFrame")
        df = pd.DataFrame(窗口指标列表)

        # 步骤8: 添加时间特征
        print("\n⏰ 步骤8: 添加时间特征")
        df = 添加时间特征(df)

        # 步骤9: 处理缺失值
        print("\n🔧 步骤9: 处理缺失值")
        df = 处理缺失值(df)

        # 步骤10: 保存时间序列数据
        print("\n💾 步骤10: 保存时间序列数据")
        csv_path = 保存时间序列数据(df, 时间戳目录, 开始日期文件夹, 结束日期文件夹)

        # 步骤11: 保存基准信息
        print("\n💾 步骤11: 保存基准信息")
        基准路径 = 保存基准信息(基准信息, 时间戳目录, 开始日期文件夹, 结束日期文件夹)

        # 步骤12: 保存标准化器和基准中心
        print("\n💾 步骤12: 保存标准化器和基准中心")
        标准化器目录 = os.path.join(特征工程结果根目录, "2_4_benchmark")
        os.makedirs(标准化器目录, exist_ok=True)

        # 保存标准化器
        标准化器路径 = os.path.join(标准化器目录, f"scaler_{开始日期文件夹}_to_{结束日期文件夹}.pkl")
        with open(标准化器路径, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ 标准化器已保存到: {标准化器路径}")

        # 保存基准中心
        基准中心路径 = os.path.join(标准化器目录, f"benchmark_center_{开始日期文件夹}_to_{结束日期文件夹}.npy")
        np.save(基准中心路径, 基准中心)
        print(f"✅ 基准中心已保存到: {基准中心路径}")

        # 保存特征名称
        特征名称路径 = os.path.join(标准化器目录, "29维特征名称.json")
        with open(特征名称路径, 'w', encoding='utf-8') as f:
            json.dump(特征名称, f, ensure_ascii=False, indent=2)
        print(f"✅ 29维特征名称已保存到: {特征名称路径}")

        # 步骤13: 数据质量分析
        print("\n📊 步骤13: 数据质量分析")
        数据质量分析(df, 基准信息, 时间戳目录, 开始日期文件夹, 结束日期文件夹)

        # 步骤14: 生成汇总报告
        print("\n📋 步骤14: 生成汇总报告")
        报告路径 = 生成汇总报告(df, 基准信息, 时间戳目录, 开始日期文件夹, 结束日期文件夹)

        # 生成最终汇总信息
        结束总时间 = time.time()
        总耗时 = 结束总时间 - 开始总时间

        print("\n" + "=" * 80)
        print("🎉 处理完成!")
        print("=" * 80)
        print(f"📊 处理统计:")
        print(f"   总耗时: {总耗时:.1f} 秒")
        print(f"   10min窗口数: {len(df)}")
        print(f"   时间范围: {df['窗口开始时间'].min()} 到 {df['窗口开始时间'].max()}")
        print(f"   总事件数: {int(df['事件数量'].sum())}")
        print(f"   总列车事件数: {int(df['列车事件数量'].sum())}")
        print(f"   总非列车事件数: {int(df['非列车事件数量'].sum())}")
        print(f"   传感器信息: {基准信息['传感器信息']['sensor_id']}_{基准信息['传感器信息']['sensor_name']}")
        print(f"   统一存储目录: {特征工程结果根目录}")
        print(f"   本次运行目录: {时间戳目录}")
        print("=" * 80)


    except Exception as e:

        print(f"❌ 程序执行失败: {e}")

        traceback.print_exc()

        # 保存错误信息到特征工程根目录的错误文件夹

        错误目录 = os.path.join(特征工程结果根目录, "errors")

        os.makedirs(错误目录, exist_ok=True)

        错误文件路径 = os.path.join(错误目录, f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        with open(错误文件路径, 'w', encoding='utf-8') as f:

            f.write(f"错误时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            f.write(f"开始日期: {开始日期文件夹}\n")

            f.write(f"结束日期: {结束日期文件夹}\n")

            f.write(f"错误信息: {str(e)}\n")

            f.write("错误堆栈:\n")

            traceback.print_exc(file=f)

        print(f"📝 错误信息已保存到: {错误文件路径}")