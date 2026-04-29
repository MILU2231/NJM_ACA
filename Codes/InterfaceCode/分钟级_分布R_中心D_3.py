# -*- coding: utf-8 -*-
"""
10min级特征工程（从事件切片JSON文件） - 多进程并行版
读取列车事件和非列车事件的JSON文件，按10min窗口聚合并计算聚类指标
使用多进程加速处理，基准选择：第一个列车事件≥2的10min窗口
支持按日期文件夹过滤数据
传感器信息从数据文件中读取并统一应用
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
# 路径配置
# =========================
# 事件切片根目录（更新为新的路径）
事件切片根目录 = r"X:\NJM_Item\ACA_对接优化数据集\C13_ACA列车切片数据\C13_02列车事件切片"

# 结果保存路径
结果根目录 = r"X:\NJM_Item\ACA_对接优化数据集\10min级特征_聚类结果"

# 创建带时间戳的子目录
时间戳目录 = os.path.join(结果根目录, f"10min级特征_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(时间戳目录, exist_ok=True)

print(f"📁 结果将保存在: {时间戳目录}")

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
# 工具函数
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
    with ProcessPoolExecutor(max_workers=min(cpu_count(), 4)) as executor:
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
# 主要处理函数
# =========================
def 加载所有事件数据_并行(开始日期文件夹=None, 结束日期文件夹=None):
    """并行加载列车事件和非列车事件数据（按日期文件夹过滤）"""
    print("📂 并行加载所有事件数据...")

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
    global 全局传感器ID, 全局传感器名称
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
            print(f"  {col}: {缺失统计[col]} 个缺失值 ({缺失统计[col] / len(df) * 100:.1f}%)")

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

    return df


def 保存详细结果JSON(df, 基准信息, 时间戳目录):
    """保存详细的JSON格式结果，包含传感器信息"""
    print("💾 保存详细JSON结果...")

    global 全局传感器ID, 全局传感器名称

    结果数据 = {
        'metadata': {
            '生成时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '时间粒度': '10min',
            '数据源': 事件切片根目录,
            '基准信息': 基准信息,
            '总窗口数': len(df),
            '时间范围': {
                '开始': df['窗口开始时间'].min().strftime('%Y-%m-%d %H:%M:%S'),
                '结束': df['窗口开始时间'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            '设备信息': {
                'sensor_id': 全局传感器ID or 'NJM-ACA-C13-02',
                'sensor_name': 全局传感器名称 or '下游侧NMC27索索力',
                '备注': '传感器信息从数据文件中读取并统一应用'
            }
        },
        'features_description': {
            '中心偏移距离': '当前窗口特征中心与基准中心的欧氏距离',
            '分布半径': '窗口内样本到中心的95百分位距离',
            '聚类密度': '样本数量 / 分布半径',
            'P1均值': '信号均值特征',
            'P5偏斜度': '原始三阶矩特征',
            'P16峭度指标': '标准化峭度特征'
        },
        'data': []
    }

    # 添加前100个窗口的数据（避免JSON文件过大）
    for idx, row in df.head(100).iterrows():
        窗口数据 = {
            '窗口开始时间': row['窗口开始时间'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['窗口开始时间'],
                                                                                            datetime) else str(
                row['窗口开始时间']),
            '事件数量': int(row['事件数量']),
            '列车事件数量': int(row['列车事件数量']),
            '非列车事件数量': int(row['非列车事件数量']),
            '主要传感器': row['主要传感器'],
            '中心偏移距离': float(row['中心偏移距离']) if not pd.isna(row['中心偏移距离']) else None,
            '分布半径': float(row['分布半径']) if not pd.isna(row['分布半径']) else None,
            '聚类密度': float(row['聚类密度']) if not pd.isna(row['聚类密度']) else None,
            '时间特征': {
                '年': int(row['年']),
                '月': int(row['月']),
                '日': int(row['日']),
                '小时': int(row['小时']),
                '十分钟块': int(row['十分钟块']),
                '星期几': int(row['星期几']),
                '是否工作日': int(row['是否工作日']),
                '是否高峰时段': int(row['是否高峰时段'])
            }
        }
        结果数据['data'].append(窗口数据)

    # 保存JSON
    json_path = os.path.join(时间戳目录, "10min级时间序列_详细.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(结果数据, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ 详细JSON结果已保存到: {json_path}")

    # 同时保存简化版JSON（完整数据）
    简化数据 = df.to_dict(orient='records')
    简化路径 = os.path.join(时间戳目录, "10min级时间序列_完整.json")
    with open(简化路径, 'w', encoding='utf-8') as f:
        json.dump(简化数据, f, ensure_ascii=False, indent=2, default=str)

    print(f"✅ 完整JSON结果已保存到: {简化路径}")


def 数据质量分析(df, 基准信息):
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

    # 保存报告
    报告路径 = os.path.join(时间戳目录, "数据质量分析报告.txt")
    with open(报告路径, 'w', encoding='utf-8') as f:
        f.write('\n'.join(报告内容))

    print(f"✅ 数据质量报告已保存到: {报告路径}")

    # 生成可视化图表（不包含传感器分布图）
    生成数据质量图表(df, 基准信息)

    return 报告内容


def 生成数据质量图表(df, 基准信息):
    """生成数据质量可视化图表（不包含传感器分布）"""
    print("📈 生成数据质量图表...")

    # 设置图表样式
    plt.style.use('seaborn-v0_8-darkgrid')

    # 确保字体已正确设置
    设置中文字体()

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
    图表路径1 = os.path.join(时间戳目录, "数据质量分析图_时间序列.png")
    plt.savefig(图表路径1, dpi=300, bbox_inches='tight')
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
    图表路径2 = os.path.join(时间戳目录, "数据质量分析图_特征分布.png")
    plt.savefig(图表路径2, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 特征相关性热力图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 关键特征的相关性分析
    关键特征列表 = ['中心偏移距离', '分布半径', 'P1均值', 'P5偏斜度', 'P16峭度指标', '标签均值', '事件数量',
                    '列车事件数量']
    可用特征 = [feat for feat in 关键特征列表 if feat in df.columns]

    if len(可用特征) >= 3:
        # 计算相关性矩阵
        相关性矩阵 = df[可用特征].corr()

        # 子图1: 相关性热力图
        im = axes[0, 0].imshow(相关性矩阵, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[0, 0].set_title('关键特征相关性热力图', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticks(range(len(可用特征)))
        axes[0, 0].set_yticks(range(len(可用特征)))
        axes[0, 0].set_xticklabels(可用特征, rotation=45, ha='right')
        axes[0, 0].set_yticklabels(可用特征)
        plt.colorbar(im, ax=axes[0, 0])

        # 添加数值标签
        for i in range(len(可用特征)):
            for j in range(len(可用特征)):
                axes[0, 0].text(j, i, f'{相关性矩阵.iloc[i, j]:.2f}',
                                ha='center', va='center',
                                color='white' if abs(相关性矩阵.iloc[i, j]) > 0.5 else 'black', fontsize=8)

    # 子图2: 标签均值与事件数量的关系
    有效数据 = df.dropna(subset=['标签均值', '事件数量'])
    if len(有效数据) > 0:
        scatter = axes[0, 1].scatter(有效数据['事件数量'], 有效数据['标签均值'], alpha=0.5, s=20, c=有效数据['小时'],
                                     cmap='viridis')
        axes[0, 1].set_title('事件数量 vs 标签均值（按小时着色）', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('事件数量')
        axes[0, 1].set_ylabel('标签均值（列车事件比例）')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='小时')

    # 子图3: P1均值与P5偏斜度的关系
    有效数据 = df.dropna(subset=['P1均值', 'P5偏斜度'])
    if len(有效数据) > 0:
        scatter = axes[1, 0].scatter(有效数据['P1均值'], 有效数据['P5偏斜度'], alpha=0.5, s=20,
                                     c=有效数据['是否工作日'], cmap='coolwarm')
        axes[1, 0].set_title('P1均值 vs P5偏斜度（按工作日着色）', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('P1均值（信号均值）')
        axes[1, 0].set_ylabel('P5偏斜度（原始三阶矩）')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='工作日（1=是）')

    # 子图4: 按小时的平均标签均值
    axes[1, 1].bar(range(24), df.groupby('小时')['标签均值'].mean().reindex(range(24), fill_value=0), color='orange',
                   alpha=0.7)
    axes[1, 1].set_title('按小时的平均标签均值（列车事件比例）', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('小时')
    axes[1, 1].set_ylabel('平均标签均值')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xticks(range(0, 24, 2))

    plt.suptitle(
        f'29维特征相关性分析（10min窗口）\n基准选择理由: {基准信息["选择理由"]}\n设备: {基准信息["传感器信息"]["sensor_id"]}_{基准信息["传感器信息"]["sensor_name"]}',
        fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    图表路径3 = os.path.join(时间戳目录, "数据质量分析图_特征相关性.png")
    plt.savefig(图表路径3, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 数据质量图表已保存到: {图表路径1}, {图表路径2}, {图表路径3}")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 10min级特征工程（29维特征并行版，无标签聚类）")
    print(f"🎯 CPU核心数: {cpu_count()}")
    print(f"📁 数据源目录: {事件切片根目录}")
    print(f"📁 结果目录: {时间戳目录}")
    print("=" * 80)

    # 用户可配置的开始和结束日期文件夹
    # 格式: "2023-03-23" （文件夹名称格式）
    开始日期文件夹 = "2023-03-23"  # 例如: "2023-03-23"
    结束日期文件夹 = "2023-04-23"  # 例如: "2023-04-23"

    开始总时间 = time.time()

    try:
        # 步骤1: 并行加载所有事件数据（按日期文件夹过滤）
        print("\n📂 步骤1: 并行加载所有事件数据")
        所有事件 = 加载所有事件数据_并行(开始日期文件夹, 结束日期文件夹)

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

        # 保存基准信息
        基准信息路径 = os.path.join(时间戳目录, "基准信息.json")
        with open(基准信息路径, 'w', encoding='utf-8') as f:
            json.dump(基准信息, f, ensure_ascii=False, indent=2)
        print(f"✅ 基准信息已保存: {基准信息路径}")

        # 步骤5: 并行计算窗口聚类指标
        print("\n🧮 步骤5: 并行计算窗口聚类指标")
        窗口指标列表 = 并行计算窗口聚类指标(窗口事件字典, 窗口列表, scaler, 基准中心)

        # 步骤6: 创建DataFrame
        print("\n📋 步骤6: 创建DataFrame")
        df = pd.DataFrame(窗口指标列表)

        # 步骤7: 添加时间特征
        print("\n⏰ 步骤7: 添加时间特征")
        df = 添加时间特征(df)

        # 步骤8: 处理缺失值（传感器信息统一处理）
        print("\n🔧 步骤8: 处理缺失值")
        df = 处理缺失值(df)

        # 步骤9: 保存数据
        print("\n💾 步骤9: 保存数据")
        输出路径 = os.path.join(时间戳目录, "10min级时间序列.csv")
        df.to_csv(输出路径, index=False, encoding='utf-8-sig')
        print(f"✅ 10min级时间序列已保存到: {输出路径}")

        # 保存JSON文件（包含传感器信息）
        保存详细结果JSON(df, 基准信息, 时间戳目录)

        # 保存标准化器
        标准化器路径 = os.path.join(时间戳目录, "10min级标准化器.pkl")
        with open(标准化器路径, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ 标准化器已保存到: {标准化器路径}")

        # 保存基准中心
        基准中心路径 = os.path.join(时间戳目录, "基准中心.npy")
        np.save(基准中心路径, 基准中心)
        print(f"✅ 基准中心已保存到: {基准中心路径}")

        # 保存29维特征名称
        特征名称路径 = os.path.join(时间戳目录, "29维特征名称.json")
        with open(特征名称路径, 'w', encoding='utf-8') as f:
            json.dump(特征名称, f, ensure_ascii=False, indent=2)
        print(f"✅ 29维特征名称已保存到: {特征名称路径}")

        # 步骤10: 数据质量分析（不包含传感器分布图）
        print("\n📊 步骤10: 数据质量分析")
        数据质量分析(df, 基准信息)

        # 生成汇总信息
        结束总时间 = time.time()
        总耗时 = 结束总时间 - 开始总时间

        print("\n" + "=" * 80)
        print("📋 数据汇总信息:")
        print("=" * 80)
        print(f"总耗时: {总耗时:.1f} 秒")
        print(f"10min窗口数: {len(df)}")
        print(f"时间范围: {df['窗口开始时间'].min()} 到 {df['窗口开始时间'].max()}")
        print(f"特征数量: {len(df.columns)}")
        print(f"总事件数: {int(df['事件数量'].sum())}")
        print(f"总列车事件数: {int(df['列车事件数量'].sum())}")
        print(f"总非列车事件数: {int(df['非列车事件数量'].sum())}")
        print(f"标签均值（列车事件比例）: {df['标签均值'].mean():.3f}")
        print(f"有事件窗口比例: {(df['事件数量'] > 0).sum() / len(df) * 100:.1f}%")
        print(f"传感器信息: {基准信息['传感器信息']['sensor_id']}_{基准信息['传感器信息']['sensor_name']}")
        print(f"数据文件大小: {os.path.getsize(输出路径) / 1024 / 1024:.2f} MB")

        # 显示关键特征统计
        print("\n📊 关键特征统计:")
        for col in ['中心偏移距离', '分布半径', 'P1均值', 'P5偏斜度', '标签均值']:
            if col in df.columns:
                非空值 = df[col].dropna()
                if len(非空值) > 0:
                    print(f"  {col}:")
                    print(f"    有效值: {len(非空值)}/{len(df)} ({len(非空值) / len(df) * 100:.1f}%)")
                    print(f"    均值: {非空值.mean():.4f}")
                    print(f"    标准差: {非空值.std():.4f}")

        print("\n" + "=" * 80)
        print("✅ 10min级特征工程（29维特征并行版）完成!")
        print(f"📁 所有结果保存在: {时间戳目录}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        traceback.print_exc()