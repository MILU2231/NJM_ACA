# -*- coding: utf-8 -*-
"""
完整管道：特征提取 → 中心偏移距离 → 分布半径偏差
- 一键运行三个阶段
- 输出目录完全独立，避免混淆
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool ## [新增/修改点] 导入 Pool
from functools import partial ## [新增/修改点] 导入 partial


# 自定义 JSON 编码器
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super().default(obj)


# ==============================================
# 第一阶段：特征提取（从 CSV 到基础 JSON）
# ==============================================
def extract_features_from_csv(csv_path, output_base, max_hours=None):
    """处理单个10分钟CSV文件并保存基础特征JSON（不限制时间）"""
    try:
        # 读取文件
        df = pd.read_csv(csv_path, header=None, low_memory=False)

        if df.shape[0] == 0:
            # print(f"  {os.path.basename(csv_path)}: 文件为空，跳过") # [修改点] 并行时减少打印，避免混乱
            return None

        # 基本处理
        sensor_id = str(df.iloc[0, 0]).strip()
        df[1] = pd.to_datetime(df[1], errors='coerce')
        df = df.dropna(subset=[1]).copy()

        if len(df) == 0:
            # print(f"  {os.path.basename(csv_path)}: 无有效时间戳，跳过") # [修改点] 并行时减少打印，避免混乱
            return None

        df = df.sort_values(by=1).reset_index(drop=True)
        start_time = df[1].min()
        window_start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

        # 统计信息
        # time_range = df[1].max() - df[1].min() # [修改点] 并行时减少打印
        # print(f"  {os.path.basename(csv_path)}: {len(df)}行, 时长: {time_range}") # [修改点] 并行时减少打印

        # 数据清洗和特征提取（保持不变）
        data_cols = df.columns[2:]
        for col in data_cols:
            df[col] = df[col].astype(str).str.replace(r'\.1$', '', regex=True)
        for col in data_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        raw_signal = df.iloc[:, 2:].values.flatten().astype(np.float64)
        if np.any(np.isnan(raw_signal)):
            s = pd.Series(raw_signal)
            s = s.interpolate(method='linear', limit_direction='both').ffill().bfill()
            raw_signal = s.to_numpy()

        features_29 = extract_29_features(raw_signal, fs=50)

        # 保存结果
        result = {
            "sensor_id": sensor_id,
            "trigger_timestamp": window_start_str,
            "features_29": features_29
        }

        output_subdir = os.path.join(output_base, "feature_engineering_results")
        os.makedirs(output_subdir, exist_ok=True)

        time_str = start_time.strftime("%Y%m%d_%H%M%S")
        json_name = f"{time_str}_features.json"
        json_fullpath = os.path.join(output_subdir, json_name)

        with open(json_fullpath, 'w', encoding='utf-8') as f:
            json.dump(result, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

        # print(f"  已保存: {json_name}") # [修改点] 并行时减少打印
        return json_fullpath

    except Exception as e:
        print(f"  {os.path.basename(csv_path)}: 特征提取出错: {e}")
        return None

def extract_29_features(signal, fs=50):
    if len(signal) < 2:
        return np.full(29, np.nan)

    time_feats = compute_time_domain_features(signal)
    L = len(signal)
    nfft = 2 ** int(np.ceil(np.log2(L)))
    freqs = fs * np.arange(0, nfft // 2) / nfft
    yf = fft(signal, n=nfft)
    amps = 2 * np.abs(yf[:nfft // 2]) / L
    freq_feats = compute_frequency_domain_features(freqs, amps)
    return np.concatenate([time_feats, freq_feats])


def compute_time_domain_features(signal):
    N = len(signal)
    if N == 0:
        return np.zeros(16)
    mean_val = np.mean(signal)
    x = signal - mean_val
    rms = np.sqrt(np.sum(x ** 2) / N)
    root_amplitude = (np.sum(np.sqrt(np.abs(x))) / N) ** 2
    abs_mean = np.sum(np.abs(x)) / N
    skew_moment = np.sum(x ** 3) / N
    kurt_moment = np.sum(x ** 4) / N
    variance = np.var(x, ddof=1)
    max_val = np.max(x)
    min_val = np.min(x)
    peak_to_peak = max_val - min_val
    EPS = 1e-9
    shape_factor = rms / (abs_mean + EPS)
    crest_factor = max_val / (rms + EPS)
    impulse_factor = max_val / (abs_mean + EPS)
    clearance_factor = max_val / (root_amplitude + EPS)
    skew_factor = skew_moment / (np.sqrt(variance) ** 3 + EPS)
    kurt_factor = kurt_moment / (np.sqrt(variance) ** 4 + EPS)
    return np.array([mean_val, rms, root_amplitude, abs_mean, skew_moment, kurt_moment, variance,
                     max_val, min_val, peak_to_peak, shape_factor, crest_factor, impulse_factor,
                     clearance_factor, skew_factor, kurt_factor])


def compute_frequency_domain_features(frequencies, amplitudes):
    N = len(amplitudes)
    if N == 0:
        return np.zeros(13)
    mean_amp = np.mean(amplitudes)
    var_amp = np.sum((amplitudes - mean_amp) ** 2) / N
    skew_amp = np.sum((amplitudes - mean_amp) ** 3) / (N * (var_amp ** 1.5)) if var_amp > 0 else 0
    kurt_amp = np.sum((amplitudes - mean_amp) ** 4) / (N * (var_amp ** 2)) if var_amp > 0 else 0
    mean_freq = np.sum(frequencies * amplitudes) / np.sum(amplitudes) if np.sum(amplitudes) > 0 else 0
    std_freq = np.sqrt(np.sum((frequencies - mean_freq) ** 2 * amplitudes) / N) if N > 0 else 0
    rms_freq = np.sqrt(np.sum(frequencies ** 2 * amplitudes) / np.sum(amplitudes)) if np.sum(amplitudes) > 0 else 0
    rvf = np.sqrt(np.sum(frequencies ** 4 * amplitudes) / np.sum(frequencies ** 2 * amplitudes)) if np.sum(frequencies ** 2 * amplitudes) > 0 else 0
    ivf = np.sum(frequencies ** 2 * amplitudes) / np.sqrt(np.sum(amplitudes) * np.sum(frequencies ** 4 * amplitudes)) if np.sum(amplitudes) * np.sum(frequencies ** 4 * amplitudes) > 0 else 0
    freq_cv = std_freq / mean_freq if mean_freq > 0 else 0
    freq_skew = np.sum((frequencies - mean_freq) ** 3 * amplitudes) / (std_freq ** 3 * N) if std_freq > 0 else 0
    freq_kurt = np.sum((frequencies - mean_freq) ** 4 * amplitudes) / (std_freq ** 4 * N) if std_freq > 0 else 0
    abs_dev_weight = np.sum(np.sqrt(np.abs(frequencies - mean_freq)) * amplitudes) / (np.sqrt(std_freq) * N) if std_freq > 0 else 0
    return np.array([mean_amp, var_amp, skew_amp, kurt_amp, mean_freq, std_freq, rms_freq, rvf, ivf,
                     freq_cv, freq_skew, freq_kurt, abs_dev_weight])


# ==============================================
# 第二阶段：计算中心偏移距离
# ==============================================
def compute_center_shift(input_json_dir, scaler_path, benchmark_path, output_base):
    MAIN_OUT_DIR = os.path.join(output_base, "Center_Shift_Results")
    JSON_SUBDIR = os.path.join(MAIN_OUT_DIR, "shift_json_files")
    os.makedirs(MAIN_OUT_DIR, exist_ok=True)
    os.makedirs(JSON_SUBDIR, exist_ok=True)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    benchmark_center = np.load(benchmark_path)

    print(f"[中心偏移] 处理 JSON 目录: {input_json_dir}")
    print(f"  输出目录: {MAIN_OUT_DIR}")

    all_results = []
    json_files = sorted([os.path.join(input_json_dir, f) for f in os.listdir(input_json_dir) if f.endswith(".json")])

    for json_path in json_files:
        result = process_center_shift(json_path, scaler, benchmark_center, JSON_SUBDIR)
        if result:
            all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results).sort_values('trigger_timestamp')
        csv_path = os.path.join(MAIN_OUT_DIR, "10min_center_shift_series.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        generate_shift_plot(df, MAIN_OUT_DIR)

    print(f"[中心偏移] 处理完成，生成 {len(all_results)} 个结果")
    return all_results


def process_center_shift(json_path, scaler, benchmark_center, json_output_subdir):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sensor_id = data.get("sensor_id", "unknown")
        trigger_timestamp = data.get("trigger_timestamp", "")
        features_29 = np.array(data.get("features_29", []))

        if len(features_29) != 29:
            return None

        standardized = scaler.transform(features_29.reshape(1, -1))[0]
        center_shift = float(np.linalg.norm(standardized - benchmark_center))

        new_result = {
            "sensor_id": sensor_id,
            "trigger_timestamp": trigger_timestamp,
            "current_center_shift": center_shift,
            "features_29": features_29
        }

        time_str = trigger_timestamp.replace(":", "").replace(" ", "_").replace("-", "")
        json_name = f"{time_str}_with_shift.json"
        json_path_out = os.path.join(json_output_subdir, json_name)

        with open(json_path_out, 'w', encoding='utf-8') as f:
            json.dump(new_result, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

        return {
            "trigger_timestamp": trigger_timestamp,
            "sensor_id": sensor_id,
            "current_center_shift": center_shift
        }

    except Exception as e:
        print(f"中心偏移处理出错 {os.path.basename(json_path)}: {e}")
        return None


def generate_shift_plot(df, output_dir):
    plt.figure(figsize=(14, 7))
    df['trigger_timestamp'] = pd.to_datetime(df['trigger_timestamp'])
    df = df.sort_values('trigger_timestamp')
    plt.plot(df['trigger_timestamp'], df['current_center_shift'], 'r-', lw=2, label='Center Shift')
    plt.title('10min Center Offset Distance Time Series', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Current Center Shift')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "center_shift_time_series.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================
# 第三阶段：计算分布半径偏差
# ==============================================
def compute_cluster_radius(input_json_dir, output_base):
    MAIN_OUT_DIR = os.path.join(output_base, "Distribution_Radius_Results")
    JSON_SUBDIR = os.path.join(MAIN_OUT_DIR, "radius_json_files")
    os.makedirs(MAIN_OUT_DIR, exist_ok=True)
    os.makedirs(JSON_SUBDIR, exist_ok=True)

    CENTER_PATH = os.path.join(MAIN_OUT_DIR, "fixed_center_point.npy")
    SCALER_PATH = os.path.join(MAIN_OUT_DIR, "fixed_scaler.pkl")
    BENCHMARK_NORMAL_RADIUS_PATH = os.path.join(MAIN_OUT_DIR, "benchmark_normal_radius.npy") ## [修改点] 添加基准正常半径的保存路径

    json_files = sorted([os.path.join(input_json_dir, f) for f in os.listdir(input_json_dir) if f.endswith(".json")])

    if len(json_files) < 10:
        print("[分布半径] 文件不足10个，无法可靠检测跳变")
        return

    center = None
    scaler = None
    benchmark_normal_radius = None ## [修改点] 初始化 benchmark_normal_radius

    ## [修改点] 检查所有三个基准文件是否存在
    if not os.path.exists(CENTER_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(BENCHMARK_NORMAL_RADIUS_PATH):
        print("[分布半径] 检测跳变并计算固定圆心、标准化器和基准正常半径...") ## [修改点] 更新打印信息
        ## [修改点] compute_center_from_jump 现在返回四个值
        center, scaler, jump_idx, benchmark_normal_radius = compute_center_from_jump(json_files, pre=4, post=5)

        if center is None or scaler is None or benchmark_normal_radius is None: ## [修改点] 增加对 None 值的检查
            print("[分布半径] 无法计算固定圆心、标准化器或基准正常半径，跳过后续处理。")
            return

        np.save(CENTER_PATH, center)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        np.save(BENCHMARK_NORMAL_RADIUS_PATH, benchmark_normal_radius) ## [修改点] 保存基准正常半径
    else:
        center = np.load(CENTER_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        benchmark_normal_radius = np.load(BENCHMARK_NORMAL_RADIUS_PATH) ## [修改点] 加载基准正常半径

    print("[分布半径] 使用固定圆心、标准化器和基准正常半径（基于跳变前后10个点）") ## [修改点] 更新打印信息

    results = []
    for path in json_files:
        ## [修改点] 传递 benchmark_normal_radius 给 process_cluster_radius
        res = process_cluster_radius(path, scaler, center, JSON_SUBDIR, benchmark_normal_radius)
        if res:
            results.append(res)

    if results:
        df = pd.DataFrame(results).sort_values("trigger_timestamp")
        # csv文件和图表会使用新的current_cluster_radius (相对偏差)
        df.to_csv(os.path.join(MAIN_OUT_DIR, "10min_cluster_radius_series.csv"), index=False, encoding='utf-8-sig')
        plot_radius_series(df, MAIN_OUT_DIR)

    print(f"[分布半径] 处理完成，生成 {len(results)} 个结果")


def compute_center_from_jump(json_files, pre=4, post=5):
    if len(json_files) < 2:
        raise ValueError("至少需要2个JSON文件")

    features_seq = []
    for fpath in json_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        feat = np.array(data.get("features_29", []))
        if len(feat) == 29:
            features_seq.append(feat)

    features_seq = np.array(features_seq)
    if len(features_seq) < 2: ## [修改点] 增加检查，防止特征序列不足
        print("[跳变检测] 特征序列不足2个有效数据点，无法检测跳变。")
        return None, None, None, None

    diffs = np.linalg.norm(features_seq[1:] - features_seq[:-1], axis=1)
    jump_idx = np.argmax(diffs) + 1

    print(f"[跳变检测] 最大跳变在第 {jump_idx} 个块 (距离: {diffs[jump_idx-1]:.4f})")

    start = max(0, jump_idx - pre)
    end = min(len(features_seq), jump_idx + post + 1)
    selected = features_seq[start:end]

    if len(selected) == 0: ## [修改点] 增加检查，防止选定窗口为空
        print("[跳变检测] 选定窗口内没有有效数据，无法计算中心。")
        return None, None, None, None

    scaler = StandardScaler()
    std = scaler.fit_transform(selected)
    center = np.mean(std, axis=0)

    ## [修改点] 计算基准正常半径
    distances_from_center = np.linalg.norm(std - center, axis=1)
    benchmark_normal_radius = np.mean(distances_from_center)
    # 防止分母过小导致相对偏差过大，设置一个最小值
    if benchmark_normal_radius < 1e-9:
        benchmark_normal_radius = 1e-9

    ## [修改点] 返回四个值
    return center, scaler, jump_idx, benchmark_normal_radius


## [修改点] process_cluster_radius 接受 benchmark_normal_radius 参数
def process_cluster_radius(json_path, scaler, center_point, json_output_subdir, benchmark_normal_radius):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sensor_id = data.get("sensor_id", "unknown")
        trigger_timestamp = data.get("trigger_timestamp", "")
        features_29 = np.array(data.get("features_29", []))

        if len(features_29) != 29:
            print(f"[分布半径] {os.path.basename(json_path)}: 特征向量长度不为29，跳过。") ## [修改点] 增加错误信息
            return None

        std = scaler.transform(features_29.reshape(1, -1))[0]
        # 计算当前点到基准中心的绝对距离
        absolute_distance = float(np.linalg.norm(std - center_point)) ## [修改点] 变量名更清晰

        # 计算相对偏差
        ## [修改点] 增加对 benchmark_normal_radius 为0的保护
        if benchmark_normal_radius == 0:
             relative_deviation = 0.0 # 或者其他适当的值，表示无法计算相对偏差
        else:
            relative_deviation = absolute_distance / benchmark_normal_radius

        new_result = {
            "sensor_id": sensor_id,
            "trigger_timestamp": trigger_timestamp,
            "current_cluster_radius": relative_deviation, # 现在存储的是相对偏差 ## [修改点]
            "absolute_distance_from_center": absolute_distance, # 可选，保留绝对距离以供参考 ## [修改点]
            "features_29": features_29
        }

        time_str = trigger_timestamp.replace(":", "").replace(" ", "_").replace("-", "")
        json_name = f"{time_str}_with_radius.json"
        json_path_out = os.path.join(json_output_subdir, json_name)

        with open(json_path_out, 'w', encoding='utf-8') as f:
            json.dump(new_result, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

        return {
            "trigger_timestamp": trigger_timestamp,
            "sensor_id": sensor_id,
            "current_cluster_radius": relative_deviation # 返回相对偏差 ## [修改点]
        }

    except Exception as e:
        print(f"[分布半径] 处理出错 {os.path.basename(json_path)}: {e}")
        return None


def plot_radius_series(df, output_dir):
    plt.figure(figsize=(14, 7))
    df['trigger_timestamp'] = pd.to_datetime(df['trigger_timestamp'])
    df = df.sort_values('trigger_timestamp')
    plt.plot(df['trigger_timestamp'], df['current_cluster_radius'], 'b-', lw=2, label='Cluster Relative Deviation') ## [修改点] 更新图例
    plt.title('10min Cluster Relative Deviation Time Series', fontsize=16) ## [修改点] 更新标题
    plt.xlabel('Time')
    plt.ylabel('Current Cluster Relative Deviation') ## [修改点] 更新Y轴标签
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_relative_deviation_time_series.png"), dpi=300, bbox_inches='tight') ## [修改点] 更新文件名
    plt.close()


# ==============================================
# 第四阶段：整合最终 JSON
# ==============================================
def integrate_final_json(output_base):
    FINAL_JSON_DIR = os.path.join(output_base, "final_integrated_json")
    os.makedirs(FINAL_JSON_DIR, exist_ok=True)

    # 三个来源目录
    shift_dir   = os.path.join(output_base, "Center_Shift_Results",     "shift_json_files")
    radius_dir  = os.path.join(output_base, "Distribution_Radius_Results", "radius_json_files")
    feature_dir = os.path.join(output_base, "feature_engineering_results")

    # 以 trigger_timestamp 为键进行合并（假设时间戳唯一）
    merged_data = {}

    # 1. 先读所有基础特征
    for fname in os.listdir(feature_dir):
        if not fname.endswith(".json"): continue
        path = os.path.join(feature_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        ts = d["trigger_timestamp"]
        merged_data[ts] = {
            "sensor_id": d["sensor_id"],
            "trigger_timestamp": ts,
            "features_29": d["features_29"]
        }

    # 2. 叠加 center_shift
    for fname in os.listdir(shift_dir):
        if not fname.endswith(".json"): continue
        path = os.path.join(shift_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        ts = d["trigger_timestamp"]
        if ts in merged_data:
            merged_data[ts]["current_center_shift"] = d["current_center_shift"]

    # 3. 叠加 cluster_radius (现在是相对偏差)
    for fname in os.listdir(radius_dir):
        if not fname.endswith(".json"): continue
        path = os.path.join(radius_dir, fname)
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        ts = d["trigger_timestamp"]
        if ts in merged_data:
            merged_data[ts]["current_cluster_radius"] = d["current_cluster_radius"] # 现在是相对偏差

    # 4. 补充 rms / variance / frequency_center 并写出最终文件
    for ts, item in merged_data.items():
        if "features_29" not in item or len(item["features_29"]) != 29:
            continue

        feat = np.array(item["features_29"])

        # 提取需要的特征值
        rms_value = float(feat[1])     # rms（均方根）
        variance_value = float(feat[6])     # variance（方差）
        mean_freq_value = float(feat[20])    # mean_freq（频率中心）

        # 按照你要求的顺序构建JSON：RMS、均方差、频率中心在前，特征向量最后
        ordered_result = {
            "sensor_id":              item["sensor_id"],
            "trigger_timestamp":      item["trigger_timestamp"],
            "current_center_shift":   item.get("current_center_shift", None),
            "current_cluster_radius": item.get("current_cluster_radius", None), # 现在是相对偏差
            "current_rms":            rms_value,  # RMS 均方根
            "current_variance":       variance_value,  # 均方差
            "current_frequency_center": mean_freq_value,  # 频率中心
            "features_29":            item["features_29"]   # ← 29个特征放在最后
        }

        # 文件名示例：20250101_120000_final.json
        time_clean = item["trigger_timestamp"].replace(":", "").replace(" ", "_").replace("-", "")
        out_name = f"{time_clean}_final.json"
        out_path = os.path.join(FINAL_JSON_DIR, out_name)

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(ordered_result, f, cls=NumpyEncoder, ensure_ascii=False, indent=2)

        print(f"已生成最终整合 JSON: {out_path}")

    print(f"\n最终整合 JSON 全部保存至：{FINAL_JSON_DIR}")
    print(f"共生成 {len(merged_data)} 个文件")

# ==============================================
# 主程序 - 一键运行全流程
# ==============================================
if __name__ == "__main__":
    import os

    # 当前脚本目录：ACA_docking_project/scripts
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # 项目根目录：ACA_docking_project
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

    # ===============================
    # 相对路径统一配置
    # ===============================
    INPUT_CSV_DIR = os.path.join(PROJECT_ROOT, "split_10min_files_Month")

    OUTPUT_BASE = os.path.join(PROJECT_ROOT, "DData_Month_results")
    FEATURE_JSON_DIR = os.path.join(OUTPUT_BASE, "feature_engineering_results")

    SCALER_PATH = os.path.join(
        PROJECT_ROOT,
        "NEW_lstm_benchmark_files",
        "10_minute_grade_standardizer.pkl"
    )

    BENCHMARK_PATH = os.path.join(
        PROJECT_ROOT,
        "NEW_lstm_benchmark_files",
        "BenchmarkCenter.npy"
    )

    # ===============================
    # 基本检查（推荐保留）
    # ===============================
    for p, name in [
        (INPUT_CSV_DIR, "INPUT_CSV_DIR"),
        (SCALER_PATH, "SCALER_PATH"),
        (BENCHMARK_PATH, "BENCHMARK_PATH"),
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{name} 不存在: {p}")

    os.makedirs(OUTPUT_BASE, exist_ok=True)


    # 获取所有CSV文件
    csv_files = [os.path.join(INPUT_CSV_DIR, f) for f in os.listdir(INPUT_CSV_DIR) if f.endswith(".csv")]
    csv_files.sort()  # 按文件名排序

    print("=" * 80)
    print(f"开始处理 {len(csv_files)} 个CSV文件")
    print(f"输入目录: {INPUT_CSV_DIR}")
    print(f"输出目录: {OUTPUT_BASE}")
    print("=" * 80)

    # 第一阶段：特征提取
    print("阶段1：特征提取（处理所有数据）")
    print("=" * 80)

    successful_count = 0
    failed_count = 0

    ## [新增/修改点] 使用 multiprocessing.Pool 进行并行处理
    # 准备函数，固定 output_base 和 max_hours 参数
    partial_extract_features = partial(extract_features_from_csv, output_base=OUTPUT_BASE, max_hours=None)

    # 使用10个CPU核心创建进程池
    with Pool(processes=10) as pool:
        # imap_unordered 会尽快返回结果，不需要等待所有任务完成
        # tqdm 用于显示进度条
        for result in tqdm(pool.imap_unordered(partial_extract_features, csv_files), total=len(csv_files), desc="特征提取进度"):
            if result:
                successful_count += 1
            else:
                failed_count += 1

    print(f"\n特征提取完成: 成功 {successful_count}, 失败 {failed_count}") ## [修改点] 增加换行符

    # 检查生成了多少JSON文件
    json_files = [f for f in os.listdir(FEATURE_JSON_DIR) if f.endswith(".json")] if os.path.exists(
        FEATURE_JSON_DIR) else []
    print(f"生成 {len(json_files)} 个JSON文件")

    if len(json_files) == 0:
        print("警告: 没有生成JSON文件，停止后续处理")
    else:
        # 第二阶段：中心偏移距离
        print("\n" + "=" * 80)
        print("阶段2：中心偏移距离计算")
        print("=" * 80)
        compute_center_shift(FEATURE_JSON_DIR, SCALER_PATH, BENCHMARK_PATH, OUTPUT_BASE)

        # 第三阶段：分布半径偏差
        print("\n" + "=" * 80)
        print("阶段3：分布半径偏差计算 (现在计算的是相对偏差)") ## [修改点] 更新阶段说明
        print("=" * 80)
        compute_cluster_radius(FEATURE_JSON_DIR, OUTPUT_BASE)

        # 第四阶段：整合最终JSON
        print("\n" + "=" * 80)
        print("阶段4：整合最终特征JSON")
        print("=" * 80)
        integrate_final_json(OUTPUT_BASE)

    print("\n" + "=" * 80)
    print("全流程完成！")
    print("=" * 80)
