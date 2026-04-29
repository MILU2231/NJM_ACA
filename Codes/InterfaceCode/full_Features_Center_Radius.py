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
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler


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
def extract_features_from_csv(csv_path, output_base, max_hours=12):
    """处理单个 10 分钟 CSV 文件并保存基础特征 JSON"""
    print(f"[特征提取] 处理文件: {csv_path}")

    try:
        df = pd.read_csv(csv_path, header=None, low_memory=False)
        if df.shape[0] == 0:
            print("  文件为空，跳过")
            return None

        sensor_id = str(df.iloc[0, 0]).strip()
        df[1] = pd.to_datetime(df[1], errors='coerce')
        df = df.dropna(subset=[1]).copy()

        if len(df) == 0:
            print("  无有效时间戳，跳过")
            return None

        df = df.sort_values(by=1).reset_index(drop=True)
        start_time = df[1].min()
        window_start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

        end_limit = start_time + timedelta(hours=max_hours)
        df = df[df[1] <= end_limit]

        if len(df) == 0:
            print("  时间范围内无数据")
            return None

        # 清洗 .1 后缀
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

        print(f"  已保存基础特征: {json_fullpath}")
        return json_fullpath

    except Exception as e:
        print(f"  特征提取出错: {e}")
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

    json_files = sorted([os.path.join(input_json_dir, f) for f in os.listdir(input_json_dir) if f.endswith(".json")])

    if len(json_files) < 10:
        print("[分布半径] 文件不足10个，无法可靠检测跳变")
        return

    if not os.path.exists(CENTER_PATH) or not os.path.exists(SCALER_PATH):
        print("[分布半径] 检测跳变并计算固定圆心...")
        center, scaler, jump_idx = compute_center_from_jump(json_files, pre=4, post=5)
        np.save(CENTER_PATH, center)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
    else:
        center = np.load(CENTER_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)

    print("[分布半径] 使用固定圆心（基于跳变前后10个点）")

    results = []
    for path in json_files:
        res = process_cluster_radius(path, scaler, center, JSON_SUBDIR)
        if res:
            results.append(res)

    if results:
        df = pd.DataFrame(results).sort_values("trigger_timestamp")
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
    diffs = np.linalg.norm(features_seq[1:] - features_seq[:-1], axis=1)
    jump_idx = np.argmax(diffs) + 1

    print(f"[跳变检测] 最大跳变在第 {jump_idx} 个块 (距离: {diffs[jump_idx-1]:.4f})")

    start = max(0, jump_idx - pre)
    end = min(len(features_seq), jump_idx + post + 1)
    selected = features_seq[start:end]

    scaler = StandardScaler()
    std = scaler.fit_transform(selected)
    center = np.mean(std, axis=0)

    return center, scaler, jump_idx


def process_cluster_radius(json_path, scaler, center_point, json_output_subdir):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sensor_id = data.get("sensor_id", "unknown")
        trigger_timestamp = data.get("trigger_timestamp", "")
        features_29 = np.array(data.get("features_29", []))

        if len(features_29) != 29:
            return None

        std = scaler.transform(features_29.reshape(1, -1))[0]
        radius = float(np.linalg.norm(std - center_point))

        new_result = {
            "sensor_id": sensor_id,
            "trigger_timestamp": trigger_timestamp,
            "current_cluster_radius": radius,
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
            "current_cluster_radius": radius
        }

    except Exception as e:
        print(f"[分布半径] 处理出错 {os.path.basename(json_path)}: {e}")
        return None


def plot_radius_series(df, output_dir):
    plt.figure(figsize=(14, 7))
    df['trigger_timestamp'] = pd.to_datetime(df['trigger_timestamp'])
    df = df.sort_values('trigger_timestamp')
    plt.plot(df['trigger_timestamp'], df['current_cluster_radius'], 'b-', lw=2, label='Cluster Radius')
    plt.title('10min Cluster Radius Time Series', fontsize=16)
    plt.xlabel('Time')
    plt.ylabel('Current Cluster Radius')
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cluster_radius_time_series.png"), dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================
# 主程序 - 一键运行全流程
# ==============================================
if __name__ == "__main__":
    # 路径配置（根据你的实际路径调整）
    INPUT_CSV_DIR = r"/ACA_docking_project/Data_results/split_10min_files"
    OUTPUT_BASE = r"O:\Teamwork\南纪门项目组\NJM_Project_CONDA\ACA_docking_project\Dataaaaaa_results"
    FEATURE_JSON_DIR = os.path.join(OUTPUT_BASE, "feature_engineering_results")

    SCALER_PATH = r"/ACA_docking_project/Models_and_benchmarks/DistributionRadius_CenterOffsetDistance/10_minute_grade_standardizer.pkl"
    BENCHMARK_PATH = r"/ACA_docking_project/Models_and_benchmarks/DistributionRadius_CenterOffsetDistance/BenchmarkCenter.npy"

    # 第一阶段：特征提取（如果需要运行）
    # 注释掉如果已经跑过
    print("=" * 80)
    print("阶段1：特征提取（从 CSV 到基础 JSON）")
    print("=" * 80)
    csv_files = [os.path.join(INPUT_CSV_DIR, f) for f in os.listdir(INPUT_CSV_DIR) if f.endswith(".csv")]
    for csv in csv_files:
        extract_features_from_csv(csv, OUTPUT_BASE)

    # 第二阶段：中心偏移距离
    print("\n" + "=" * 80)
    print("阶段2：中心偏移距离计算")
    print("=" * 80)
    compute_center_shift(FEATURE_JSON_DIR, SCALER_PATH, BENCHMARK_PATH, OUTPUT_BASE)

    # 第三阶段：分布半径偏差
    print("\n" + "=" * 80)
    print("阶段3：分布半径偏差计算")
    print("=" * 80)
    compute_cluster_radius(FEATURE_JSON_DIR, OUTPUT_BASE)

    print("\n" + "=" * 80)
    print("全流程完成！")
    print("输出目录：")
    print(f"  基础特征 JSON: {FEATURE_JSON_DIR}")
    print(f"  中心偏移结果: {os.path.join(OUTPUT_BASE, 'Center_Shift_Results')}")
    print(f"  分布半径结果: {os.path.join(OUTPUT_BASE, 'Distribution_Radius_Results')}")
    print("=" * 80)