import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import glob
from tqdm import tqdm
import re
from datetime import datetime, timedelta
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# 并行处理参数
PARALLEL_WORKERS = 14  # 使用8个核心进行并行处理

# 传感器名称映射表
SENSOR_MAPPING = {
    "NJM-ACA-C01-01": "上游侧SSC16索索力",
    "NJM-ACA-C01-02": "下游侧SSC16索索力",
    "NJM-ACA-C02-01": "上游侧SSC8索索力",
    "NJM-ACA-C02-02": "下游侧SSC8索索力",
    "NJM-ACA-C03-01": "上游侧SSC1索索力",
    "NJM-ACA-C03-02": "下游侧SSC1索索力",
    "NJM-ACA-C04-02": "下游侧SMC1索索力",
    "NJM-ACA-C08-01": "上游侧SMC8索索力",
    "NJM-ACA-C08-02": "下游侧SMC8索索力",
    "NJM-ACA-C12-02": "下游侧SMC16索索力",
    "NJM-ACA-C13-02": "下游侧NMC27索索力",
    "NJM-ACA-C17-01": "上游侧NMC13索索力",
    "NJM-ACA-C17-02": "下游侧NMC13索索力",
    "NJM-ACA-C21-02": "下游侧NMC1索索力",
    "NJM-ACA-C22-01": "上游侧NSC1索索力",
    "NJM-ACA-C22-02": "下游侧NSC1索索力",
    "NJM-ACA-C23-01": "上游侧NSC13索索力",
    "NJM-ACA-C23-02": "下游侧NSC13索索力",
    "NJM-ACA-C24-01": "上游侧NSC27索索力",
    "NJM-ACA-C24-02": "下游侧NSC27索索力"
}


def generate_date_range(start_date_str, end_date_str):
    """
    生成日期范围列表

    Parameters:
    start_date_str: 开始日期字符串 (格式: YYYY-MM-DD)
    end_date_str: 结束日期字符串 (格式: YYYY-MM-DD)

    Returns:
    date_list: 日期字符串列表
    """

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

    date_list = []
    current_date = start_date

    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)

    return date_list


def get_year_month_folders(base_path):
    """
    获取基础路径下的年月文件夹列表

    Parameters:
    base_path: 基础目录路径

    Returns:
    year_month_folders: 年月文件夹列表，格式如 ['2023_01', '2023_02', ...]
    """
    if not os.path.exists(base_path):
        return []

    # 获取所有文件夹，过滤出符合年月格式的文件夹
    all_items = os.listdir(base_path)
    year_month_folders = []

    for item in all_items:
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # 检查是否符合年月格式：YYYY_MM
            if re.match(r'^\d{4}_\d{2}$', item):
                year_month_folders.append(item)

    # 按年月排序
    year_month_folders.sort()

    return year_month_folders


def get_date_folder_path(base_path, date_str):
    """
    根据日期字符串获取对应的文件夹路径

    Parameters:
    base_path: 基础目录路径
    date_str: 日期字符串 (格式: YYYY-MM-DD)

    Returns:
    date_folder_path: 日期文件夹完整路径，如果不存在则返回None
    """
    try:
        # 将日期字符串转换为年月格式
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year_month_str = date_obj.strftime("%Y_%m")

        # 构建完整路径
        year_month_path = os.path.join(base_path, year_month_str)
        date_folder_path = os.path.join(year_month_path, date_str)

        if os.path.exists(date_folder_path):
            return date_folder_path
        else:
            return None
    except Exception as e:
        return None


def get_sorted_csv_files(folder_path):
    """
    获取并按照数字顺序排序CSV文件

    Parameters:
    folder_path: 文件夹路径

    Returns:
    sorted_files: 按数字顺序排序的文件列表
    """

    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        return []

    # 提取文件名中的数字进行排序
    file_info = []
    for file_path in csv_files:
        file_name = os.path.basename(file_path)

        # 使用正则表达式提取数字部分
        match = re.search(r'_(\d+)\.csv$', file_name)
        if match:
            file_num = int(match.group(1))
            file_info.append((file_num, file_name, file_path))
        else:
            # 如果没有数字，按文件名排序
            file_info.append((999999, file_name, file_path))

    # 按数字排序
    file_info.sort(key=lambda x: x[0])

    # 返回排序后的文件路径
    sorted_files = [info[2] for info in file_info]

    return sorted_files


def get_sensor_id_from_csv(file_path):
    """
    从CSV文件中读取传感器ID（第一列数据）

    Parameters:
    file_path: CSV文件路径

    Returns:
    sensor_id: 传感器ID字符串，如果读取失败返回None
    """
    try:
        # 读取CSV文件的第一行第一列
        df_sample = pd.read_csv(file_path, header=None, nrows=1)
        if df_sample.shape[1] > 0:
            sensor_id = str(df_sample.iloc[0, 0]).strip()
            return sensor_id
    except Exception as e:
        print(f"读取传感器ID失败 {file_path}: {e}")

    return None


def load_and_concatenate_data(date_folder_path):
    """
    读取指定日期文件夹中的所有CSV文件，并拼接数据列

    Parameters:
    date_folder_path: 日期文件夹路径

    Returns:
    concatenated_data: 拼接后的numpy数组
    file_count: 处理的文件数量
    total_points: 总数据点数
    file_info_list: 文件信息列表
    sensor_id: 传感器ID
    """

    # 获取按数字顺序排序的CSV文件
    csv_files = get_sorted_csv_files(date_folder_path)

    if not csv_files:
        return None, 0, 0, [], None

    # 用于存储所有数据
    all_data_segments = []
    file_info_list = []

    # 传感器ID（从第一个文件中读取）
    sensor_id = None

    # 读取并处理每个文件
    for i, file_path in enumerate(csv_files):
        try:
            # 读取CSV文件，没有表头
            df = pd.read_csv(file_path, header=None)

            # 检查数据列数
            if df.shape[1] < 3:
                continue

            # 如果是第一个文件，获取传感器ID
            if i == 0 and sensor_id is None:
                sensor_id = str(df.iloc[0, 0]).strip()

            # 获取数据列（跳过第一列设备号和第二列时间戳）
            data_columns = df.iloc[:, 2:].values  # 获取所有数据列

            # 将数据展平为一维数组（按行展开）
            flattened_data = data_columns.flatten()

            all_data_segments.append(flattened_data)

            # 记录文件信息
            file_info = {
                'file_num': i + 1,
                'file_name': os.path.basename(file_path),
                'rows': df.shape[0],
                'columns': df.shape[1],
                'data_points': flattened_data.shape[0],
                'start_value': float(flattened_data[0]) if len(flattened_data) > 0 else None,
                'end_value': float(flattened_data[-1]) if len(flattened_data) > 0 else None,
            }
            file_info_list.append(file_info)

        except Exception as e:
            continue

    if not all_data_segments:
        return None, 0, 0, [], None

    # 拼接所有数据
    concatenated_data = np.concatenate(all_data_segments)

    total_points = len(concatenated_data)
    file_count = len(all_data_segments)

    return concatenated_data, file_count, total_points, file_info_list, sensor_id


def create_simple_visualization(data, date_str, sensor_name, output_folder):
    """
    创建简化的数据可视化图形

    Parameters:
    data: 拼接后的数据
    date_str: 日期字符串
    sensor_name: 传感器名称
    output_folder: 输出文件夹路径

    Returns:
    plot_path: 图片文件路径
    """

    # 创建输出文件路径
    plot_path = os.path.join(output_folder, f"{date_str}_concatenated.png")

    # 设置中文字体
    try:
        font_path = "C:/Windows/Fonts/msyh.ttc"
        if os.path.exists(font_path):
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    total_points = len(data)

    # 2. 创建数据曲线图
    plt.figure(figsize=(14, 7))

    # 如果数据量太大，进行采样显示
    if total_points > 100000:
        step = max(1, total_points // 50000)
        display_data = data[::step]
        display_indices = np.arange(0, total_points, step)
    else:
        display_data = data
        display_indices = np.arange(total_points)

    # 绘制数据曲线
    plt.plot(display_indices, display_data, linewidth=0.5, alpha=0.7, color='blue')

    # 设置图形属性
    title = f'ACA数据拼接曲线 - {date_str}\n传感器: {sensor_name}\n数据点: {total_points:,}'
    plt.title(title, fontsize=14)
    plt.xlabel('数据点索引', fontsize=12)
    plt.ylabel('数据值', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def save_json_data(data, date_str, sensor_id, sensor_name, output_folder, file_info_list):
    """
    保存数据为JSON文件

    Parameters:
    data: 拼接后的数据
    date_str: 日期字符串
    sensor_id: 传感器ID
    sensor_name: 传感器名称
    output_folder: 输出文件夹路径
    file_info_list: 文件信息列表
    """

    json_path = os.path.join(output_folder, f"{date_str}_concatenated.json")

    # 准备JSON数据结构
    json_data = {
        "sensor_id": sensor_id,
        "sensor_name": sensor_name,
        "date": date_str,
        "total_points": len(data),
        "data_min": float(data.min()),
        "data_max": float(data.max()),
        "data_mean": float(data.mean()),
        "data_std": float(data.std()),
        "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_count": len(file_info_list),
        "files": file_info_list,
        "data": data.tolist()  # 将numpy数组转换为列表
    }

    # 保存为JSON文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    return json_path


def save_summary_report(date_str, output_folder, file_info_list, data, sensor_id, sensor_name):
    """
    保存处理摘要报告

    Parameters:
    date_str: 日期字符串
    output_folder: 输出文件夹
    file_info_list: 文件信息列表
    data: 拼接后的数据
    sensor_id: 传感器ID
    sensor_name: 传感器名称
    """

    report_path = os.path.join(output_folder, f"{date_str}_report.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"ACA数据拼接处理报告 - {date_str}\n")
        f.write(f"传感器ID: {sensor_id}\n")
        f.write(f"传感器名称: {sensor_name}\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n")
        f.write("=" * 60 + "\n\n")

        # 1. 处理概览
        f.write("1. 处理概览\n")
        f.write("-" * 40 + "\n")
        f.write(f"处理日期: {date_str}\n")
        f.write(f"传感器ID: {sensor_id}\n")
        f.write(f"传感器名称: {sensor_name}\n")
        f.write(f"总文件数: {len(file_info_list)}\n")
        f.write(f"总数据点数: {len(data):,}\n")
        f.write(f"数据形状: {data.shape}\n")
        f.write(f"数据范围: [{data.min():.6f}, {data.max():.6f}]\n")
        f.write(f"数据均值: {data.mean():.6f}\n")
        f.write(f"数据标准差: {data.std():.6f}\n\n")

        # 2. 前10个文件信息
        f.write("2. 文件列表 (前10个)\n")
        f.write("-" * 40 + "\n")
        f.write("序号 | 文件名 | 行数 | 列数 | 数据点\n")
        f.write("-" * 50 + "\n")

        for info in file_info_list[:10]:
            f.write(f"{info['file_num']:3d} | {info['file_name']:20s} | "
                    f"{info['rows']:4d} | {info['columns']:3d} | "
                    f"{info['data_points']:7d}\n")

        if len(file_info_list) > 10:
            f.write(f"... 还有 {len(file_info_list) - 10} 个文件\n")

        f.write("\n")

        # 3. 输出文件列表
        f.write("3. 生成的文件\n")
        f.write("-" * 40 + "\n")
        f.write(f"数据文件 (.json): {date_str}_concatenated.json\n")
        f.write(f"曲线图: {date_str}_concatenated.png\n")
        f.write(f"处理报告: {date_str}_report.txt\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("处理完成!\n")
        f.write("=" * 60 + "\n")


def get_output_month_folder_path(output_base_dir, date_str):
    """
    根据日期获取输出月份文件夹路径

    Parameters:
    output_base_dir: 输出基础目录
    date_str: 日期字符串 (格式: YYYY-MM-DD)

    Returns:
    month_folder_path: 输出月份文件夹路径
    date_folder_path: 输出日期文件夹路径
    """
    try:
        # 将日期字符串转换为年月格式
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year_month_str = date_obj.strftime("%Y_%m")

        # 构建月份文件夹路径
        month_folder_path = os.path.join(output_base_dir, year_month_str)

        # 构建日期文件夹路径（在月份文件夹内）
        date_folder_path = os.path.join(month_folder_path, date_str)

        return month_folder_path, date_folder_path
    except Exception as e:
        # 如果出错，使用扁平结构
        date_folder_path = os.path.join(output_base_dir, date_str)
        return output_base_dir, date_folder_path


def process_single_date(date_str, input_base_dir, output_base_dir):
    """
    处理单个日期的数据

    Parameters:
    date_str: 日期字符串
    input_base_dir: 输入基础目录
    output_base_dir: 输出基础目录

    Returns:
    result_dict: 处理结果字典
    """

    # 获取输入日期文件夹路径（考虑年月目录层次）
    date_folder_path = get_date_folder_path(input_base_dir, date_str)

    # 获取输出文件夹路径（按照月份组织）
    month_folder_path, output_folder = get_output_month_folder_path(output_base_dir, date_str)

    # 检查输入文件夹是否存在
    if date_folder_path is None or not os.path.exists(date_folder_path):
        return {"date": date_str, "status": "skipped", "reason": "文件夹不存在"}

    # 创建输出月份文件夹和日期文件夹
    os.makedirs(output_folder, exist_ok=True)

    try:
        # 加载并拼接数据
        data, file_count, total_points, file_info_list, sensor_id = load_and_concatenate_data(date_folder_path)

        if data is None or len(data) == 0 or sensor_id is None:
            return {"date": date_str, "status": "skipped", "reason": "无数据或数据为空"}

        # 获取传感器名称
        sensor_name = SENSOR_MAPPING.get(sensor_id, "未知传感器")

        # 创建可视化图形
        plot_path = create_simple_visualization(data, date_str, sensor_name, output_folder)

        # 保存JSON数据文件
        json_path = save_json_data(data, date_str, sensor_id, sensor_name, output_folder, file_info_list)

        # 保存处理报告
        save_summary_report(date_str, output_folder, file_info_list, data, sensor_id, sensor_name)

        # 准备结果信息
        result_dict = {
            'date': date_str,
            'status': 'success',
            'sensor_id': sensor_id,
            'sensor_name': sensor_name,
            'file_count': file_count,
            'total_points': total_points,
            'data_min': float(data.min()),
            'data_max': float(data.max()),
            'data_mean': float(data.mean()),
            'data_std': float(data.std()),
            'json_path': json_path,
            'plot_path': plot_path,
            'output_folder': output_folder,
            'output_month_folder': month_folder_path,
            'source_folder': date_folder_path  # 记录源文件夹路径
        }

        return result_dict

    except Exception as e:
        return {"date": date_str, "status": "error", "error": str(e)}


def process_single_date_wrapper(args):
    """
    包装函数，用于并行处理

    Parameters:
    args: 包含参数的元组 (date_str, input_base_dir, output_base_dir)

    Returns:
    result_dict: 处理结果字典
    """
    date_str, input_base_dir, output_base_dir = args
    return process_single_date(date_str, input_base_dir, output_base_dir)


def find_available_dates(input_base_dir, start_date_str, end_date_str):
    """
    查找在输入目录中实际存在的日期

    Parameters:
    input_base_dir: 输入基础目录
    start_date_str: 开始日期
    end_date_str: 结束日期

    Returns:
    available_dates: 实际存在的日期列表
    """
    print("正在扫描输入目录结构...")

    # 获取所有年月文件夹
    year_month_folders = get_year_month_folders(input_base_dir)

    if not year_month_folders:
        print("警告: 未找到任何年月文件夹")
        return []

    print(f"找到年月文件夹: {', '.join(year_month_folders)}")

    # 生成完整日期范围
    all_dates = generate_date_range(start_date_str, end_date_str)
    available_dates = []

    # 使用tqdm显示进度
    with tqdm(total=len(all_dates), desc="扫描日期") as pbar:
        for date_str in all_dates:
            date_path = get_date_folder_path(input_base_dir, date_str)
            if date_path and os.path.exists(date_path):
                # 检查是否有CSV文件
                csv_files = glob.glob(os.path.join(date_path, "*.csv"))
                if csv_files:
                    available_dates.append(date_str)
            pbar.update(1)

    return available_dates


def create_month_folders_in_output(output_base_dir, date_list):
    """
    在输出目录中创建月份文件夹

    Parameters:
    output_base_dir: 输出基础目录
    date_list: 日期列表

    Returns:
    month_folders: 创建的月份文件夹列表
    """
    month_folders = set()

    for date_str in date_list:
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            year_month_str = date_obj.strftime("%Y_%m")
            month_folders.add(year_month_str)
        except:
            continue

    # 创建月份文件夹
    for month_folder in month_folders:
        month_path = os.path.join(output_base_dir, month_folder)
        os.makedirs(month_path, exist_ok=True)

    return list(month_folders)


def main():
    """
    主函数：批量处理多日期的ACA数据
    """

    # 设置路径
    input_base_dir = r"X:\NJM_Item\ACA数据处理后\ACA_C13_02"
    output_base_dir = r"X:\NJM_Item\ACA_对接优化数据集\原始数据拼接1440"

    # 定义处理日期范围
    start_date = "2023-01-11"
    end_date = "2023-12-31"

    print("=" * 70)
    print("ACA数据批量拼接与可视化程序")
    print(f"输入目录: {input_base_dir}")
    print(f"输出目录: {output_base_dir}")
    print(f"处理日期范围: {start_date} 到 {end_date}")
    print(f"并行工作数: {PARALLEL_WORKERS}")
    print("=" * 70)

    # 创建输出基础目录
    os.makedirs(output_base_dir, exist_ok=True)

    # 查找实际存在的日期
    date_list = find_available_dates(input_base_dir, start_date, end_date)

    if not date_list:
        print("错误: 未找到任何可处理的日期文件夹")
        return

    print(f"\n找到 {len(date_list)} 个可处理的日期")
    print("前10个日期:", ", ".join(date_list[:10]))
    if len(date_list) > 10:
        print(f"... 还有 {len(date_list) - 10} 个日期")

    # 在输出目录中创建月份文件夹
    print("\n创建输出月份文件夹...")
    month_folders = create_month_folders_in_output(output_base_dir, date_list)
    month_folders.sort()
    print(f"创建了 {len(month_folders)} 个月份文件夹: {', '.join(month_folders)}")
    print()

    # 用于存储所有处理结果
    all_results = []
    success_count = 0
    skip_count = 0
    error_count = 0

    print(f"开始批量处理...\n")

    # 准备并行处理参数
    process_args = [(date_str, input_base_dir, output_base_dir) for date_str in date_list]

    # 使用并行处理
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        # 提交所有任务
        future_to_date = {executor.submit(process_single_date_wrapper, args): args[0] for args in process_args}

        # 使用tqdm显示进度
        with tqdm(total=len(date_list), desc="处理进度") as pbar:
            for future in as_completed(future_to_date):
                date_str = future_to_date[future]
                try:
                    result = future.result()
                    all_results.append(result)

                    if result['status'] == 'success':
                        success_count += 1
                        pbar.write(f"完成: {date_str} ({result['file_count']}文件, {result['total_points']:,}点)")
                    elif result['status'] == 'skipped':
                        skip_count += 1
                        pbar.write(f"跳过: {date_str} ({result.get('reason', '未知原因')})")
                    elif result['status'] == 'error':
                        error_count += 1
                        pbar.write(f"错误: {date_str} - {result.get('error', '未知错误')}")

                except Exception as e:
                    error_count += 1
                    pbar.write(f"异常: {date_str} - {str(e)}")

                pbar.update(1)

    # 打印汇总信息
    print("\n" + "=" * 70)
    print("批量处理完成!")
    print("=" * 70)
    print(f"扫描到日期: {len(date_list)} 天")
    print(f"成功处理: {success_count} 天")
    print(f"跳过: {skip_count} 天 (无数据或文件夹为空)")
    print(f"错误: {error_count} 天")
    print("=" * 70)

    # 如果成功处理了数据，生成汇总报告
    successful_results = [r for r in all_results if r.get('status') == 'success']

    if successful_results:
        print("\n生成汇总报告...")

        # 创建汇总报告
        summary_path = os.path.join(output_base_dir, "batch_processing_summary.txt")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(f"ACA数据批量处理汇总报告\n")
            f.write(f"输入目录: {input_base_dir}\n")
            f.write(f"输出目录: {output_base_dir}\n")
            f.write(f"处理日期范围: {start_date} 到 {end_date}\n")
            f.write(f"生成时间: {pd.Timestamp.now()}\n")
            f.write(f"并行处理工作数: {PARALLEL_WORKERS}\n")
            f.write("=" * 70 + "\n\n")

            # 目录结构信息
            f.write("1. 目录结构信息\n")
            f.write("-" * 40 + "\n")
            f.write(f"输入基础路径: {input_base_dir}\n")
            input_year_month_folders = get_year_month_folders(input_base_dir)
            if input_year_month_folders:
                f.write(f"输入年月文件夹: {', '.join(input_year_month_folders)}\n")
            f.write(f"输出基础路径: {output_base_dir}\n")
            f.write(f"输出年月文件夹: {', '.join(month_folders)}\n")
            f.write(f"扫描到有效日期: {len(date_list)} 天\n")
            f.write(f"成功处理日期: {len(successful_results)} 天\n")
            f.write("\n")

            # 处理统计
            f.write("2. 处理统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"扫描到日期总数: {len(date_list)}\n")
            f.write(f"成功处理: {success_count}\n")
            f.write(f"跳过: {skip_count} (无数据或文件夹为空)\n")
            f.write(f"错误: {error_count}\n\n")

            # 传感器信息
            if successful_results:
                sensor_ids = list(set([r.get('sensor_id', '未知') for r in successful_results]))
                sensor_names = list(set([r.get('sensor_name', '未知') for r in successful_results]))

                f.write("3. 传感器信息\n")
                f.write("-" * 40 + "\n")
                for i, (sensor_id, sensor_name) in enumerate(zip(sensor_ids, sensor_names)):
                    f.write(f"传感器 {i + 1}: {sensor_id} - {sensor_name}\n")
                f.write("\n")

            # 按月份统计
            f.write("4. 按月份处理统计\n")
            f.write("-" * 40 + "\n")

            # 统计每个月份的处理情况
            month_stats = {}
            for result in successful_results:
                try:
                    date_obj = datetime.strptime(result['date'], "%Y-%m-%d")
                    year_month = date_obj.strftime("%Y-%m")
                    if year_month not in month_stats:
                        month_stats[year_month] = {
                            'date_count': 0,
                            'file_count': 0,
                            'total_points': 0
                        }
                    month_stats[year_month]['date_count'] += 1
                    month_stats[year_month]['file_count'] += result.get('file_count', 0)
                    month_stats[year_month]['total_points'] += result.get('total_points', 0)
                except:
                    continue

            # 按月份排序并输出
            for year_month in sorted(month_stats.keys()):
                stats = month_stats[year_month]
                f.write(f"{year_month}: {stats['date_count']}天, "
                        f"{stats['file_count']}文件, "
                        f"{stats['total_points']:,}数据点\n")
            f.write("\n")

            # 成功处理的日期列表
            f.write("5. 成功处理的日期 (前50个)\n")
            f.write("-" * 40 + "\n")
            for i, result in enumerate(successful_results[:50]):
                f.write(f"{i + 1:3d}. {result['date']}: {result.get('sensor_id', '未知')} - "
                        f"{result.get('sensor_name', '未知')} - "
                        f"{result.get('file_count', 0)}文件, "
                        f"{result.get('total_points', 0):,}点, "
                        f"范围[{result.get('data_min', 0):.6f}, {result.get('data_max', 0):.6f}]\n")

            if len(successful_results) > 50:
                f.write(f"... 还有 {len(successful_results) - 50} 个日期\n")
            f.write("\n")

            # 整体统计数据
            if successful_results:
                total_files = sum([r.get('file_count', 0) for r in successful_results])
                total_points = sum([r.get('total_points', 0) for r in successful_results])
                all_data_mins = [r.get('data_min', 0) for r in successful_results]
                all_data_maxs = [r.get('data_max', 0) for r in successful_results]

                f.write("6. 整体统计\n")
                f.write("-" * 40 + "\n")
                f.write(f"总文件数: {total_files:,}\n")
                f.write(f"总数据点数: {total_points:,}\n")
                f.write(f"整体最小值: {min(all_data_mins):.6f}\n")
                f.write(f"整体最大值: {max(all_data_maxs):.6f}\n")
                f.write(f"平均每个日期数据点: {total_points / len(successful_results):,.0f}\n")
                f.write(f"平均每个日期文件数: {total_files / len(successful_results):.1f}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("批量处理完成!\n")
            f.write("=" * 70 + "\n")

        # 保存汇总JSON文件
        summary_json_path = os.path.join(output_base_dir, "batch_processing_summary.json")
        summary_json = {
            "processing_date_range": f"{start_date} 到 {end_date}",
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "parallel_workers": PARALLEL_WORKERS,
            "total_scanned_dates": len(date_list),
            "successful_days": success_count,
            "skipped_days": skip_count,
            "error_days": error_count,
            "input_directory": input_base_dir,
            "output_directory": output_base_dir,
            "input_year_month_folders": get_year_month_folders(input_base_dir),
            "output_month_folders": month_folders,
            "processed_results": successful_results,
            "sensor_mapping": SENSOR_MAPPING
        }

        with open(summary_json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, ensure_ascii=False, indent=2)

        print(f"汇总报告已保存: {summary_path}")
        print(f"汇总JSON已保存: {summary_json_path}")

        # 打印成功处理的日期
        print("\n成功处理的日期 (前20个):")
        for i, result in enumerate(successful_results[:20]):
            print(f"{i + 1:2d}. {result['date']}: {result.get('sensor_id', '未知')} - "
                  f"{result.get('file_count', 0):4d}文件, "
                  f"{result.get('total_points', 0):10,}点")

        if len(successful_results) > 20:
            print(f"... 还有 {len(successful_results) - 20} 个日期")

    print(f"\n输出目录结构:")
    print(f"{output_base_dir}")
    print(f"├── 2023_01/")
    print(f"│   ├── 2023-01-11/")
    print(f"│   │   ├── 2023-01-11_concatenated.json")
    print(f"│   │   ├── 2023-01-11_concatenated.png")
    print(f"│   │   └── 2023-01-11_report.txt")
    print(f"│   ├── 2023-01-12/")
    print(f"│   │   ├── 2023-01-12_concatenated.json")
    print(f"│   │   ├── 2023-01-12_concatenated.png")
    print(f"│   │   └── 2023-01-12_report.txt")
    print(f"│   └── ...")
    print(f"├── 2023_02/")
    print(f"├── ...")
    print(f"├── 2023_12/")
    print(f"│   ├── 2023-12-30/")
    print(f"│   └── 2023-12-31/")
    print(f"├── batch_processing_summary.txt")
    print(f"└── batch_processing_summary.json")
    print("=" * 70)
    print(f"输入目录结构示例:")
    print(f"{input_base_dir}")
    print(f"├── 2023_01/")
    print(f"│   ├── 2023-01-11/")
    print(f"│   │   ├── ACA_C13_02_001.csv")
    print(f"│   │   ├── ACA_C13_02_002.csv")
    print(f"│   │   └── ...")
    print(f"│   ├── 2023-01-12/")
    print(f"│   └── ...")
    print(f"├── 2023_02/")
    print(f"└── ...")
    print("=" * 70)


if __name__ == "__main__":

    # 运行主程序
    main()