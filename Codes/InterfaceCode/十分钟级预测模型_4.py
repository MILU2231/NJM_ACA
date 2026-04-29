# -*- coding: utf-8 -*-
"""
十分钟级LSTM预测模型
基于十分钟级时间序列进行LSTM预测
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =========================
# 字体设置函数
# =========================
def setup_matplotlib_fonts():
    """设置Matplotlib字体，解决中文显示问题"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        available_fonts = set([f.name for f in fm.fontManager.ttflist])
        if 'SimHei' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            print("✅ 使用 SimHei 字体")
        elif 'Microsoft YaHei' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
            print("✅ 使用 Microsoft YaHei 字体")
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            print("⚠️ 未找到中文字体，使用 DejaVu Sans")

        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14

    except Exception as e:
        print(f"⚠️ 字体设置失败: {e}")
        plt.rcParams['axes.unicode_minus'] = False


setup_matplotlib_fonts()

# 固定随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# 路径配置
# =========================
# 十分钟级数据路径（由特征工程代码生成）
十分钟级数据路径 = r"X:\NJM_Item\ACA_对接优化数据集\10min级特征_聚类结果\10min级特征_20260204_045848\10min级时间序列.csv"

# 创建结果目录
当前时间 = datetime.now().strftime("%Y%m%d_%H%M%S")
结果根目录 = os.path.join(
    r"X:\NJM_Item\ACA_对接优化数据集\10min级特征_预测结果",
    f"LSTM_10分钟级预测_{当前时间}")
os.makedirs(结果根目录, exist_ok=True)

print(f"📁 结果将保存在: {结果根目录}")


# =========================
# 数据加载和预处理
# =========================
def 数据探索(df):
    """探索数据结构和类型"""
    print("\n🔍 数据探索:")
    print("=" * 50)

    print(f"数据形状: {df.shape}")

    print("\n📊 数据类型统计:")
    dtypes_count = df.dtypes.value_counts()
    for dtype, count in dtypes_count.items():
        print(f"  {dtype}: {count} 列")

    print("\n🔤 对象/字符串列详情:")
    object_cols = df.select_dtypes(include=['object']).columns
    if len(object_cols) > 0:
        for col in object_cols:
            print(f"  {col}: {df[col].dtype}")
            if df[col].notna().sum() > 0:
                print(f"    示例值: {df[col].iloc[0]}")
                print(f"    唯一值数量: {df[col].nunique()}")
    else:
        print("  无对象/字符串列")

    print("\n🔢 数值列详情:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"  数值列数量: {len(numeric_cols)}")
    if len(numeric_cols) > 0:
        print(f"  前10个数值列: {list(numeric_cols)[:10]}")

    print("-" * 50)
    return df


def 加载十分钟级数据(文件路径):
    """加载十分钟级时间序列数据"""
    print("📂 加载十分钟级数据...")

    try:
        # 尝试CSV格式
        if 文件路径.endswith('.csv'):
            df = pd.read_csv(文件路径, encoding='utf-8-sig')
            print("✅ CSV文件加载成功")
        # 尝试JSON格式
        elif 文件路径.endswith('.json'):
            with open(文件路径, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            print("✅ JSON文件加载成功")
        else:
            print("❌ 不支持的文件格式")
            return None

        print(f"原始数据形状: {df.shape}")

        # 确保时间列存在
        时间列名 = None
        for 列名 in ['窗口开始时间', 'timestamp', 'time', '日期', '时间']:
            if 列名 in df.columns:
                时间列名 = 列名
                break

        if 时间列名 is None:
            print("⚠️ 未找到标准时间列，检查所有列:")
            for col in df.columns[:10]:  # 只显示前10列
                print(f"  - {col}: {df[col].dtype}")

            # 尝试自动识别时间列
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower() or '时间' in col or '日期' in col:
                    时间列名 = col
                    print(f"📌 自动识别时间列: {col}")
                    break

        if 时间列名 is None and len(df) > 0:
            print("⚠️ 未找到时间列，使用索引作为时间")
            df['窗口开始时间'] = pd.date_range(start='2023-01-01', periods=len(df), freq='10min')
        elif 时间列名:
            df[时间列名] = pd.to_datetime(df[时间列名])
            df = df.rename(columns={时间列名: '窗口开始时间'})

        # 按时间排序
        df = df.sort_values('窗口开始时间')

        print(f"\n✅ 已加载 {len(df)} 个10分钟窗口的数据")
        print(f"📋 特征数量: {len(df.columns) - 1}")
        print(f"📊 时间范围: {df['窗口开始时间'].min()} 到 {df['窗口开始时间'].max()}")
        print(f"📊 时间跨度: {(df['窗口开始时间'].max() - df['窗口开始时间'].min()).total_seconds() / 3600:.1f} 小时")

        # 数据探索
        数据探索(df)

        return df
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def 准备十分钟特征和目标(df):
    """准备特征列和目标列（十分钟级）- 精简版"""
    print("\n🎯 准备十分钟级特征和目标列...")

    # 1. 首先找出所有数值列
    数值列 = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"📊 数值列数量: {len(数值列)}")

    if len(数值列) == 0:
        print("❌ 没有找到数值列，尝试转换数据类型...")
        # 尝试将可能的数据转换为数值
        for col in df.columns:
            if col != '窗口开始时间':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass

        数值列 = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"转换后数值列数量: {len(数值列)}")

    # 2. 排除明显的时间/ID列和不需要的列
    排除列 = ['年', '月', '日', '小时', '十分钟块', '星期几', '是否工作日', '是否周末',
              '是否高峰时段', '是否夜间', '时间段编码', '事件数量', '列车事件数量',
              '非列车事件数量', '事件数量_原始', '标签均值', 'P1均值', '时间段']

    # 添加字符串列的排除
    字符串列 = df.select_dtypes(include=['object']).columns.tolist()
    if 字符串列:
        print(f"🔤 排除字符串列: {字符串列}")
        排除列.extend(字符串列)

    # 3. 特征列：数值列中排除目标列和排除列
    特征列 = [col for col in 数值列 if col not in 排除列]

    # 4. 目标列 - 只选择核心指标
    目标列 = []
    for col in ['中心偏移距离', '分布半径', '聚类密度', '距离均值']:
        if col in 数值列 and col not in 排除列:
            目标列.append(col)

    # 如果没找到目标列，使用数值列的前几个（排除不需要的）
    if not 目标列:
        print("⚠️ 未找到核心目标列，从数值列选择")
        for col in 数值列:
            if (col not in 排除列 and
                    col not in 特征列[:5] and
                    '标签' not in col and 'P1' not in col):
                目标列.append(col)
                if len(目标列) >= 2:
                    break

    # 5. 确保目标列在特征列中被排除
    特征列 = [col for col in 特征列 if col not in 目标列]

    # 6. 限制特征数量，避免维度灾难
    if len(特征列) > 25:
        print(f"⚠️ 特征列过多 ({len(特征列)})，选择前25个")
        # 选择方差较大的特征
        特征方差 = df[特征列].var().sort_values(ascending=False)
        特征列 = 特征方差.index[:25].tolist()

    print(f"\n🎯 最终选择的目标列 ({len(目标列)}个): {目标列}")
    print(f"📊 最终选择的特征列 ({len(特征列)}个): {特征列}")

    return 特征列, 目标列


def 创建十分钟滑动窗口(df, 特征列, 目标列, window_size=72):
    """创建十分钟级的滑动窗口数据集"""
    print(f"\n🔄 创建十分钟级滑动窗口 (窗口大小: {window_size} 个10分钟窗口 = {window_size * 10 / 60:.1f} 小时)")

    # 1. 提取数据前，确保数据类型正确
    print("🔍 检查数据类型...")

    for col in 特征列 + 目标列:
        if col not in df.columns:
            print(f"❌ 错误: 列 '{col}' 不存在于数据中")
            return None, None, None, None

        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"⚠️ 警告: 列 '{col}' 不是数值类型 ({df[col].dtype})，尝试转换...")
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                print(f"  转换失败: {e}")
                return None, None, None, None

    # 2. 提取数据
    特征数据 = df[特征列].values
    目标数据 = df[目标列].values

    print(f"原始数据形状 - 特征: {特征数据.shape}, 目标: {目标数据.shape}")

    # 3. 处理NaN值
    print("🧹 处理NaN值...")
    nan_count_features = np.isnan(特征数据).sum()
    nan_count_targets = np.isnan(目标数据).sum()

    if nan_count_features > 0 or nan_count_targets > 0:
        print(f"  特征数据NaN数量: {nan_count_features}")
        print(f"  目标数据NaN数量: {nan_count_targets}")

        # 使用线性插值填充NaN
        for i in range(特征数据.shape[1]):
            col_data = 特征数据[:, i]
            if np.isnan(col_data).any():
                nan_mask = np.isnan(col_data)
                col_data[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(~nan_mask)[0],
                    col_data[~nan_mask]
                )

        for i in range(目标数据.shape[1]):
            col_data = 目标数据[:, i]
            if np.isnan(col_data).any():
                nan_mask = np.isnan(col_data)
                col_data[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(~nan_mask)[0],
                    col_data[~nan_mask]
                )

    特征数据 = np.nan_to_num(特征数据, nan=0.0)
    目标数据 = np.nan_to_num(目标数据, nan=0.0)

    # 4. 标准化特征
    print("📊 标准化特征数据...")
    特征标准化器 = StandardScaler()
    特征_scaled = 特征标准化器.fit_transform(特征数据)

    # 5. 标准化目标（每个目标单独标准化）
    print("📊 标准化目标数据...")
    目标标准化器列表 = []
    目标_scaled_list = []

    for i in range(len(目标列)):
        print(f"  标准化目标 '{目标列[i]}'...")
        目标标准化器 = StandardScaler()
        目标_scaled = 目标标准化器.fit_transform(目标数据[:, i].reshape(-1, 1))
        目标_scaled_list.append(目标_scaled)
        目标标准化器列表.append(目标标准化器)

    目标_scaled = np.hstack(目标_scaled_list)

    # 6. 创建滑动窗口
    print("🔄 创建滑动窗口...")
    X, y = [], []

    total_windows = len(特征_scaled) - window_size
    print(f"  可创建的窗口数量: {total_windows}")

    for i in range(total_windows):
        X.append(特征_scaled[i:i + window_size])
        y.append(目标_scaled[i + window_size])

    X = np.array(X)
    y = np.array(y)

    print(f"\n✅ 创建了 {len(X)} 个十分钟级样本")
    print(f"  输入形状: {X.shape}")  # (样本数, 窗口大小, 特征数)
    print(f"  输出形状: {y.shape}")  # (样本数, 目标数)

    return X, y, 特征标准化器, 目标标准化器列表


# =========================
# 模型定义
# =========================
class CNN_LSTM模型(nn.Module):
    """CNN-LSTM混合模型（十分钟级版本）"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=1, bidirectional=False):
        super().__init__()

        # 1D CNN层
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # LSTM层
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # 注意力机制
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 8), output_dim)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape

        # CNN处理
        x_cnn = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x_cnn = self.conv1d(x_cnn)  # [batch, 64, seq_len]
        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, 64]

        # LSTM处理
        lstm_out, _ = self.lstm(x_cnn)  # [batch, seq_len, hidden_dim*2]

        # 注意力
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # 加权求和
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_dim*2]

        # 全连接
        output = self.fc(context)  # [batch, output_dim]

        return output


# =========================
# 数据集
# =========================
class 十分钟级数据集(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx].copy()

        # 数据增强
        if self.augment and np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise

            scale = np.random.uniform(0.98, 1.02)
            x = x * scale

        return torch.FloatTensor(x), torch.FloatTensor(y)


# =========================
# sklearn模型包装器
# =========================
class SklearnWrapper:
    """把 sklearn 回归模型包装成类似 torch model 的可调用对象"""

    def __init__(self, model, window_size):
        self.model = model
        self.window_size = window_size

    def eval(self):
        return

    def __call__(self, x_tensor):
        # x_tensor: [batch, seq_len, input_dim]
        x = x_tensor.cpu().numpy()
        batch = x.shape[0]
        flat = x.reshape(batch, -1)
        preds = self.model.predict(flat).reshape(batch, 1)
        return torch.from_numpy(preds.astype(np.float32))


# =========================
# 训练模型
# =========================
def 训练十分钟模型(X_train, y_train, X_val, y_val, 输入维度, 模型名称, 早停耐心=25):
    """训练十分钟级的预测模型"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 使用设备: {device}")

    # 创建数据集
    train_dataset = 十分钟级数据集(X_train, y_train, augment=True)
    val_dataset = 十分钟级数据集(X_val, y_val, augment=False)

    batch_size = min(32, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  训练批次大小: {batch_size}")
    print(f"  训练批次数量: {len(train_loader)}")

    # 根据样本量自动选择模型复杂度
    if len(X_train) < 200:
        chosen_hidden = 32
        chosen_layers = 1
        chosen_bi = False
    else:
        chosen_hidden = 64
        chosen_layers = 2
        chosen_bi = True

    model = CNN_LSTM模型(
        input_dim=输入维度,
        hidden_dim=chosen_hidden,
        num_layers=chosen_layers,
        output_dim=1,
        bidirectional=chosen_bi
    ).to(device)

    print(f"\n🤖 训练 {模型名称} 模型（十分钟级）")
    print(f"  输入维度: {输入维度}")
    print(f"  训练样本: {len(X_train)}")
    print(f"  验证样本: {len(X_val)}")
    print(f"  模型参数: hidden_dim={chosen_hidden}, layers={chosen_layers}, bidirectional={chosen_bi}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # 损失函数
    criterion = nn.HuberLoss(delta=1.0)

    # 训练参数
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience_counter = 0

    # 训练循环
    for epoch in range(200):
        # 训练
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        if len(train_loader) == 0:
            continue

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        valid_val_batches = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                val_loss += loss.item()
                valid_val_batches += 1

        if valid_val_batches == 0:
            continue

        avg_val_loss = val_loss / valid_val_batches
        val_losses.append(avg_val_loss)

        # 学习率调整
        scheduler.step(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                '模型配置': {
                    'input_dim': 输入维度,
                    'hidden_dim': chosen_hidden,
                    'num_layers': chosen_layers,
                    'bidirectional': chosen_bi
                }
            }, os.path.join(结果根目录, f"最佳_{模型名称}_10分钟模型.pth"))
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= 早停耐心:
            print(f"🎯 {模型名称} 早停触发于 epoch {epoch + 1}")
            break

        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"  Epoch {epoch + 1:03d} | 训练损失: {avg_train_loss:.6f} | 验证损失: {avg_val_loss:.6f} | LR: {current_lr:.6f}")

    # 保存训练历史
    损失_df = pd.DataFrame({
        "训练损失": train_losses,
        "验证损失": val_losses
    })
    损失_df.to_csv(
        os.path.join(结果根目录, f"{模型名称}_10分钟训练历史.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 绘制训练历史
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失', linewidth=2)
    plt.plot(val_losses, label='验证损失', linewidth=2)
    plt.xlabel('训练轮次', fontsize=11)
    plt.ylabel('损失值', fontsize=11)
    plt.title(f'{模型名称} - 十分钟级训练历史', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(结果根目录, f"{模型名称}_10分钟训练历史图.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ {模型名称} 训练完成，最佳验证损失: {best_val_loss:.6f}")
    return model, device


# =========================
# 十分钟级滚动预测
# =========================
def 进行十分钟滚动预测(df, 模型列表, device, 特征标准化器, 目标标准化器列表, 特征列, 目标列, window_size=72):
    """十分钟级滑窗递推预测"""

    for model in 模型列表:
        if hasattr(model, 'eval'):
            model.eval()

    特征数据 = df[特征列].values
    特征数据 = np.nan_to_num(特征数据, nan=0.0)
    特征_scaled = 特征标准化器.transform(特征数据)

    总窗口数 = len(df)
    if 总窗口数 <= window_size:
        print("❌ 数据量不足")
        return None

    # 存储预测结果
    预测结果列表 = [[] for _ in range(len(目标列))]
    真实结果列表 = [[] for _ in range(len(目标列))]
    预测时间列表 = []

    print(f"\n🔮 开始十分钟级滑窗滚动预测...")
    print(f"   总窗口数: {总窗口数}")
    print(f"   预测窗口数: {总窗口数 - window_size}")

    with torch.no_grad():
        for i in range(window_size, 总窗口数):
            # 用前window_size个10分钟预测第i个10分钟
            当前窗口 = 特征_scaled[i - window_size:i]
            输入_tensor = torch.tensor(当前窗口, dtype=torch.float32).unsqueeze(0).to(device)

            # 预测每个目标
            for j, model in enumerate(模型列表):
                if isinstance(model, SklearnWrapper):
                    pred_scaled = model(输入_tensor).cpu().numpy()
                else:
                    pred_scaled = model(输入_tensor).cpu().numpy()

                if len(pred_scaled.shape) > 1:
                    pred_scaled = pred_scaled[0][0]
                else:
                    pred_scaled = pred_scaled[0]

                # 反标准化
                pred_original = 目标标准化器列表[j].inverse_transform([[pred_scaled]])[0][0]
                预测结果列表[j].append(pred_original)

                # 获取真实值
                真实值 = df[目标列[j]].iloc[i] if j < len(目标列) else 0
                真实结果列表[j].append(真实值)

            预测时间列表.append(df["窗口开始时间"].iloc[i])

            if (i - window_size + 1) % 500 == 0:
                print(f"   进度: {i - window_size + 1}/{总窗口数 - window_size}")

    print(f"✅ 十分钟级滚动预测完成，共预测 {len(预测时间列表)} 个10分钟窗口")

    # 创建结果DataFrame
    结果数据 = {'日期': 预测时间列表}
    for j, 目标名 in enumerate(目标列):
        结果数据[f'真实_{目标名}'] = 真实结果列表[j]
        结果数据[f'预测_{目标名}'] = 预测结果列表[j]
        结果数据[f'{目标名}_误差'] = np.abs(np.array(真实结果列表[j]) - np.array(预测结果列表[j]))

    结果_df = pd.DataFrame(结果数据)

    # 处理NaN值
    结果_df = 结果_df.fillna(0)

    结果_df.to_csv(
        os.path.join(结果根目录, "十分钟级滚动预测结果.csv"),
        index=False, encoding="utf-8-sig"
    )

    return 结果_df


# =========================
# 评估函数
# =========================
def 评估预测结果(结果_df, 目标列):
    """评估预测结果"""
    print("\n📈 评估预测结果:")
    print("=" * 50)

    评估结果 = {}

    for 目标名 in 目标列:
        真实列 = f'真实_{目标名}'
        预测列 = f'预测_{目标名}'

        if 真实列 in 结果_df.columns and 预测列 in 结果_df.columns:
            真实值 = 结果_df[真实列].values
            预测值 = 结果_df[预测列].values

            # 过滤NaN
            有效掩码 = ~(np.isnan(真实值) | np.isnan(预测值))
            真实值 = 真实值[有效掩码]
            预测值 = 预测值[有效掩码]

            if len(真实值) > 0:
                try:
                    r2 = r2_score(真实值, 预测值)
                    rmse = np.sqrt(mean_squared_error(真实值, 预测值))
                    mae = mean_absolute_error(真实值, 预测值)

                    评估结果[目标名] = {
                        'R2': r2,
                        'RMSE': rmse,
                        'MAE': mae,
                        '样本数': len(真实值)
                    }

                    print(f"\n{目标名}:")
                    print(f"  R²: {r2:.4f}")
                    print(f"  RMSE: {rmse:.4f}")
                    print(f"  MAE: {mae:.4f}")
                    print(f"  样本数: {len(真实值)}")

                    if r2 > 0.7:
                        print("  ✅ 预测效果优秀")
                    elif r2 > 0.5:
                        print("  👍 预测效果良好")
                    elif r2 > 0.3:
                        print("  👌 预测效果一般")
                    else:
                        print("  ⚠️ 预测效果有待改进")

                except Exception as e:
                    print(f"  ❌ 评估失败: {e}")

    # 保存评估结果
    if 评估结果:
        with open(os.path.join(结果根目录, "评估结果.txt"), "w", encoding="utf-8") as f:
            f.write("十分钟级预测评估结果\n")
            f.write("=" * 60 + "\n\n")

            for 目标名, 指标 in 评估结果.items():
                f.write(f"{目标名}:\n")
                f.write(f"  R²: {指标['R2']:.4f}\n")
                f.write(f"  RMSE: {指标['RMSE']:.4f}\n")
                f.write(f"  MAE: {指标['MAE']:.4f}\n")
                f.write(f"  样本数: {指标['样本数']}\n\n")

    return 评估结果


# =========================
# 十分钟级预测对比图绘制函数（优化排版版）
# =========================
def 绘制十分钟级预测对比图(结果_df, 目标列, 评估结果):
    """绘制十分钟级预测对比图（优化排版）"""

    if 结果_df is None or len(结果_df) == 0:
        print("⚠️ 没有预测结果数据，跳过图表绘制")
        return

    print("\n🖼️ 绘制十分钟级预测对比图...")

    # 确保日期列正确
    if '日期' in 结果_df.columns:
        预测日期 = pd.to_datetime(结果_df["日期"])
    elif '窗口开始时间' in 结果_df.columns:
        预测日期 = pd.to_datetime(结果_df["窗口开始时间"])
    else:
        print("⚠️ 未找到日期列，使用索引作为时间轴")
        预测日期 = range(len(结果_df))

    # 为每个目标绘制单独的对比图
    for 目标名 in 目标列:
        真实列 = f'真实_{目标名}'
        预测列 = f'预测_{目标名}'
        误差列 = f'{目标名}_误差'

        if 真实列 not in 结果_df.columns or 预测列 not in 结果_df.columns:
            print(f"⚠️ 跳过 {目标名}，缺少必要的列")
            continue

        # 获取R²值
        r2 = 评估结果.get(目标名, {}).get('R2', 0) if 评估结果 else 0

        # 1. 十分钟级对比图（优化排版）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.3)  # 增加子图间距

        # 子图1: 真实vs预测对比
        # 采样显示，避免图表过于密集
        if len(预测日期) > 1000:
            sample_indices = np.linspace(0, len(预测日期) - 1, 1000, dtype=int)
            ax1.plot(预测日期.iloc[sample_indices], 结果_df[真实列].iloc[sample_indices],
                     'b-', label='真实值', linewidth=1.2, alpha=0.8)
            ax1.plot(预测日期.iloc[sample_indices], 结果_df[预测列].iloc[sample_indices],
                     'r--', label='预测值', linewidth=1.2, alpha=0.8)
        else:
            ax1.plot(预测日期, 结果_df[真实列], 'b-', label='真实值', linewidth=1.2, alpha=0.8)
            ax1.plot(预测日期, 结果_df[预测列], 'r--', label='预测值', linewidth=1.2, alpha=0.8)

        ax1.set_title(f'{目标名} - 十分钟级预测对比 (R² = {r2:.4f})', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel(f'{目标名}值', fontsize=11)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 设置x轴标签格式，避免重叠
        if len(预测日期) > 50:
            ax1.xaxis.set_major_locator(plt.MaxNLocator(10))

        # 子图2: 误差序列
        if 误差列 in 结果_df.columns:
            errors = 结果_df[误差列]
        else:
            errors = np.abs(结果_df[真实列] - 结果_df[预测列])

        # 采样显示误差
        if len(预测日期) > 1000:
            sample_indices = np.linspace(0, len(预测日期) - 1, 1000, dtype=int)
            ax2.plot(预测日期.iloc[sample_indices], errors.iloc[sample_indices],
                     'g-', alpha=0.7, linewidth=0.8)
        else:
            ax2.plot(预测日期, errors, 'g-', alpha=0.7, linewidth=0.8)

        ax2.fill_between(预测日期, 0, errors, alpha=0.2, color='green')
        ax2.axhline(y=errors.mean(), color='r', linestyle='--', linewidth=1.5,
                    label=f'平均误差: {errors.mean():.4f}')
        ax2.set_title(f'{目标名} - 预测误差序列', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('时间', fontsize=11)
        ax2.set_ylabel('绝对误差', fontsize=11)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        # 设置x轴标签格式，避免重叠
        if len(预测日期) > 50:
            ax2.xaxis.set_major_locator(plt.MaxNLocator(10))

        plt.tight_layout()
        plt.savefig(os.path.join(结果根目录, f"{目标名}_十分钟级预测对比图.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 十分钟级分析图（散点图和误差分布）
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plt.subplots_adjust(wspace=0.3)  # 增加子图间距

        # 子图1: 预测vs真实散点图
        scatter = axes[0].scatter(结果_df[真实列], 结果_df[预测列],
                                  alpha=0.5, s=8, c=range(len(结果_df)), cmap='viridis')
        min_val = min(结果_df[真实列].min(), 结果_df[预测列].min())
        max_val = max(结果_df[真实列].max(), 结果_df[预测列].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, linewidth=2, label='完美预测')
        axes[0].set_title(f'{目标名} - 预测vs真实散点图', fontsize=13, fontweight='bold', pad=15)
        axes[0].set_xlabel('真实值', fontsize=11)
        axes[0].set_ylabel('预测值', fontsize=11)
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=axes[0])
        cbar.set_label('时间顺序', fontsize=10)

        # 子图2: 误差分布直方图
        axes[1].hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
        axes[1].axvline(x=errors.mean(), color='red', linestyle='--', linewidth=2,
                        label=f'均值: {errors.mean():.4f}')
        axes[1].axvline(x=errors.median(), color='green', linestyle='--', linewidth=2,
                        label=f'中位数: {errors.median():.4f}')

        # 添加正态分布曲线
        from scipy import stats
        if len(errors) > 1 and errors.std() > 0:
            x = np.linspace(errors.min(), errors.max(), 100)
            axes[1].plot(x, stats.norm.pdf(x, errors.mean(), errors.std()),
                         'r-', linewidth=2, label='正态分布')

        axes[1].set_title(f'{目标名} - 预测误差分布', fontsize=13, fontweight='bold', pad=15)
        axes[1].set_xlabel('绝对误差', fontsize=11)
        axes[1].set_ylabel('密度', fontsize=11)
        axes[1].legend(loc='upper right', fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(结果根目录, f"{目标名}_十分钟级误差分析图.png"), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ 已生成 {目标名} 的十分钟级对比图")

    # 3. 多目标综合对比图（如果目标列多于1个）
    if len(目标列) > 1:
        print("\n📊 生成多目标综合对比图...")

        # 创建多个子图
        n_targets = len(目标列)
        fig, axes = plt.subplots(n_targets, 1, figsize=(16, 4 * n_targets))
        plt.subplots_adjust(hspace=0.4)  # 增加子图间距

        if n_targets == 1:
            axes = [axes]

        for idx, 目标名 in enumerate(目标列):
            真实列 = f'真实_{目标名}'
            预测列 = f'预测_{目标名}'

            if 真实列 in 结果_df.columns and 预测列 in 结果_df.columns:
                r2 = 评估结果.get(目标名, {}).get('R2', 0) if 评估结果 else 0

                # 采样显示，避免图表过于密集
                if len(预测日期) > 500:
                    sample_indices = np.linspace(0, len(预测日期) - 1, 500, dtype=int)
                    axes[idx].plot(预测日期.iloc[sample_indices], 结果_df[真实列].iloc[sample_indices],
                                   'b-', label='真实值', linewidth=0.8, alpha=0.7)
                    axes[idx].plot(预测日期.iloc[sample_indices], 结果_df[预测列].iloc[sample_indices],
                                   'r--', label='预测值', linewidth=0.8, alpha=0.7)
                else:
                    axes[idx].plot(预测日期, 结果_df[真实列], 'b-', label='真实值', linewidth=0.8, alpha=0.7)
                    axes[idx].plot(预测日期, 结果_df[预测列], 'r--', label='预测值', linewidth=0.8, alpha=0.7)

                axes[idx].set_title(f'{目标名} (R² = {r2:.4f})', fontsize=12, fontweight='bold', pad=10)
                axes[idx].set_ylabel(目标名, fontsize=10)
                axes[idx].legend(loc='upper right', fontsize=9)
                axes[idx].grid(True, alpha=0.3)

                # 设置x轴标签格式
                if len(预测日期) > 50:
                    axes[idx].xaxis.set_major_locator(plt.MaxNLocator(8))

                if idx == n_targets - 1:
                    axes[idx].set_xlabel('时间', fontsize=11)
                    axes[idx].tick_params(axis='x', rotation=45)

        plt.suptitle('十分钟级多目标预测对比', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(结果根目录, "十分钟级多目标预测综合对比图.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✅ 已生成多目标综合对比图")

    print("✅ 所有十分钟级预测对比图已生成")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 十分钟级LSTM滑窗预测")
    print(f"📁 输入文件: {十分钟级数据路径}")
    print("=" * 60)

    try:
        # 步骤1: 加载十分钟级数据
        print("\n📊 步骤1：加载十分钟级数据")
        df = 加载十分钟级数据(十分钟级数据路径)

        if df is None or len(df) == 0:
            print("❌ 数据加载失败或数据为空")
            exit(1)

        # 步骤2: 准备特征和目标（精简版）
        print("\n🎯 步骤2：准备十分钟级特征和目标")
        特征列, 目标列 = 准备十分钟特征和目标(df)

        if not 特征列:
            print("❌ 没有找到可用的特征列")
            exit(1)

        if not 目标列:
            print("❌ 没有找到可用的目标列")
            exit(1)

        print(f"\n✅ 目标列选择完成: {目标列}")

        # 窗口大小设置：72个10分钟 = 12小时
        window_size = 72

        # 步骤3: 创建滑动窗口数据集
        print(f"\n🔄 步骤3：创建十分钟级滑动窗口 (窗口大小: {window_size} 个10分钟窗口)")
        X, y, 特征标准化器, 目标标准化器列表 = 创建十分钟滑动窗口(df, 特征列, 目标列, window_size=window_size)

        if X is None or len(X) == 0:
            print("❌ 滑动窗口创建失败")
            exit(1)

        # 保存标准化器
        with open(os.path.join(结果根目录, '十分钟级_特征标准化器.pkl'), 'wb') as f:
            pickle.dump(特征标准化器, f)

        print(f"\n✅ 数据准备完成")
        print(f"  总样本数: {len(X)}")
        print(f"  特征维度: {len(特征列)}")
        print(f"  目标数量: {len(目标列)}")

        # 步骤4: 训练每个目标的模型
        print(f"\n🤖 步骤4：训练 {len(目标列)} 个目标的十分钟级模型")

        # 数据分割（80%训练，20%验证）
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]

        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")

        模型列表 = []
        设备列表 = []

        for i, 目标名 in enumerate(目标列):
            print(f"\n🎯 训练目标 {i + 1}/{len(目标列)}: {目标名}")

            y_target = y[:, i].reshape(-1, 1)
            y_train = y_target[:split_idx]
            y_val = y_target[split_idx:]

            # 保存该目标的标准化器
            with open(os.path.join(结果根目录, f'十分钟级_{目标名}_标准化器.pkl'), 'wb') as f:
                pickle.dump(目标标准化器列表[i], f)

            # 检查样本量，决定使用什么模型
            if len(X_train) < 100:
                print(f"⚠️ 训练样本较少 ({len(X_train)})，使用线性回归")
                X_train_flat = X_train.reshape(len(X_train), -1)
                lr = LinearRegression()
                lr.fit(X_train_flat, y_train.reshape(-1))

                model = SklearnWrapper(lr, window_size)
                device = torch.device("cpu")
            else:
                # 训练LSTM模型
                model, device = 训练十分钟模型(
                    X_train, y_train, X_val, y_val,
                    len(特征列), 目标名, 早停耐心=25
                )
                设备列表.append(device)

            模型列表.append(model)
            print(f"✅ {目标名} 模型训练完成")

        # 步骤5: 十分钟级滚动预测
        print(f"\n🔮 步骤5：十分钟级全量滑窗滚动预测")

        if len(模型列表) > 0:
            # 使用第一个设备的device
            main_device = 设备列表[0] if 设备列表 else torch.device("cpu")

            结果_df = 进行十分钟滚动预测(
                df, 模型列表, main_device,
                特征标准化器, 目标标准化器列表,
                特征列, 目标列, window_size=window_size
            )

            if 结果_df is not None:
                # 步骤6: 评估预测结果
                print(f"\n📈 步骤6：评估十分钟级预测结果")
                评估结果 = 评估预测结果(结果_df, 目标列)

                # 步骤7: 绘制十分钟级对比图
                print(f"\n🖼️ 步骤7：绘制十分钟级预测对比图")
                绘制十分钟级预测对比图(结果_df, 目标列, 评估结果)

                # 显示总体结果
                print("\n" + "=" * 50)
                print("🎯 总体预测结果:")
                print("=" * 50)
                for 目标名, 指标 in 评估结果.items():
                    print(f"{目标名}: R² = {指标['R2']:.4f}")

                print(f"\n✅ 十分钟级预测完成！")
                print(f"   预测窗口数: {len(结果_df)}")
                print(f"   目标数量: {len(目标列)}")

        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print(f"✅ 十分钟级预测程序完成！")
        print(f"   所有结果已保存在：{结果根目录}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback

        traceback.print_exc()