# -*- coding: utf-8 -*-
"""
10-minute Level LSTM Prediction Model - Only predict center shift and distribution radius
Based on 10-minute time series for LSTM prediction, only predict two core indicators
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# =========================
# Font Setup Function
# =========================
def setup_matplotlib_fonts():
    """设置Matplotlib字体，解决中文显示问题"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        available_fonts = set([f.name for f in fm.fontManager.ttflist])
        if 'SimHei' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            print("✅ Using SimHei font")
        elif 'Microsoft YaHei' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
            print("✅ Using Microsoft YaHei font")
        else:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            print("⚠️ Chinese font not found, using DejaVu Sans")

        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
        plt.rcParams['figure.titlesize'] = 14

    except Exception as e:
        print(f"⚠️ Font setup failed: {e}")
        plt.rcParams['axes.unicode_minus'] = False

setup_matplotlib_fonts()

# Fixed random seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# Path Configuration
# =========================
# Path to modified code output
modified_code_output_dir = r"O:\Teamwork\南纪门项目组\NJM_Project_CONDA\ACA_docking_project\Data_720h_results"
FINAL_JSON_DIR = os.path.join(modified_code_output_dir, "final_integrated_json")

# Create main output directory
BASE_PREDICTION_DIR = r"O:\Teamwork\南纪门项目组\NJM_Project_CONDA\ACA_docking_project"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
result_root_dir = os.path.join(
    BASE_PREDICTION_DIR,
    f"Prediction_Results_{current_time}")
os.makedirs(result_root_dir, exist_ok=True)

# Create subdirectories for specific results
CENTER_SHIFT_RESULTS_DIR = os.path.join(result_root_dir, "Center_Shift_Predictions")
CLUSTER_RADIUS_RESULTS_DIR = os.path.join(result_root_dir, "Cluster_Radius_Predictions")
JSON_OUTPUT_DIR = os.path.join(result_root_dir, "Prediction_JSON_Files")

os.makedirs(CENTER_SHIFT_RESULTS_DIR, exist_ok=True)
os.makedirs(CLUSTER_RADIUS_RESULTS_DIR, exist_ok=True)
os.makedirs(JSON_OUTPUT_DIR, exist_ok=True)

print(f"📁 Results will be saved in: {result_root_dir}")
print(f"  - Center Shift Predictions: {CENTER_SHIFT_RESULTS_DIR}")
print(f"  - Cluster Radius Predictions: {CLUSTER_RADIUS_RESULTS_DIR}")
print(f"  - JSON Output Files: {JSON_OUTPUT_DIR}")

# =========================
# Custom JSON Encoder
# =========================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# =========================
# Data Loading and Preprocessing
# =========================
def load_final_json_data(directory):
    """从修改后代码生成的最终JSON文件夹加载数据"""
    print("📂 Loading data from final JSON folder...")

    final_json_dir = os.path.join(directory, "final_integrated_json")
    if not os.path.exists(final_json_dir):
        print(f"❌ Cannot find final JSON folder: {final_json_dir}")
        return None

    json_files = [f for f in os.listdir(final_json_dir) if f.endswith('_final.json')]
    if not json_files:
        print(f"❌ No _final.json files found in {final_json_dir}")
        return None

    print(f"✅ Found {len(json_files)} final JSON files")

    # Read all JSON files and merge
    all_data = []
    for json_file in json_files[:1000]:  # Limit reading to avoid memory issues
        try:
            with open(os.path.join(final_json_dir, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"⚠️ Failed to read {json_file}: {e}")

    if not all_data:
        print("❌ Could not read any JSON data")
        return None

    print(f"✅ Successfully read {len(all_data)} JSON records")

    # Convert to DataFrame
    try:
        df = pd.DataFrame(all_data)

        # Ensure time column exists and convert to datetime
        if 'trigger_timestamp' in df.columns:
            df['trigger_timestamp'] = pd.to_datetime(df['trigger_timestamp'])
            df = df.rename(columns={'trigger_timestamp': 'window_start_time'})

        # Sort by time
        df = df.sort_values('window_start_time')

        print(f"\n✅ Data loading successful:")
        print(f"   Data shape: {df.shape}")
        print(f"   Time range: {df['window_start_time'].min()} to {df['window_start_time'].max()}")
        print(
            f"   Time span: {(df['window_start_time'].max() - df['window_start_time'].min()).total_seconds() / 3600:.1f} hours")

        # Display core indicators statistics
        core_indicators = ['current_center_shift', 'current_cluster_radius',
                           'current_rms', 'current_variance', 'current_frequency_center']
        print(f"\n📊 Core indicators statistics:")
        for indicator in core_indicators:
            if indicator in df.columns:
                print(f"   {indicator}: mean={df[indicator].mean():.4f}, std={df[indicator].std():.4f}")

        return df

    except Exception as e:
        print(f"❌ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_core_features_and_targets(df):
    """准备特征列和目标列 - 仅针对中心偏移距离和分布半径"""
    print("\n🎯 Preparing core features and target columns...")

    # Check if required core indicators exist
    required_indicators = ['current_center_shift', 'current_cluster_radius']
    for indicator in required_indicators:
        if indicator not in df.columns:
            print(f"❌ Missing required indicator: {indicator}")
            # Try using alternative names
            if indicator == 'current_center_shift':
                for alt in ['center_shift_distance', 'center_shift', 'shift_distance']:
                    if alt in df.columns:
                        df['current_center_shift'] = df[alt]
                        print(f"  Using alternative column: {alt}")
                        break
            elif indicator == 'current_cluster_radius':
                for alt in ['cluster_radius', 'radius', 'distribution_radius']:
                    if alt in df.columns:
                        df['current_cluster_radius'] = df[alt]
                        print(f"  Using alternative column: {alt}")
                        break

    # Target columns - only these two core indicators
    target_columns = [col for col in required_indicators if col in df.columns]
    if len(target_columns) < 2:
        print(f"❌ Only {len(target_columns)} valid target columns found!")
        return None, None

    # Feature columns - use other available features, exclude target columns
    exclude_columns = target_columns.copy() + ['sensor_id', 'window_start_time', 'features_29']

    # Add other possible exclusion columns
    exclude_columns.extend(['year', 'month', 'day', 'hour', 'minute', 'second'])

    # Select numeric columns as features
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_columns if col not in exclude_columns]

    # If too few feature columns, add some derived features
    if len(feature_columns) < 5:
        print("⚠️ Few feature columns, adding time features...")
        if 'window_start_time' in df.columns:
            df['timestamp_seconds'] = df['window_start_time'].astype(np.int64) // 10 ** 9
            df['hour_of_day'] = df['window_start_time'].dt.hour
            df['minute_of_hour'] = df['window_start_time'].dt.minute
            df['day_of_week'] = df['window_start_time'].dt.dayofweek

            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in numeric_columns if col not in exclude_columns]

    print(f"\n🎯 Target columns (2 core indicators):")
    for target in target_columns:
        if target in df.columns:
            print(
                f"  {target}: samples={df[target].notna().sum()}, range=[{df[target].min():.4f}, {df[target].max():.4f}]")
        else:
            print(f"  ⚠️ {target}: does not exist!")

    print(f"\n📊 Feature columns ({len(feature_columns)}):")
    if len(feature_columns) <= 10:
        print(f"  {feature_columns}")
    else:
        print(f"  First 10: {feature_columns[:10]}...")

    return feature_columns, target_columns

def create_core_indicators_sliding_window(df, feature_columns, target_columns, window_size=30):
    """为核心指标创建滑动窗口数据集（修改为30个10分钟窗口）"""
    print(
        f"\n🔄 Creating core indicators sliding window (window size: {window_size} 10-min windows = {window_size * 10 / 60:.1f} hours)")

    # Ensure target columns exist
    for target in target_columns:
        if target not in df.columns:
            print(f"❌ Error: Target column '{target}' does not exist")
            return None, None, None, None

    # Extract data
    feature_data = df[feature_columns].values
    target_data = df[target_columns].values

    print(f"Raw data shape - features: {feature_data.shape}, targets: {target_data.shape}")

    # Handle NaN values
    print("🧹 Handling NaN values...")
    nan_count_features = np.isnan(feature_data).sum()
    nan_count_targets = np.isnan(target_data).sum()

    if nan_count_features > 0 or nan_count_targets > 0:
        print(f"  Feature data NaN count: {nan_count_features}")
        print(f"  Target data NaN count: {nan_count_targets}")

        # Use linear interpolation to fill NaN
        for i in range(feature_data.shape[1]):
            col_data = feature_data[:, i]
            if np.isnan(col_data).any():
                nan_mask = np.isnan(col_data)
                col_data[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(~nan_mask)[0],
                    col_data[~nan_mask]
                )

        for i in range(target_data.shape[1]):
            col_data = target_data[:, i]
            if np.isnan(col_data).any():
                nan_mask = np.isnan(col_data)
                col_data[nan_mask] = np.interp(
                    np.where(nan_mask)[0],
                    np.where(~nan_mask)[0],
                    col_data[~nan_mask]
                )

    feature_data = np.nan_to_num(feature_data, nan=0.0)
    target_data = np.nan_to_num(target_data, nan=0.0)

    # Standardize features
    print("📊 Standardizing feature data...")
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(feature_data)

    # Standardize targets (each target separately)
    print("📊 Standardizing target data...")
    target_scalers = []
    targets_scaled_list = []

    for i in range(len(target_columns)):
        print(f"  Standardizing target '{target_columns[i]}'...")
        target_scaler = StandardScaler()
        target_scaled = target_scaler.fit_transform(target_data[:, i].reshape(-1, 1))
        targets_scaled_list.append(target_scaled)
        target_scalers.append(target_scaler)

    targets_scaled = np.hstack(targets_scaled_list)

    # Create sliding windows
    print("🔄 Creating sliding windows...")
    X, y = [], []

    # 修正滑窗计算逻辑：需要 window_size + 1 个数据才能生成1个样本
    total_windows = len(features_scaled) - window_size
    if total_windows < 0:
        total_windows = 0
    print(f"  Number of windows that can be created: {total_windows}")

    if total_windows <= 0:
        print(f"❌ Insufficient data for sliding window! Need at least {window_size + 1} time steps, got {len(features_scaled)}")
        return None, None, None, None

    for i in range(total_windows):
        X.append(features_scaled[i:i + window_size])
        y.append(targets_scaled[i + window_size])

    X = np.array(X)
    y = np.array(y)

    print(f"\n✅ Created {len(X)} 10-minute level samples")
    print(f"  Input shape: {X.shape}")  # (samples, window_size, features)
    print(f"  Output shape: {y.shape}")  # (samples, targets)

    return X, y, feature_scaler, target_scalers

# =========================
# Model Definition - Dual Output Version
# =========================
class DualOutputCNNLSTMModel(nn.Module):
    """CNN-LSTM Hybrid Model - Dual Output Version (simultaneously predicts center shift and cluster radius)"""

    def __init__(self, input_dim, hidden_dim=64, num_layers=1, output_dim=2, bidirectional=False):
        super().__init__()

        # 1D CNN layers
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

        # LSTM layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Fully connected layers - dual output
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim // 2, 8), output_dim)
        )

        # Initialize weights
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

        # CNN processing
        x_cnn = x.transpose(1, 2)  # [batch, input_dim, seq_len]
        x_cnn = self.conv1d(x_cnn)  # [batch, 64, seq_len]
        x_cnn = x_cnn.transpose(1, 2)  # [batch, seq_len, 64]

        # LSTM processing
        lstm_out, _ = self.lstm(x_cnn)  # [batch, seq_len, hidden_dim*2]

        # Attention
        attention_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # [batch, hidden_dim*2]

        # Fully connected
        output = self.fc(context)  # [batch, output_dim]

        return output

# =========================
# Dataset
# =========================
class CoreIndicatorsDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx].copy()

        # Data augmentation
        if self.augment and np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.01, x.shape)
            x = x + noise

            scale = np.random.uniform(0.98, 1.02)
            x = x * scale

        return torch.FloatTensor(x), torch.FloatTensor(y)

# =========================
# Train Model - Modified for Dual Output
# =========================
def train_dual_output_model(X_train, y_train, X_val, y_val, input_dim, model_name="DualOutputModel",
                            early_stop_patience=25):
    """训练双输出模型（同时预测两个核心指标）"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Create datasets
    train_dataset = CoreIndicatorsDataset(X_train, y_train, augment=True)
    val_dataset = CoreIndicatorsDataset(X_val, y_val, augment=False)

    batch_size = min(32, len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Training batch size: {batch_size}")
    print(f"  Number of training batches: {len(train_loader)}")

    # Automatically select model complexity based on sample size
    if len(X_train) < 200:
        chosen_hidden = 32
        chosen_layers = 1
        chosen_bi = False
    else:
        chosen_hidden = 64
        chosen_layers = 2
        chosen_bi = True

    # Use dual output model, output dimension is 2
    model = DualOutputCNNLSTMModel(
        input_dim=input_dim,
        hidden_dim=chosen_hidden,
        num_layers=chosen_layers,
        output_dim=2,  # Dual output: center shift and cluster radius
        bidirectional=chosen_bi
    ).to(device)

    print(f"\n🤖 Training {model_name} model (dual output)")
    print(f"  Input dimension: {input_dim}")
    print(f"  Output dimension: 2 (center_shift, cluster_radius)")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Model parameters: hidden_dim={chosen_hidden}, layers={chosen_layers}, bidirectional={chosen_bi}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
    )

    # Loss function - combined loss for dual output
    criterion = nn.HuberLoss(delta=1.0)

    # Training parameters
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    patience_counter = 0

    # Training loop
    for epoch in range(200):
        # Training
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

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1} (patience limit reached)")
                break

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/200] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(result_root_dir, f'{model_name}_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return model, best_val_loss, train_losses, val_losses

# =========================
# Prediction and Evaluation
# =========================
def evaluate_model(model, X_test, y_test, target_scalers, target_columns):
    """评估模型性能并反标准化预测结果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Predict
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    # Inverse transform to original scale
    y_pred = np.zeros_like(y_pred_scaled)
    y_true = np.zeros_like(y_test)

    for i in range(len(target_columns)):
        y_pred[:, i] = target_scalers[i].inverse_transform(y_pred_scaled[:, i].reshape(-1, 1)).flatten()
        y_true[:, i] = target_scalers[i].inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()

    # Calculate metrics
    metrics = {}
    for i, target in enumerate(target_columns):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        metrics[target] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

        print(f"\n📈 Evaluation for {target}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R² Score: {r2:.4f}")

    # Plot predictions vs actual
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for i, (target, ax) in enumerate(zip(target_columns, axes)):
        ax.plot(y_true[:, i], label='Actual', color='blue', alpha=0.7)
        ax.plot(y_pred[:, i], label='Predicted', color='red', alpha=0.7)
        ax.set_title(f'{target} - Actual vs Predicted')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel(target)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(result_root_dir, 'predictions_vs_actual.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return metrics, y_pred, y_true

def save_predictions_to_json(y_true, y_pred, target_columns, save_path):
    """保存预测结果到JSON文件"""
    prediction_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'target_columns': target_columns,
        'actual_values': y_true,
        'predicted_values': y_pred,
        'metrics': {
            target_columns[0]: {
                'MSE': mean_squared_error(y_true[:, 0], y_pred[:, 0]),
                'RMSE': np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0])),
                'MAE': mean_absolute_error(y_true[:, 0], y_pred[:, 0]),
                'R2': r2_score(y_true[:, 0], y_pred[:, 0])
            },
            target_columns[1]: {
                'MSE': mean_squared_error(y_true[:, 1], y_pred[:, 1]),
                'RMSE': np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1])),
                'MAE': mean_absolute_error(y_true[:, 1], y_pred[:, 1]),
                'R2': r2_score(y_true[:, 1], y_pred[:, 1])
            }
        }
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(prediction_data, f, cls=NumpyEncoder, indent=4, ensure_ascii=False)

    print(f"\n💾 Predictions saved to: {save_path}")

# =========================
# Main Execution
# =========================
def main():
    """主执行函数"""
    print("="*80)
    print("🚀 Starting 10-minute Level LSTM Prediction for Core Indicators")
    print("="*80)

    # 1. Load data
    df = load_final_json_data(modified_code_output_dir)
    if df is None:
        print("❌ Data loading failed, exiting...")
        return

    # 2. Prepare features and targets
    feature_columns, target_columns = prepare_core_features_and_targets(df)
    if feature_columns is None or target_columns is None:
        print("❌ Feature/target preparation failed, exiting...")
        return

    # 3. Create sliding windows (核心修改：window_size=30)
    X, y, feature_scaler, target_scalers = create_core_indicators_sliding_window(
        df=df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        window_size=30  # 改为30个10分钟窗口
    )

    if X is None or y is None:
        print("❌ Sliding window creation failed, exiting...")
        return

    # 4. Train-validation-test split
    print("\n🔪 Splitting data into train/validation/test sets...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, shuffle=False)

    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")

    # 5. Train model
    input_dim = len(feature_columns)
    model, best_val_loss, train_losses, val_losses = train_dual_output_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        model_name="CoreIndicatorsModel"
    )

    # 6. Evaluate model
    print("\n📊 Evaluating model on test set...")
    metrics, y_pred, y_true = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        target_scalers=target_scalers,
        target_columns=target_columns
    )

    # 7. Save predictions
    json_save_path = os.path.join(JSON_OUTPUT_DIR, "core_indicators_predictions.json")
    save_predictions_to_json(y_true, y_pred, target_columns, json_save_path)

    # 8. Save model and scalers
    print("\n💾 Saving model and scalers...")
    model_save_path = os.path.join(result_root_dir, "core_indicators_model.pth")
    torch.save(model.state_dict(), model_save_path)

    scaler_save_path = os.path.join(result_root_dir, "scalers.pkl")
    with open(scaler_save_path, 'wb') as f:
        pickle.dump({
            'feature_scaler': feature_scaler,
            'target_scalers': target_scalers
        }, f)

    print(f"  - Model saved to: {model_save_path}")
    print(f"  - Scalers saved to: {scaler_save_path}")

    print("\n" + "="*80)
    print("🎉 Prediction completed successfully!")
    print(f"📊 Results summary:")
    print(f"  - Best validation loss: {best_val_loss:.4f}")
    print(f"  - Test R² (center shift): {metrics[target_columns[0]]['R2']:.4f}")
    print(f"  - Test R² (cluster radius): {metrics[target_columns[1]]['R2']:.4f}")
    print(f"📁 All results saved in: {result_root_dir}")
    print("="*80)

if __name__ == "__main__":
    main()