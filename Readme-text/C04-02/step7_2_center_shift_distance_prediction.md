Step7-2 现在做的事情可以概括成一句话：

**用过去 8 小时的窗口特征序列，预测当前 10 分钟窗口的 `center_shift_distance`，并用“实际值 - 预测值”的残差，给 Step8 判断健康中心漂移异常提供依据。**

---

## 1. Step7-2 的输入是什么

Step7-2 主要读取 Step5 的窗口级结果表：

```text
step5_standard_svdd_results/
└── 02_tables/
    └── window_level/
        └── all_window_standard_svdd_scores.csv
```

同时，如果 Step5 表里没有现成的 `center_shift_distance`，脚本会读取 Step4 的结果来补算：

```text
step4_baseline_center_radius_results/
├── scaler.pkl
├── feature_columns.json
└── baseline_center.npy
```

你这次运行时就触发了补算逻辑，因为原始 Step5 表里没有 `center_shift_distance`。脚本用 Step4 的标准化器、健康中心和 29 个建模特征，重新计算每个窗口的局部中心偏移距离。

---

## 2. Step7-2 预测的目标是什么

目标值是：

```text
center_shift_distance_t
```

它表示：

```text
当前局部时间段的特征中心
相对健康基准中心的偏移距离
```

也就是说，它不是看单个窗口离健康中心多远，而是看一段时间的整体状态中心有没有发生漂移。

它主要用于发现：

```text
缓慢劣化
整体状态漂移
长期趋势异常
```

---

## 3. 每个样本是怎么构造的

每个训练样本使用过去 48 个窗口：

```text
t-48, t-47, ..., t-1
```

预测当前窗口：

```text
t
```

因为每个窗口是 10 分钟，所以 48 个窗口就是：

```text
48 × 10 分钟 = 8 小时
```

所以 Step7-2 的监督学习形式是：

```text
X = 过去 8 小时的历史序列
y = 当前窗口的 center_shift_distance
```

---

## 4. 输入特征包括哪些

v4 版本的输入特征分成几类。

第一类是 Step2 的 29 个振动特征：

```text
时域特征
频域特征
```

第二类是 Step4/Step5 产生的健康空间距离特征：

```text
center_distance
svdd_distance
svdd_distance_ratio
svdd_boundary_margin
历史 center_shift_distance
```

第三类是质量控制特征：

```text
missing_ratio
max_missing_span
drift_score
outlier_ratio
valid_for_feature_flag
各种 missing mask
quality_status_code
data_anomaly_type_code
```

第四类是时间编码：

```text
window_sin
window_cos
hour_sin
hour_cos
```

第五类是 v4 新增的时间泛化特征：

```text
center_shift_lag_1
center_shift_lag_3
center_shift_lag_6
center_shift_lag_12
center_shift_lag_24
center_shift_lag_48

center_shift_delta_1
center_shift_delta_3
center_shift_delta_6
center_shift_delta_12
center_shift_delta_24
center_shift_delta_48

center_shift_rolling_mean_6 / 12 / 24
center_shift_rolling_median_6 / 12 / 24
center_shift_rolling_std_6 / 12 / 24

center_shift_trend_6 / 12 / 24
```

这些 v4 特征的目的，是让模型更容易捕捉“中心偏移距离”的短期惯性、局部趋势和变化速度。

---

## 5. Step7-2 训练了什么模型

Step7-2 训练的是一个单目标 BiLSTM 回归模型：

```text
输入：过去 48 个窗口的多维特征序列
输出：当前窗口的 center_shift_distance 预测值
```

模型不是分类器，不直接判断异常。

它输出的是一个连续预测值：

```text
center_shift_distance_pred
```

然后脚本会和实际值比较：

```text
center_shift_distance_actual
center_shift_residual = |actual - pred|
```

后续 Step8 才根据残差和健康阈值做异常融合判断。

---

## 6. Step7-2 为什么只用正常目标训练

当前设置是：

```text
train_only_normal_targets = True
```

这意味着模型主要学习“正常状态下，中心偏移距离应该如何随时间变化”。

这样做的目的很重要：

```text
不要让模型把异常模式也学进去。
```

如果模型把异常也学会了，那么异常发生时预测值也会跟着异常走，残差反而不明显，不利于 Step8 发现异常。

所以 Step7-2 的逻辑是：

```text
正常状态下应该预测得准；
异常状态下预测不准，残差变大；
Step8 根据残差和阈值判断中心漂移异常。
```

---

## 7. Step7-2 输出了什么

主要输出目录是：

```text
step7_2_center_shift_distance_prediction_results/
```

里面包括：

```text
00_config/     配置参数
01_datasets/   构造后的训练数据
02_models/     best 模型和 last 模型
03_tables/     预测结果表
04_figures/    训练曲线和预测效果图
05_reports/    Markdown 报告
```

最关键的预测表是：

```text
03_tables/all_sequence_predictions.csv
```

里面会有：

```text
center_shift_distance_actual
center_shift_distance_pred
center_shift_distance_residual
center_shift_prediction_threshold
center_shift_prediction_anomaly_flag
split
device_id
date
window_index
start_time
```

这些字段会被 Step8 使用。

---

## 8. v4 相比前面版本改进了什么

v4 最大的变化不是简单调参数，而是增加了“时间泛化能力”。

它新增了三类东西：

### 第一，历史趋势特征

帮助模型知道：

```text
center_shift_distance 最近是上升、下降，还是稳定。
```

### 第二，基线模型对照

报告里会比较：

```text
BiLSTM
naive_last
rolling6
rolling12
rolling24
```

其中：

```text
naive_last = 直接用上一个窗口的 center_shift_distance 当预测
rolling6 = 用过去 6 个窗口均值当预测
```

这可以判断 BiLSTM 是否真的比简单时间延续模型更有价值。

### 第三，更多残差阈值来源

v4 不只给训练集阈值，还给：

```text
train 阈值
validation 阈值
train_validation_combined 阈值
```

这样 Step8 可以避免只用训练残差阈值导致误报。

---

## 9. 你现在这个 v4 模型结果说明什么

现在 v4 的整体表现是比较好的。

关键指标：

```text
train R² = 0.991819
validation R² = 0.954791
test R² = 0.952632
```

训练集、验证集、测试集都很高，而且测试集没有像 v3 那样明显崩掉。

泛化差距：

```text
train-test R² gap = 0.039186
test/train RMSE ratio = 1.963933
```

这比 v3 的测试集表现明显好很多。v3 的 test R² 只有 `0.690656`，而 v4 提升到 `0.952632`。

所以现在可以判断：

```text
v4 已经基本解决了 v3 的测试集泛化问题。
```

---

## 10. 但还有一个重要发现

虽然 BiLSTM 的 test R² 很高：

```text
BiLSTM test R² = 0.952632
```

但是 `naive_last` 更强：

```text
naive_last test R² = 0.994904
```

也就是说，测试集上直接用上一个窗口的 `center_shift_distance` 预测当前窗口，比 BiLSTM 还准。

这说明 `center_shift_distance` 是一个非常平滑、强惯性的指标。

它的变化通常很慢，所以：

```text
上一个窗口的值
本身就是非常强的预测器。
```

这不是坏事，反而说明这个指标非常适合用来发现“偏离惯性趋势”的异常。

---

## 11. 最终结论

Step7-2 现在的作用是：

```text
建立 center_shift_distance 的时间序列预测模型，
判断当前中心偏移是否符合过去 8 小时的演化趋势。
```

它不负责计算健康中心，不负责训练 SVDD，也不直接给最终异常结论。

它负责输出：

```text
center_shift_distance_actual
center_shift_distance_pred
center_shift_residual
center_shift_residual_threshold
center_shift_prediction_anomaly_flag
```

然后 Step8 再结合：

```text
center_shift_distance 是否超过健康阈值
center_shift_residual 是否超过预测残差阈值
radius_distance 是否异常
svdd_distance 是否异常
数据质量是否异常
```

最终判断：

```text
健康中心漂移异常
结构响应异常
数据质量异常
或正常
```

我的建议是：**Step7-2 v4 可以作为当前候选正式版本进入 Step8，但 Step8 里要同时保留 naive_last baseline 的残差作为辅助参考。**
