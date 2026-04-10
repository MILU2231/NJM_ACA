好，下面给你一份 **Step2.1 设计脚本/设计文档**。
这份不是代码，而是**代码实现前的正式设计说明**。你先看这份，确认后我再按这份输出 Step2.1 的修改代码。

---

# Step2.1 设计文档

## 名称

**事件段导出与 30s 标准样本生成**

---

# 1. Step2.1 的目标

Step2 目前已经能完成：

* 列车上下文识别
* 事件段检测
* 背景段检测
* 10 分钟窗口级上下文统计

例如你这一天已经得到：

* 事件段数量 `67`
* 背景段数量 `68`
* 超长事件候选数量 `16`
* 平均列车时间占比 `0.072014`
* 平均背景时间占比 `0.905104`。

但当前结果还**不适合直接喂后续模型**，因为还缺：

1. 单独导出的列车事件段
2. 单独导出的背景段
3. 每个段的单独可视化
4. 固定时长的标准化样本
5. 统一的样本级 JSON 格式

所以 Step2.1 的目标是：

## 目标 A：导出自然边界事件段

把 Step2 检测出的列车事件段和背景段单独保存。

## 目标 B：导出固定 30s 的模型样本

从自然边界事件段中进一步生成标准化样本：

* 列车 30s 样本
* 背景 30s 样本

## 目标 C：为后续模型训练和预测提供统一数据接口

让后续：

* 特征提取
* 分布半径/中心偏移
* SVDD
* 健康预测

都有统一输入。

---

# 2. Step2.1 在总流程中的位置

## Step1

原始日级 CSV → 秒级基础特征

## Step1.5

面向事件切分的秒级特征修正
输出 `refined.json`，其中秒级数据保存在 `second_level_features_refined` 下的列式数组中。

## Step2

列车上下文识别
输出：

* `events.json`
* `background.json`
* `window_context.json`
* `second_labels.json`
* `step2.png`
* `step2.md`。

## Step2.1

自然事件段导出 + 30s 标准样本生成

## Step3

特征提取 + 健康表示构建

## Step4

SVDD + 健康预测

---

# 3. Step2.1 的输入

Step2.1 的输入来自两部分。

## 3.1 必须输入：Step2 输出

### A. `events.json`

列车事件段清单

### B. `background.json`

背景段清单

### C. `second_labels.json`

秒级状态标签

### D. `window_context.json`

10 分钟窗口上下文，可作为辅助

---

## 3.2 建议输入：Step1.5 refined JSON

因为 Step2 输出里通常是“边界和摘要”，而真正做样本切片时，我们还需要回到秒级特征链。

所以 Step2.1 建议同时读取：

* `YYYYMMDD.refined.json`

从其中的：

* `second_level_features_refined`

取出原始秒级序列。你的 Step1.5 明确是按列式数组保存，例如：

* `timestamp`
* `rms`
* `peak_abs`
* `band_energy_total`
* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`
* `confidence_level_final`
* `long_missing_segment`
* `protection_belt`
* `is_imputed_final`。

---

# 4. Step2.1 的输出

Step2.1 需要新增两大类产物。

---

## 4.1 自然边界段导出结果

目录建议：

```text
YYYYMM/
    step2_1_segments/
        train/
            json/
            png/
        background/
            json/
            png/
```

### train/json

每个列车事件段一个 JSON

### train/png

每个列车事件段一张图

### background/json

每个背景段一个 JSON

### background/png

每个背景段一张图

---

## 4.2 固定 30s 标准样本结果

目录建议：

```text
YYYYMM/
    step2_1_model_slices/
        train_30s/
            json/
            png/
        background_30s/
            json/
            png/
```

### train_30s/json

每个列车 30s 样本一个 JSON

### train_30s/png

每个列车 30s 样本一张图

### background_30s/json

每个背景 30s 样本一个 JSON

### background_30s/png

每个背景 30s 样本一张图

---

## 4.3 日级说明文档

建议继续保存：

```text
YYYYMM/
    step2_1_md/
        YYYYMMDD.step2_1.md
```

用于总结这一天：

* 导出多少自然事件段
* 导出多少 30s 列车样本
* 导出多少 30s 背景样本
* 丢弃了多少过短背景段
* 超长事件处理情况如何

---

# 5. Step2.1 的核心设计原则

## 原则 1：保留两套结果

必须同时保留：

### 自然边界段

用于上下文分析、真实持续时间分析

### 固定 30s 样本

用于模型训练和后续标准化处理

---

## 原则 2：列车与背景分开导出

不能混在一起输出。
必须明确：

* `label = 1`：列车事件
* `label = 0`：背景样本

---

## 原则 3：固定时长样本优先选择 30s

第一版统一固定：

* `TARGET_SLICE_SEC = 30`

这样最利于后续特征提取和模型统一。

---

## 原则 4：保留样本来源映射关系

每个 30s 样本必须知道它来自哪个自然段。

例如：

* `source_event_id`
* `source_background_id`

---

## 原则 5：补边/填充必须显式记录

如果某个列车事件不足 30s，需要补边，必须记录：

* 是否补边
* 左右补了多少秒

不能“静默补”。

---

# 6. Step2.1 的处理流程

Step2.1 建议分成 8 个子步骤。

---

## Step2.1.1 读取输入文件

读取：

* `refined.json`
* `events.json`
* `background.json`

重建秒级 DataFrame。

### 必须重建的列

* `timestamp`
* `rms`
* `peak_abs`
* `band_energy_total`
* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`
* `confidence_level_final`
* `long_missing_segment`
* `protection_belt`
* `is_imputed_final`

---

## Step2.1.2 导出自然列车事件段

对 `events.json` 中每个事件段：

### 提取秒级片段

按事件段 `start_idx ~ end_idx` 从秒级 DataFrame 切出。

### 生成自然段 JSON

每个事件一个文件。

### 绘图

至少画 3 条曲线：

* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`

并标注：

* 起止时间
* 持续时间
* event_id
* 是否超长
* 可信度

---

## Step2.1.3 导出自然背景段

对 `background.json` 中每个背景段：

### 提取秒级片段

按背景段 `start_idx ~ end_idx` 切出。

### 生成自然段 JSON

每段一个文件。

### 绘图

同样至少画：

* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`

---

## Step2.1.4 列车事件 → 30s 标准样本

这是最关键的一步。

---

### A. 事件长度 < 30s

若自然事件段长度不足 30s：

#### 做法

以事件段中心或主峰中心为中心，构造 30s 窗口。

#### 不足部分补边

补边策略第一版建议：

* 边缘值重复填充
  或
* 反射填充

我建议第一版用**边缘重复填充**，更稳定更可解释。

#### 必须记录

* `is_padded = True`
* `pad_left_sec`
* `pad_right_sec`

---

### B. 事件长度 = 30s

直接作为标准样本导出。

---

### C. 事件长度 > 30s

若事件长于 30s：

#### 第一版规则

找该事件段内：

* `event_score` 最大秒
  或
* `rms_smooth_5s` 最大秒

以该点为中心切 30s。

#### 目的

提取“最代表列车核心响应”的样本。

---

### D. 超长事件

你的当前 Step2 结果里有 `16` 个超长事件候选。

#### 第一版处理

每个超长事件先只导出 1 个“主峰中心 30s 样本”。

#### 第二版再考虑

* 多峰拆分
* 一段导出多个样本

---

## Step2.1.5 背景段 → 30s 标准样本

背景样本不需要像列车那样围绕峰来切。

### 背景段长度 < 30s

直接丢弃，不生成样本。

### 背景段长度 >= 30s

按固定步长切。

---

### 背景切片步长建议

#### 第一版

* `BACKGROUND_STEP_SEC = 30`

也就是不重叠切片。

#### 后续可选

* 15s 半重叠

---

### 为什么第一版先不重叠

* 样本数量更可控
* 独立性更强
* 便于先检查质量

---

## Step2.1.6 生成标准样本 JSON

每个 30s 样本都要生成统一 JSON。

---

## Step2.1.7 生成标准样本 PNG

每个 30s 样本画一张图。

### 建议画法

上下 3 行：

1. `rms_smooth_5s`
2. `peak_abs_smooth_5s`
3. `band_energy_total_smooth_5s`

并在标题写清楚：

* 样本 ID
* label
* 来源 event/background
* 是否补边

---

## Step2.1.8 生成日级摘要 md

总结：

* 自然列车事件段数量
* 自然背景段数量
* 30s 列车样本数
* 30s 背景样本数
* 被丢弃背景段数
* 被补边列车样本数
* 超长事件处理数

---

# 7. 自然边界段 JSON 设计

---

## 7.1 列车自然段 JSON 字段

建议字段如下：

* `date`
* `sensor_id`
* `event_id`
* `segment_type = "train_event_segment"`
* `label = 1`
* `start_time`
* `end_time`
* `start_idx`
* `end_idx`
* `duration_sec`
* `is_oversized_candidate`
* `event_confidence_score`
* `contains_imputed_sec_ratio`
* `window_mode_hint`（可选）
* `sequence`：保存秒级序列

### `sequence` 下保存

* `timestamp`
* `rms`
* `peak_abs`
* `band_energy_total`
* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`
* `confidence_level_final`

---

## 7.2 背景自然段 JSON 字段

建议字段如下：

* `date`
* `sensor_id`
* `background_id`
* `segment_type = "background_segment"`
* `label = 0`
* `start_time`
* `end_time`
* `start_idx`
* `end_idx`
* `duration_sec`
* `background_confidence_score`
* `contains_imputed_sec_ratio`
* `sequence`

---

# 8. 30s 标准样本 JSON 设计

---

## 8.1 列车 30s 样本 JSON

建议字段：

* `date`
* `sensor_id`
* `sample_id`
* `slice_type = "train_30s"`
* `label = 1`
* `source_event_id`
* `source_event_duration_sec`
* `source_is_oversized_candidate`
* `slice_start_time`
* `slice_end_time`
* `slice_start_idx`
* `slice_end_idx`
* `duration_sec = 30`
* `is_padded`
* `pad_left_sec`
* `pad_right_sec`
* `anchor_type`
  可选值：

  * `event_center`
  * `peak_center`
* `sequence`

---

## 8.2 背景 30s 样本 JSON

建议字段：

* `date`
* `sensor_id`
* `sample_id`
* `slice_type = "background_30s"`
* `label = 0`
* `source_background_id`
* `source_background_duration_sec`
* `slice_start_time`
* `slice_end_time`
* `slice_start_idx`
* `slice_end_idx`
* `duration_sec = 30`
* `is_padded = False`
* `step_sec = 30`
* `sequence`

---

## 8.3 `sequence` 保存内容

两类样本统一保存：

* `timestamp`
* `rms`
* `peak_abs`
* `band_energy_total`
* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`
* `confidence_level_final`

---

# 9. 是否需要保存原始 50Hz 波形

## 当前建议

**Step2.1 第一版不保存原始 50Hz 波形。**

原因：

* 现在主链路基于秒级特征
* 后续特征提取和健康建模也优先围绕秒级特征
* 保存 50Hz 会让文件大很多

---

## 后续可选增强

如果你后面确实需要做更细粒度信号分析，可以在 Step2.2 再加：

* 回溯原始波形保存

但第一版不做。

---

# 10. 图片设计规范

---

## 10.1 自然段图片

### 列车事件图

建议文件名：

* `event_0001.train.segment.png`

### 背景段图

建议文件名：

* `background_0001.segment.png`

---

## 10.2 30s 样本图片

### 列车

* `train30s_0001.png`

### 背景

* `background30s_0001.png`

---

## 10.3 图片标题建议

### 自然列车段

`事件段 0001 | 列车事件 | 12.0s | high_conf=0.94`

### 列车 30s 样本

`列车30s样本 0001 | source_event=0001 | padded=True`

### 背景 30s 样本

`背景30s样本 0001 | source_background=0003`

---

# 11. 可调参数表

这些参数都应写在 `main()` 中。

## 长度相关

* `TARGET_SLICE_SEC = 30`
* `BACKGROUND_STEP_SEC = 30`

## 列车样本对齐策略

* `TRAIN_SLICE_ANCHOR = "peak_center"`
  可选：

  * `peak_center`
  * `event_center`

## 补边策略

* `PAD_MODE = "edge"`
  可选：

  * `edge`
  * `reflect`

## 背景过滤

* `MIN_BACKGROUND_SEG_SEC = 30`

## 是否导出自然段

* `EXPORT_NATURAL_SEGMENTS = True`

## 是否导出 30s 样本

* `EXPORT_STANDARD_SLICES = True`

## 是否绘图

* `SAVE_SEGMENT_PNG = True`
* `SAVE_SLICE_PNG = True`

---

# 12. 与后续步骤的接口关系

---

## 12.1 给 Step3 的输入

Step3 做特征提取时，应优先读取：

* `step2_1_model_slices/train_30s/json/*.json`
* `step2_1_model_slices/background_30s/json/*.json`

也就是说：

### Step3 的标准训练输入

不再直接来自 Step2 的事件清单，而来自 Step2.1 的标准化 30s 样本。

---

## 12.2 给健康检测与预测的输入

对于窗口级健康检测和预测：

* 仍主要使用 `window_context.json`
* 同时可以利用 Step2.1 导出的标准样本做更细粒度特征聚合

---

# 13. 验收标准

Step2.1 完成后，至少要满足：

## 验收 1

列车自然事件段和背景自然段都被单独保存。

## 验收 2

每个自然段都有对应图。

## 验收 3

30s 列车样本被成功导出。

## 验收 4

30s 背景样本被成功导出。

## 验收 5

每个 30s 样本都能追溯到来源自然段。

## 验收 6

补边信息被显式记录。

## 验收 7

日级 md 能总结当天样本产出情况。

---

# 14. 我建议的第一版实现边界

为了避免一下做太重，我建议 Step2.1 第一版：

## 必做

* 自然列车段导出
* 自然背景段导出
* 30s 列车样本导出
* 30s 背景样本导出
* 全部 json
* 全部 png
* 日级 md

## 暂不做

* 超长事件多峰拆分
* 相汇列车细分
* 原始 50Hz 波形回溯保存
* 数据增强
* 样本平衡策略

---

# 15. 一句话总结

**Step2.1 的本质不是重新做事件检测，而是在 Step2 已经得到的列车事件段与背景段基础上，进一步导出自然边界段和固定 30s 标准样本，形成后续模型训练和预测的统一输入。**

---

如果你认可这份设计，我下一步就按这份直接给你写 **Step2.1 的 Python 代码**。
