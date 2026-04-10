下面是我给你整理的 **新的 Step 2 详细规则设计文档**。
这份文档的定位是：

**作为 Step 2 代码实现前的正式规则说明书。**

它会明确：

* Step 2 的目标
* 输入依赖哪些 Step 1.5 结果
* 事件切分的总体逻辑
* 检测规则、屏蔽规则、阈值规则
* 输出结构
* 与后续 Step 3 / Step 4 的接口关系
* 需要保留的可调参数
* 第一版代码实现边界

你确认这份规则没问题后，我们就可以直接写 Step 2 代码。

---

# Step 2 详细规则设计文档

## 名称

**列车上下文识别与事件/背景分段构建**

---

# 1. Step 2 的目标

Step 2 不是做异常检测，也不是直接训练分类模型。

Step 2 的目标是：

1. 从连续秒级特征链中识别出**列车经过相关的事件段**
2. 从剩余高可信区域中识别出**背景振动段**
3. 生成后续健康检测与健康预测所需的**运营上下文信息**
4. 为 Step 3 和 Step 4 提供统一、可追踪、可解释的中间数据产品

---

# 2. Step 2 在整体流程中的位置

整体链路现在应理解为：

## Step 1

原始日级 CSV 读取与秒级基础特征构建

## Step 1.5

面向事件切分的秒级特征修正
输出包括：

* 平滑后的 RMS / Peak / Energy
* 最终可信度等级
* 长缺失段与保护带标记
  并且已经给出直接使用建议：后续事件切分优先使用
  `rms_smooth_5s`、`peak_abs_smooth_5s`、`band_energy_total_smooth_5s`、`confidence_level_final`、`long_missing_segment`、`protection_belt`。

## Step 2

列车上下文识别与事件/背景分段构建

## Step 3

10 分钟窗口级健康表示构建
输出：

* 分布半径
* 中心偏移距离
* 标准化健康分数

## Step 4

SVDD 建模与健康预测
输出：

* SVDD 健康指数
* 未来窗口级健康趋势预测

---

# 3. Step 2 的核心定位

Step 2 只回答下面这些问题：

* 这一秒是否处于列车主导振动状态
* 哪些连续秒可以合并成列车事件段
* 哪些时间段是可信背景段
* 某个 10 分钟窗口内列车影响占比多大
* 当前窗口更像列车主导、背景主导还是混合窗口

Step 2 **不回答**：

* 是否异常
* 是否损伤
* 是否传感器失效
* 某次列车事件是否“有害”

这些属于后面的健康检测层。

---

# 4. Step 2 的输入

Step 2 输入来自 Step 1.5 的 refined 日级 JSON。

## 4.1 必须使用的输入字段

根据 Step 1.5 输出说明，后续事件切分建议优先使用以下字段：

* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`
* `confidence_level_final`
* `long_missing_segment`
* `protection_belt`。

因此 Step 2 第一版的主输入字段固定为：

### 时间字段

* `timestamp`

### 主检测特征

* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`

### 可用性与屏蔽字段

* `confidence_level_final`
* `long_missing_segment`
* `protection_belt`
* `is_imputed_final`

### 辅助原始秒级特征

* `rms`
* `peak_abs`
* `band_energy_total`

---

# 5. Step 2 的总体策略

Step 2 采用：

## “三特征平滑检测 + 屏蔽约束 + 迟滞阈值 + 片段合并 + 背景反推”

其思路借鉴了旧脚本中的这些方法：

* 包络化检测思想
* 进入阈值 / 退出阈值分离
* 恢复保持时间
* 片段合并
* 背景安全边界。

但不沿用旧脚本中“固定目标事件数”的策略。旧脚本曾通过阈值搜索把事件数逼近某个目标区间，例如 `EXPECTED_TARGET_EVENTS = 120`。
新的 Step 2 不使用这个约束。

---

# 6. Step 2 的处理流程

Step 2 处理流程建议拆成 9 个子步骤。

---

## Step 2.1 读取 refined JSON 并重建秒级 DataFrame

### 目标

把 Step 1.5 的 JSON 转成统一的秒级表结构。

### 结果

得到一天 86400 行的秒级 DataFrame，每行至少有：

* timestamp
* rms_smooth_5s
* peak_abs_smooth_5s
* band_energy_total_smooth_5s
* confidence_level_final
* long_missing_segment
* protection_belt
* is_imputed_final

---

## Step 2.2 构建“不可判区域”掩码

### 目标

先明确哪些秒不能用于列车事件判断。

### 规则

下列秒一律不参与列车事件检测：

1. `confidence_level_final == unusable`
2. `long_missing_segment == True`
3. `protection_belt == True`

依据 Step 1.5 文档，本日可用于事件切分，但应屏蔽长缺失段及其保护带。

### 输出

新增布尔字段：

* `mask_unusable_for_event`

---

## Step 2.3 构建事件强度分数 `event_score`

### 目标

把三条平滑特征融合成一条统一的列车事件强度曲线。

### 为什么要做

如果只看单一特征，容易出现：

* RMS 高但峰值不明显
* 峰值高但只是尖刺
* 能量高但整体抬升不明显

三者融合更稳。

### 建议方法

先对三条特征分别做鲁棒标准化，再加权融合。

#### 子步骤 A：鲁棒归一化

对以下三列分别做日内鲁棒标准化：

* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`

推荐公式：

[
z = \frac{x - \text{median}(x)}{\text{MAD}(x) + \epsilon}
]

这里仅在“可判区域”上计算 median 和 MAD。

#### 子步骤 B：加权融合

构造：

[
event_score = w_1 \cdot z_{rms} + w_2 \cdot z_{peak} + w_3 \cdot z_{energy}
]

### 第一版建议权重

* `w_rms = 0.45`
* `w_peak = 0.25`
* `w_energy = 0.30`

### 原因

* RMS 最稳定，权重最高
* Energy 对整体活动增强有帮助
* Peak 用来补充尖峰冲击信息，但不能权重过高

### 输出

新增列：

* `z_rms_smooth_5s`
* `z_peak_abs_smooth_5s`
* `z_band_energy_total_smooth_5s`
* `event_score`

---

## Step 2.4 计算自适应进入阈值与退出阈值

### 目标

把 `event_score` 转成候选事件段。

### 设计原则

借鉴旧脚本的“进入阈值 / 退出阈值分离”机制。旧脚本通过 `enter_thr` 与 `exit_thr = enter_thr * EXIT_RATIO` 实现迟滞切分。
新的 Step 2 也保留迟滞机制，但不采用“逼近目标事件数”的阈值搜索。

---

### 进入阈值 `enter_thr`

第一版推荐使用以下方式：

#### 方案

在“可判区域”的 `event_score` 上取较高分位数作为进入阈值。

[
enter_thr = Q_{97.5%}(event_score)
]

同时设置下限保护：

[
enter_thr = \max(enter_thr,\ \text{median}(event_score)+2.5 \cdot MAD(event_score))
]

### 目的

避免某些特别平静的天阈值过低，导致误检很多小波动。

---

### 退出阈值 `exit_thr`

采用固定比例：

[
exit_thr = 0.60 \cdot enter_thr
]

这和旧脚本 `EXIT_RATIO = 0.6` 的思想一致。

---

### 恢复保持秒数 `recover_hold_sec`

推荐：

* `recover_hold_sec = 3`

解释：
只有当连续 3 秒低于退出阈值，才真正结束一个事件。

### 原因

防止一个列车事件内部的小回落被切成多个碎片。

---

## Step 2.5 逐秒扫描生成候选事件段

### 目标

依据 `event_score` 与迟滞阈值，生成原始候选事件段。

### 扫描逻辑

#### 状态 1：当前不在事件中

当某秒满足：

* 不在 `mask_unusable_for_event`
* `event_score > enter_thr`

则启动一个新事件段。

#### 状态 2：当前在事件中

只要 `event_score >= exit_thr`，继续保持事件状态。

#### 状态 3：退出判断

若 `event_score < exit_thr` 连续达到 `recover_hold_sec` 秒，则结束该事件段。

### 输出

初始候选事件列表：

* `raw_event_segments = [(start_idx, end_idx), ...]`

---

## Step 2.6 事件段过滤与合并

### 目标

去掉明显不合理的候选片段，并合并碎片事件。

### 过滤规则

#### 最短事件时长

* `min_event_sec = 8`

短于 8 秒的候选片段先视为噪声扰动，不作为列车事件。

#### 最大事件时长

* `max_event_sec = 120`

超过 120 秒的片段先不直接丢弃，而是标记为：

* `oversized_event_candidate`

第一版中暂不拆分，只保留并标记。

---

### 合并规则

若两个事件片段之间间隔小于等于：

* `merge_gap_sec = 5`

则合并。

这个思路借鉴旧脚本中 `merge_segments` 和 `MERGE_GAP_SEC` 的做法。

### 输出

得到合并后的事件列表：

* `merged_event_segments`

---

## Step 2.7 构建背景段

### 目标

从非事件区间中构建可用于后续健康建模的背景段。

### 背景候选定义

背景候选秒需满足：

1. 不在事件段内部
2. 不在 `mask_unusable_for_event`
3. `confidence_level_final` 不是 `unusable`

---

### 背景安全边界

对每个事件段，前后各留安全边界，不把紧贴事件边缘的区域算背景。

推荐：

* `background_margin_sec = 10`

这和旧脚本里 `NORMAL_MARGIN_SEC = 10` 的思想一致。

---

### 背景段最短时长

* `min_background_sec = 20`

短于 20 秒的背景段不单独保留。

### 输出

背景段列表：

* `background_segments`

---

## Step 2.8 给每个事件段和背景段计算摘要特征

### 目标

让 Step 2 输出不仅有边界，还有结构化摘要。

---

### 事件段摘要字段

每个事件段输出：

* `event_id`
* `start_time`
* `end_time`
* `duration_sec`
* `start_idx`
* `end_idx`
* `max_event_score`
* `mean_event_score`
* `max_rms_smooth_5s`
* `mean_rms_smooth_5s`
* `max_peak_abs_smooth_5s`
* `mean_peak_abs_smooth_5s`
* `max_band_energy_total_smooth_5s`
* `mean_band_energy_total_smooth_5s`
* `event_confidence_score`
* `contains_imputed_sec_ratio`
* `is_oversized_candidate`

---

### 背景段摘要字段

每个背景段输出：

* `background_id`
* `start_time`
* `end_time`
* `duration_sec`
* `start_idx`
* `end_idx`
* `mean_rms_smooth_5s`
* `std_rms_smooth_5s`
* `mean_event_score`
* `background_confidence_score`
* `contains_imputed_sec_ratio`

---

## Step 2.9 生成 10 分钟窗口级上下文表

### 目标

把秒级事件/背景结果聚合成后续健康建模直接可用的窗口级上下文特征。

这是 Step 2 最重要的输出之一。

### 每个 10 分钟窗口建议输出的字段

#### 基础上下文字段

* `window_start_time`
* `window_end_time`
* `has_train`
* `train_event_count`
* `train_time_ratio`
* `background_time_ratio`
* `masked_time_ratio`

#### 事件强度聚合字段

* `train_event_score_mean`
* `train_event_score_max`
* `train_rms_mean`
* `train_peak_mean`
* `train_energy_mean`

#### 背景聚合字段

* `background_rms_mean`
* `background_peak_mean`
* `background_energy_mean`

#### 运行模式字段

* `window_mode`

### `window_mode` 的定义

推荐取值：

* `train_dominant`
* `background_dominant`
* `mixed`
* `unusable`

#### 规则建议

* 若 `masked_time_ratio > 0.3` → `unusable`
* 否则若 `train_time_ratio >= 0.4` → `train_dominant`
* 否则若 `train_time_ratio <= 0.1` → `background_dominant`
* 其他 → `mixed`

---

# 7. 可信度与权重规则

Step 2 不应该把所有秒等价看待。

## 7.1 秒级使用规则

### `high`

* 可直接用于事件判断
* 权重 1.0

### `medium`

* 可参与事件判断
* 权重 0.7

### `low`

* 不作为触发事件的主依据
* 仅用于事件段内部统计
* 权重 0.3

### `unusable`

* 直接屏蔽
* 权重 0

这和 Step 1.5 的使用建议一致：主用 `high` 与 `medium`，谨慎参考 `low`，屏蔽 `unusable`。

---

## 7.2 事件置信分数

每个事件段计算：

[
event_confidence_score = 1 - \frac{\text{low秒数} + \text{unusable秒数}}{\text{事件总秒数}}
]

如果事件段内部有很多低可信秒，则后续可降权使用。

---

# 8. Step 2 的输出文件设计

Step 2 建议延续 Step 1 的输出风格。

## 输出目录结构建议

按月份、按天保存：

```text
202405/
    ├── step2_json/
    │      ├── 20240531.events.json
    │      ├── 20240531.background.json
    │      ├── 20240531.window_context.json
    │      └── 20240531.second_labels.json
    ├── step2_png/
    │      └── 20240531.step2.png
    └── step2_md/
           └── 20240531.step2.md
```

---

## 8.1 `events.json`

保存事件段清单。

---

## 8.2 `background.json`

保存背景段清单。

---

## 8.3 `window_context.json`

保存 10 分钟窗口级上下文表。

---

## 8.4 `second_labels.json`

保存秒级状态标签。

每秒一个字段：

* `state_label`

推荐取值：

* `train_event`
* `background`
* `masked_unusable`
* `neutral`

这里的 `neutral` 指：

* 非事件
* 非背景
* 但因为靠近边界或其它原因，不进入背景段

---

## 8.5 `step2.png`

建议一张总图至少画 5 个子图：

1. `rms_smooth_5s`
2. `peak_abs_smooth_5s`
3. `band_energy_total_smooth_5s`
4. `event_score` + `enter_thr` + `exit_thr`
5. 秒级状态标签 / 事件段与背景段标记

---

## 8.6 `step2.md`

用于总结该日 Step 2 结果，至少包括：

* 总事件数
* 总背景段数
* 列车时间占比
* 背景时间占比
* 不可用时间占比
* 超长事件数量
* 是否适合进入 Step 3

---

# 9. Step 2 与后续步骤的接口

---

## 9.1 给 Step 3 的输入

Step 3 做健康检测时，不再从原始波形直接识别列车，而直接吃 Step 2 的窗口级上下文结果。

主要使用：

* `window_context.json`

最关键字段：

* `has_train`
* `train_event_count`
* `train_time_ratio`
* `window_mode`
* 背景聚合特征

---

## 9.2 给 Step 4 的输入

Step 4 做健康预测时，直接使用窗口级上下文序列作为外部上下文变量。

例如在预测：

* 半径
* 中心偏移距离
* SVDD 健康指数

时可把以下字段作为已知历史上下文：

* `train_time_ratio`
* `train_event_count`
* `window_mode`

---

# 10. Step 2 第一版不做什么

为了尽快落地，第一版明确不做以下内容：

1. 不做相汇列车分离
   旧脚本里有 `is_convergence_event` 和双峰分裂逻辑，但第一版不纳入。

2. 不做固定 30 秒 / 60 秒样本标准化切片
   那是旧脚本为分类训练准备的。

3. 不做事件级标签训练集导出
   Step 2 只输出事件/背景结构化结果。

4. 不做重型波形降噪
   Step 2 只利用已有平滑特征做上下文识别。

5. 不做“目标事件数逼近”
   不使用旧脚本中 `EXPECTED_TARGET_EVENTS` 那套机制。

---

# 11. Step 2 的可调参数表

下面这些参数应全部写到 `main()` 中，方便后面调试。

## 屏蔽相关

* `USE_MASK_UNUSABLE = True`
* `USE_LONG_MISSING_MASK = True`
* `USE_PROTECTION_BELT_MASK = True`

## 事件得分权重

* `W_RMS = 0.45`
* `W_PEAK = 0.25`
* `W_ENERGY = 0.30`

## 阈值相关

* `ENTER_PERCENTILE = 97.5`
* `ENTER_MAD_MULT = 2.5`
* `EXIT_RATIO = 0.60`
* `RECOVER_HOLD_SEC = 3`

## 事件段规则

* `MIN_EVENT_SEC = 8`
* `MAX_EVENT_SEC = 120`
* `MERGE_GAP_SEC = 5`

## 背景段规则

* `BACKGROUND_MARGIN_SEC = 10`
* `MIN_BACKGROUND_SEC = 20`

## 窗口模式规则

* `UNUSABLE_MASK_RATIO_THR = 0.30`
* `TRAIN_DOMINANT_RATIO_THR = 0.40`
* `BACKGROUND_DOMINANT_RATIO_THR = 0.10`

---

# 12. Step 2 的验收标准

Step 2 完成后，至少应满足下面这些验收条件。

## 验收 1

长缺失段及保护带不被误识别成列车事件。

## 验收 2

明显高响应区能被识别成连续事件段，而不是很多碎片。

## 验收 3

背景段不会紧贴事件边界。

## 验收 4

每天都能输出：

* 事件清单
* 背景段清单
* 10 分钟上下文表
* 秒级标签表

## 验收 5

Step 2 的结果能直接支持 Step 3 构建窗口级健康特征。

---

# 13. 第一版代码实现建议顺序

我建议代码实现严格按下面顺序写：

1. 读取 refined json
2. 重建秒级 DataFrame
3. 构建 `mask_unusable_for_event`
4. 计算三特征鲁棒归一化
5. 计算 `event_score`
6. 计算 `enter_thr` / `exit_thr`
7. 生成原始候选事件段
8. 过滤并合并事件段
9. 构建背景段
10. 生成秒级状态标签
11. 聚合 10 分钟上下文表
12. 输出 json/png/md

---

# 14. 我给你的最终建议版本

如果现在要正式进入代码阶段，我建议用下面这个版本作为 Step 2 第一版规范：

## 输入

Step 1.5 refined json

## 主检测特征

* `rms_smooth_5s`
* `peak_abs_smooth_5s`
* `band_energy_total_smooth_5s`

## 屏蔽规则

屏蔽：

* `unusable`
* `long_missing_segment`
* `protection_belt`

## 检测逻辑

* 三特征融合 `event_score`
* 自适应进入阈值
* 迟滞退出阈值
* 最短持续时间
* 片段合并

## 输出

* 事件段
* 背景段
* 秒级标签
* 10 分钟上下文表
* 日级可视化图
* 日级说明 md

---

一句话总结：

**新的 Step 2 不再是“切训练样本”，而是“从 refined 秒级特征链中识别列车上下文、构造事件段与背景段，并生成 10 分钟窗口级运营上下文表，为后续健康检测和健康预测服务”。**

你确认这版规则后，我下一步就直接给你写 **Step 2 的 Python 代码**。
