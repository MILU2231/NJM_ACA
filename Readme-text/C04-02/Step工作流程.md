各层核心职责与数据流转详解
Step0 | 数据接收与标准化
项目	内容
输入	原始加速度CSV（每行1秒 × 50采样点/sensor_id/timestamp）
输出	144个10分钟窗口JSON，含 data_matrix(600×50) + missing_mask + zero_mask
核心操作	构建全天24h时间轴对齐、区分"真实0值"与"缺失补位"
为下一步准备	结构化窗口数据，时间轴完全连续，掩码清晰标记数据可靠性
Step1 | 质量控制预处理 (QC)
项目	内容
输入	Step0窗口JSON
输出	*_qc.json（在原始JSON基础上追加5大模块QC字段）
核心操作	信号质量检测(缺失/异常值/漂移) + 设备故障检测(饱和/恒值/低能量) + 切片异常画像(趋势/离散/混沌) + 列车周期模式识别 + 数据清洗生成 data_sequence_cleaned
为下一步准备	清洗后的信号序列 + 完整的QC元数据字段
Step2 | 窗口级特征提取
项目	内容
输入	Step1的 *_qc.json
输出	all_window_features.csv（每行=1个窗口，90+字段）
核心操作	从 data_sequence_cleaned 提取 16个时域特征 + 13个频域特征（rFFT）；继承所有Step1 QC字段；综合评定 data_anomaly_type；生成4个有效性标记；补全每日144行
为下一步准备	完整特征表，含特征值 + QC继承 + 有效性标记 → Step3用有效性标记筛选
Step3 | 健康候选日选择
项目	内容
输入	Step2的 all_window_features.csv
输出	healthy_windows.csv + healthy_dates.csv + modeling_feature_columns.json
核心操作	两级策略：天级保留（19项硬剔除规则）+ 窗口级过滤（仅保留valid_for_feature/baseline/svdd=true的窗口）；异常值默认仅警告不剔除（避免列车冲击误判）
为下一步准备	仅健康日的有效窗口子集 + 29个建模特征列清单 → Step4用于拟合scaler和基线中心
Step4 | 基线中心与正常半径
项目	内容
输入	Step3健康窗口 + Step2全量表
输出	scaler.pkl + baseline_center.npy + all_window_center_distances.csv + center_radius_thresholds.json
核心操作	在健康窗口上拟合 RobustScaler；计算基线中心 C_health（median）；基于健康窗口中心距离的分位数确定半径阈值(q95/q99)；对Step2全量窗口计算中心距离和状态
为下一步准备	scaler + 标准化健康特征(npy) + 全窗口中心距离 → Step5用标准化健康特征训练SVDD；Step7-2用标准化特征和基线中心计算 center_shift_distance
Step5 | 标准SVDD健康超球体训练
项目	内容
输入	Step4的 scaler.pkl + standardized_healthy_features.npy + feature_columns.json
输出	all_window_standard_svdd_scores.csv（含三个核心指标）
核心操作	求解SVDD对偶问题；计算超球体半径R；对所有窗口计算 svdd_distance、svdd_distance_ratio；结合局部邻域计算 local_distribution_radius
为下一步准备	三维健康指标表 → Step7三个子系统共享同一张基础表
Step7 分发 | 三个独立预测子系统（共享Step5输入表）
关键设计： Step7-1/7-2/7-3是三个完全独立的脚本，各自从Step5的 all_window_standard_svdd_scores.csv 读取数据，各自构建48窗口(8小时)序列，各自训练独立的模型，输出各自的预测结果和残差阈值。Step6（BiLSTM三目标数据集构建）是另一个版本的统一方案。

Step7-1: local_distribution_radius 预测
项目	内容
目标	预测当前窗口的局部分布半径（当前领域48窗口的分布宽度）
模型	BiLSTM (hidden=96, layers=2, dropout=0.20)
目标构造	优先使用Step5表中的 local_distribution_radius；若不存在则基于 center_distance 用滚动 median + k·1.4826·MAD 计算
训练策略	SmoothL1Loss + AdamW + ReduceLROnPlateau；仅用正常目标样本训练（center_status=normal 且 svdd边界=inside）
输出	每个split的预测表 + 四级残差异常等级（normal/mild/moderate/high）
Step7-2: center_shift_distance 预测
项目	内容
目标	预测当前窗口的局域中心偏移距离（当前48窗口的局域中心到全局健康基线中心的距离）
模型	CenterShiftBiLSTM (hidden=96, layers=2)
目标构造	若Step5表中没有，则从Step4标准化特征计算：local_center = median(z[t-47:t]) → ‖local_center - C_health‖；需要加载Step4的 scaler.pkl + baseline_center.npy + feature_columns.json
V4升级	添加无信息泄漏的时序特征（lag/rolling/趋势）;时间泛化诊断（对比各时期的R²下降）; Baseline对比表
关键意义	衡量最近的局部特征分布中心是否偏离了健康基线中心，是结构性偏移的敏感指标
Step7-3: svdd_distance 趋势预测 v4.2
项目	内容
目标	预测当前窗口的SVDD距离因果平滑趋势（不追原始尖峰）
模型	TCN（默认，3层空洞卷积[48,48,48]）或 BiLSTM
v4.2核心改进	严格因果平滑(EMA+滚动中位数，无未来信息泄漏)；归一化增量目标变换（缓解分布漂移）；组合损失(55%增量+45%重构值)；EMA模型评估 + LateBest检查点选择；尖峰/边界样本降权训练；学习率预热
预测源选择	在验证集上对比模型 vs naive_last vs rolling6/12/24 vs ensemble，按RMSE自动选择最佳预测源
输出	残差四级阈值(基于MADσ) + SVDD边界诊断(实际越界/预测越界交叉矩阵) + 为Step8预留的完整残差和边界字段
Step6 与 Step7 的关系
Step6 (step6_bilstm_databuild_three_targets.py) 是另一个版本的统一三目标数据集构建器：

从Step5表构建48窗口序列 → 输出 (样本数, 48, 特征数) → (样本数, 3) 的三维数组
只构建数据集，不训练模型
同时准备 local_distribution_radius、center_distance、svdd_distance_ratio 三个目标
而 Step7-1/7-2/7-3 各自独立读取Step5表、各自构建序列、各自训练模型
两种方案是并行的——Step6适合需要联合预测三目标的场景，Step7各子系统适合独立调优和部署
最终融合点：Step8（架构预留）
Step8将融合以下四个维度的异常信号：

维度	来源	含义
中心距离	Step4	当前窗口特征到全局健康基线的距离（静态基准）
SVDD距离	Step5	当前窗口到健康超球体边界的距离（非线性边界）
预测残差×3	Step7-1/2/3	时序模型预测值与实际值的偏差（动态趋势）
边界诊断	Step7-3	原始SVDD尖峰与趋势预测的交叉关系
融合策略（待实现）：多维度投票/加权/级联 → 输出结构异常的综合置信度和异常等级。
