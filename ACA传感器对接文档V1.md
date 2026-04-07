# 加速度传感器健康状态 & 下一10分钟预测结果推送接口文档

**接口提供方**：林同棪公司  
**接口调用方**：ACA 加速度传感器健康状态监测系统  
**通信方式**：HTTP POST (JSON格式)

## 1. 推送地址

```
POST http://127.0.0.1:38061/cqgd/api/sensor-aca/receive
```

## 2. 推送格式

- Content-Type: `application/json`
- 编码: UTF-8

## 3. 推送时机

每10分钟推送一次。

预测目标：基于最近30个10分钟窗口的历史数据，预测下一个10分钟窗口的中心偏移距离和分布半径。

## 4. 数据结构

```json
{
  "sensors": [
    {
      "sensor_id": "NJM-ACA-C13-02",
      "sensor_name": "下游侧NMC27索",
      "trigger_timestamp": "2026-02-03 11:20:00",
      "prediction_target_timestamp": "2026-02-03 11:30:00",
      "health_level": "正常",
      "health_state": 0,
      "health_message": "预测值在正常范围内",
      "alert_type": "NORMAL",
      "current_rms": 0.1245,
      "current_variance": 0.01582,
      "current_frequency_center": 12.45,
      "predicted_center_shift": 0.872,
      "predicted_cluster_radius": 2.341,
      "health_thresholds": {
        "in_range": true,
        "exceeded_percentage": 0.0,
        "historical_min_center_shift": 0.12,
        "historical_max_center_shift": 3.45,
        "yellow_upper": 1.80,
        "yellow_lower": 0.00,
        "orange_upper": 3.20,
        "orange_lower": 1.81,
        "red_upper": 5.00,
        "red_lower": 3.21
      }
    }
  ]
}
```

## 5. 字段说明

### 顶层字段

| 字段    | 类型  | 必填 | 说明               |
| ------- | ----- | ---- | ------------------ |
| sensors | array | 是   | 传感器预测结果列表 |

### sensors 数组内字段

| 字段                        | 类型    | 必填 | 说明                                                     |
| --------------------------- | ------- | ---- | -------------------------------------------------------- |
| sensor_id                   | string  | 是   | 传感器唯一标识                                           |
| sensor_name                 | string  | 是   | 传感器名称                                               |
| trigger_timestamp           | string  | 是   | 本次推送触发时间（格式：YYYY-MM-DD HH:MM:SS）            |
| prediction_target_timestamp | string  | 是   | 预测的目标时间点（本次触发时间 + 10分钟）                |
| health_level                | string  | 是   | 健康状态文字：正常 / 轻微异常 / 中级异常 / 高危异常      |
| health_state                | integer | 是   | 健康状态编码：0=正常, 1=轻微异常, 2=中级异常, 3=高危异常 |
| health_message              | string  | 是   | 健康状态说明文字                                         |
| alert_type                  | string  | 是   | 状态标识：NORMAL / LIGHT / MID / HIGH                    |
| current_rms                 | float   | 是   | 当前10分钟窗口的RMS值                                    |
| current_variance            | float   | 是   | 当前10分钟窗口的方差值                                   |
| current_frequency_center    | float   | 是   | 当前10分钟窗口的频率中心（Hz）                           |
| predicted_center_shift      | float   | 是   | 预测的下一个10分钟窗口的中心偏移距离                     |
| predicted_cluster_radius    | float   | 是   | 预测的下一个10分钟窗口的分布半径（95%分位数）            |
| health_thresholds           | object  | 是   | 健康分级阈值（基于中心偏移距离）                         |

### health_thresholds 对象字段

| 字段                        | 类型    | 说明                                  |
| --------------------------- | ------- | ------------------------------------- |
| in_range                    | boolean | 当前/预测中心偏移距离是否在健康范围内 |
| exceeded_percentage         | float   | 超出健康范围的比例（%）               |
| historical_min_center_shift | float   | 历史中心偏移距离最小值                |
| historical_max_center_shift | float   | 历史中心偏移距离最大值                |
| yellow_upper                | float   | 轻微异常上限                          |
| yellow_lower                | float   | 轻微异常下限（通常为0）               |
| orange_upper                | float   | 中级异常上限                          |
| orange_lower                | float   | 中级异常下限                          |
| red_upper                   | float   | 高危异常上限                          |
| red_lower                   | float   | 高危异常下限                          |

## 6. 响应格式

成功：
```json
true
```

或
```json
{
  "code": 200,
  "message": "成功"
}
```

失败：
```json
false
```

或
```json
{
  "code": 500,
  "message": "保存失败: xxx"
}
```

## 7. 注意事项

1. 所有时间戳均为北京时间（UTC+8）
2. 浮点数建议保留3位小数
3. 中文字段必须使用UTF-8编码
4. 推送超时建议30秒，失败可重试最多3次
5. 同一触发时间的多个传感器结果应放在同一个请求的sensors数组中

技术支持：如有疑问请联系对接人员
