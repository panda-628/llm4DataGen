# 计算每一层所有头的权值的和 - 详细指南

## 概述

在Transformer模型中，每一层都有多个注意力头（通常为8-16个），每个头都会产生注意力权重矩阵。为了分析模型的整体注意力模式，我们需要将同一层的所有头的注意力权重进行聚合。

## 注意力权重的结构

```python
# attention_weights 的结构
attention_weights: Tuple[torch.Tensor, ...]
# 每个元素 shape: [batch_size, num_heads, seq_len, seq_len]
# 例如: [1, 12, 512, 512] 表示1个batch，12个头，序列长度512
```

## 聚合方法

### 1. 求和聚合 (Sum Aggregation)

**原理**: 将所有头的注意力权重相加
**公式**: `sum_attention = Σ(head_i)`
**代码**:
```python
sum_attention = layer_attention.sum(dim=0)  # [seq_len, seq_len]
```

**优点**:
- 保留所有头的信息
- 突出强连接
- 数值较大，便于观察

**缺点**:
- 可能放大数值
- 不同头的贡献被同等对待

### 2. 平均聚合 (Mean Aggregation)

**原理**: 计算所有头注意力权重的平均值
**公式**: `mean_attention = (1/N) * Σ(head_i)`
**代码**:
```python
mean_attention = layer_attention.mean(dim=0)  # [seq_len, seq_len]
```

**优点**:
- 数值稳定
- 最常用的聚合方法
- 平衡所有头的贡献

**缺点**:
- 可能掩盖个别头的重要信息

### 3. 最大聚合 (Max Aggregation)

**原理**: 取每个位置的最大注意力权重
**公式**: `max_attention = max(head_i)`
**代码**:
```python
max_attention = layer_attention.max(dim=0)[0]  # [seq_len, seq_len]
```

**优点**:
- 突出最强连接
- 保留最显著的模式
- 便于识别关键关系

**缺点**:
- 丢失其他头的信息
- 可能过于稀疏

## 实现代码

```python
def calculate_layer_attention_sums(attention_weights):
    """
    计算每一层所有头的权值的和
    
    Args:
        attention_weights: 注意力权重元组
                          shape: (num_layers, batch_size, num_heads, seq_len, seq_len)
    
    Returns:
        layer_sums: 字典，包含每一层的聚合结果
    """
    layer_sums = {}
    
    for layer_idx, layer_attention in enumerate(attention_weights):
        # 移除batch维度
        layer_attention = layer_attention[0]  # [num_heads, seq_len, seq_len]
        
        # 方法1: 求和聚合
        sum_attention = layer_attention.sum(dim=0)  # [seq_len, seq_len]
        
        # 方法2: 平均聚合
        mean_attention = layer_attention.mean(dim=0)  # [seq_len, seq_len]
        
        # 方法3: 最大聚合
        max_attention = layer_attention.max(dim=0)[0]  # [seq_len, seq_len]
        
        # 计算统计信息
        layer_stats = {
            'sum_attention': {
                'weights': sum_attention.numpy(),
                'mean_value': float(sum_attention.mean()),
                'max_value': float(sum_attention.max()),
                'std_value': float(sum_attention.std())
            },
            'mean_attention': {
                'weights': mean_attention.numpy(),
                'mean_value': float(mean_attention.mean()),
                'max_value': float(mean_attention.max()),
                'std_value': float(mean_attention.std())
            },
            'max_attention': {
                'weights': max_attention.numpy(),
                'mean_value': float(max_attention.mean()),
                'max_value': float(max_attention.max()),
                'std_value': float(max_attention.std())
            }
        }
        
        layer_sums[layer_idx] = layer_stats
    
    return layer_sums
```

## 使用示例

```python
# 获取注意力权重
outputs = model(input_ids, token_type_ids=token_type_ids)
attention = outputs[-1]  # 注意力权重元组

# 计算每一层的聚合注意力
layer_attention_sums = calculate_layer_attention_sums(attention)

# 查看结果
for layer_idx, layer_data in layer_attention_sums.items():
    print(f"第 {layer_idx + 1} 层:")
    print(f"  求和聚合 - 均值: {layer_data['sum_attention']['mean_value']:.4f}")
    print(f"  平均聚合 - 均值: {layer_data['mean_attention']['mean_value']:.4f}")
    print(f"  最大聚合 - 均值: {layer_data['max_attention']['mean_value']:.4f}")
```

## 高级聚合方法

### 1. 加权聚合

```python
def weighted_aggregation(attention_weights, weights):
    """
    基于权重的聚合
    
    Args:
        attention_weights: [num_heads, seq_len, seq_len]
        weights: [num_heads] 每个头的权重
    
    Returns:
        weighted_attention: [seq_len, seq_len]
    """
    return (attention_weights * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)
```

### 2. 基于重要性的聚合

```python
def importance_based_aggregation(attention_weights):
    """
    基于头重要性的聚合
    """
    # 计算每个头的重要性
    head_importance = calculate_head_importance(attention_weights)
    
    # 加权聚合
    return weighted_aggregation(attention_weights, head_importance)

def calculate_head_importance(attention_weights):
    """
    计算每个头的重要性
    """
    # 基于方差
    head_variance = attention_weights.var(dim=(1, 2))
    
    # 基于熵
    attention_probs = torch.softmax(attention_weights, dim=-1)
    head_entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=(1, 2))
    
    # 综合重要性
    importance = (head_variance + head_entropy) / 2
    return torch.softmax(importance, dim=0)
```

## 注意事项

1. **数值范围**: 求和聚合的数值通常较大，平均聚合的数值在0-1之间
2. **内存使用**: 处理长序列时注意内存使用
3. **精度问题**: 使用float32或float64避免精度损失
4. **可视化**: 聚合后的权重可以用于热力图可视化

## 应用场景

1. **模型分析**: 理解不同层的注意力模式
2. **可解释性**: 分析模型的决策过程
3. **调试**: 发现模型的注意力异常
4. **研究**: 研究不同聚合方法的效果 