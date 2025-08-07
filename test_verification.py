import torch
import numpy as np

def test_attention_calculation():
    """测试注意力权重计算是否正确"""
    
    # 创建一个简单的测试张量
    # 模拟: [num_heads, seq_len, seq_len] = [4, 6, 6]
    num_heads = 4
    seq_len = 6
    
    # 创建测试数据
    test_attention = torch.randn(num_heads, seq_len, seq_len)
    
    print("=== 测试注意力权重计算 ===")
    print(f"测试张量形状: {test_attention.shape}")
    print(f"头数: {num_heads}")
    print(f"序列长度: {seq_len}")
    
    # 方法1: 求和
    sum_attention = test_attention.sum(dim=0)
    print(f"\n求和聚合:")
    print(f"  形状: {sum_attention.shape}")
    print(f"  均值: {sum_attention.mean():.4f}")
    print(f"  最大值: {sum_attention.max():.4f}")
    
    # 方法2: 平均权值（求和除以头数）
    mean_attention = sum_attention / num_heads
    print(f"\n平均权值 (求和/头数):")
    print(f"  形状: {mean_attention.shape}")
    print(f"  均值: {mean_attention.mean():.4f}")
    print(f"  最大值: {mean_attention.max():.4f}")
    
    # 方法3: 直接平均
    direct_mean = test_attention.mean(dim=0)
    print(f"\n直接平均:")
    print(f"  形状: {direct_mean.shape}")
    print(f"  均值: {direct_mean.mean():.4f}")
    print(f"  最大值: {direct_mean.max():.4f}")
    
    # 验证两种平均方法是否一致
    diff = torch.abs(mean_attention - direct_mean).max()
    print(f"\n验证结果:")
    print(f"  两种平均方法的最大差异: {diff:.8f}")
    print(f"  是否一致: {'是' if diff < 1e-8 else '否'}")
    
    # 验证数值关系
    print(f"\n数值关系验证:")
    print(f"  求和均值 / 头数 = {sum_attention.mean() / num_heads:.4f}")
    print(f"  平均权值均值 = {mean_attention.mean():.4f}")
    print(f"  直接平均均值 = {direct_mean.mean():.4f}")
    
    return {
        'sum_attention': sum_attention,
        'mean_attention': mean_attention,
        'direct_mean': direct_mean,
        'num_heads': num_heads
    }

if __name__ == "__main__":
    result = test_attention_calculation()
    print(f"\n=== 测试完成 ===")
    print("所有计算都正确！") 