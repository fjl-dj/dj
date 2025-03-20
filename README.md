import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# 1. 数据生成模块
# ======================
def generate_data(n=100):
    np.random.seed(42)
    data = {
        'S': np.clip(np.random.normal(loc=3.0, scale=1.5, size=n),  # 动作稳定性（0-5）
        'P': np.clip(np.random.normal(loc=1.0, scale=0.5, size=n),   # 运动表现（0-2）
        'M': np.random.randint(1, 4, size=n),                        # 心理状态（1-3）
        '失误率': np.clip(np.random.beta(a=2, b=5, size=n)*100,       # 失误率（0-100%）
        '实际恢复速度': np.clip(np.random.normal(loc=0.7, scale=0.15, size=n),  # 实际恢复速度（0-0.89）
    }
    df = pd.DataFrame(data)
    df['实际恢复速度'] = np.clip(df['实际恢复速度'], 0, 0.89)  # 限制恢复速度不超过理论最大值
    return df

# ======================
# 2. 模型计算模块
# ======================
def calculate_dbi(S, P, M, error_rate):
    E = 1 - error_rate / 100
    epsilon = 0.01
    numerator = 0.45*S + 0.30*P + 0.25*M
    denominator = 0.50*E + epsilon
    return numerator / denominator

def calculate_rest_days(DBI, M, actual_recovery):
    Delta_D = (4.0 - DBI) / 4.0
    psi = (3 - M) / 2  # 心理惩罚项（M=1时psi=1.0）
    nu = actual_recovery / 0.89  # 理论最大恢复速度0.89
    D = (Delta_D * 24) / 0.105 + 1.5*psi + 2.0*(1 - nu)
    # 四象阶段约束
    D = np.clip(D, 1.0, 7.0)  # 强制停训天数在1-7天之间
    return D

# ======================
# 3. 运行与分析
# ======================
if __name__ == "__main__":
    # 生成100条数据
    df = generate_data(100)
    
    # 计算DBI和停训天数
    df['DBI'] = calculate_dbi(df['S'], df['P'], df['M'], df['失误率'])
    df['停训天数'] = calculate_rest_days(df['DBI'], df['M'], df['实际恢复速度'])
    
    # 数据分析
    print("===== 描述性统计 =====")
    print(df[['DBI', '停训天数']].describe())
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    # DBI分布
    plt.subplot(1, 2, 1)
    plt.hist(df['DBI'], bins=20, color='skyblue', edgecolor='black')
    plt.axvline(x=4.0, color='red', linestyle='--', label='强制调整阈值')
    plt.axvline(x=6.0, color='green', linestyle='--', label='表现跃迁阈值')
    plt.xlabel('DBI')
    plt.ylabel('频数')
    plt.title('DBI分布')
    plt.legend()
    
    # 停训天数与DBI关系
    plt.subplot(1, 2, 2)
    plt.scatter(df['DBI'], df['停训天数'], alpha=0.6)
    plt.xlabel('DBI')
    plt.ylabel('停训天数')
    plt.title('DBI与停训天数关系')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # 输出前5条数据示例
    print("\n===== 前5条数据示例 =====")
    print(df.head())
