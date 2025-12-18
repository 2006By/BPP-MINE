# MCTS-Enhanced Hierarchical 3D Bin Packing

新增的文件和修改说明，基于原有的 Online-3D-BPP-PCT 代码库。

## 新增文件

### 1. `set_transformer.py`
**功能**: 属性感知的集合变换器（Set Transformer）网络

- 输入：Buffer中的N个物品（每个物品4维：L, W, H, Density）
- 输出：先验概率分布 + 状态价值估计
- 核心：多头自注意力机制，排列不变性（Permutation Invariance）

**关键类**:
- `SetTransformer`: 主网络
- `IndependentSetTransformer`: 独立策略-价值网络变体

### 2. `mcts_planner.py`
**功能**: 蒙特卡洛树搜索规划器

- 实现AlphaZero风格的MCTS搜索
- 使用Set Transformer提供先验概率
- 使用PCT网络作为仿真模型
- UCB算法引导探索

**关键类**:
- `MCTSNode`: 搜索树节点
- `MCTSPlanner`: MCTS主算法

### 3. `pct_envs/PctDiscrete0/bin3D_buffer.py`
**功能**: 带缓冲区的装箱环境

- 维护N=10个物品的前瞻缓冲区
- 物品队列管理和自动补充
- 与MCTS和PCT集成
- 提供Buffer观测和PCT观测

**关键类**:
- `PackingDiscreteWithBuffer`: 扩展的装箱环境

### 4. `train_mcts.py`
**功能**: MCTS训练循环

- 自我对弈数据收集
- 策略蒸馏（KL散度损失）
- 价值学习（MSE损失）
- TensorBoard日志

**关键类**:
- `MCTSTrainer`: 主训练器

### 5. `main_mcts.py`
**功能**: 主训练入口

- 完整的命令行参数解析
- 网络初始化
- 训练和评估模式
- 实验管理

## 使用方法

### 快速开始

```bash
# 从头训练（Set Transformer + PCT都从零开始）
python main_mcts.py --buffer-size 10 --mcts-simulations 200 --num-episodes 5000

# 使用预训练的PCT
python main_mcts.py --pct-model-path ./logs/experiment/PCT-xxx.pt --buffer-size 10 --mcts-simulations 200

# 评估模式
python main_mcts.py --evaluate --load-checkpoint ./logs/mcts_experiment/checkpoint_ep1000.pt --eval-episodes 100
```

### 主要参数

**环境设置**:
- `--buffer-size 10`: Buffer大小（N=10）
- `--total-items 150`: 每episode物品总数
- `--setting 2`: 实验设置（1/2/3）

**MCTS设置**:
- `--mcts-simulations 200`: 每次决策的MCTS模拟次数
- `--mcts-c-puct 1.0`: 探索常数
- `--mcts-temperature 1.0`: 动作选择温度

**网络设置**:
- `--st-d-model 128`: Set Transformer嵌入维度
- `--st-n-heads 4`: 注意力头数
- `--st-n-layers 3`: Transformer层数

**训练设置**:
- `--num-episodes 10000`: 训练episode数
- `--st-learning-rate 1e-4`: Set Transformer学习率
- `--train-pct`: 是否同时训练PCT（默认不训练）

## 架构说明

### 三层架构

```
┌─────────────────────────────────────────┐
│   Set Transformer (高层决策)             │
│   - 输入: Buffer (N=10 items)           │
│   - 输出: Prior probabilities + Value    │
└──────────────┬──────────────────────────┘
               │ 先验概率
               ↓
┌─────────────────────────────────────────┐
│   MCTS Planner (中层规划)                │
│   - Selection: UCB搜索                   │
│   - Expansion: 使用先验初始化             │
│   - Simulation: 调用PCT                  │
│   - Backpropagation: 更新Q值             │
└──────────────┬──────────────────────────┘
               │ 执行动作
               ↓
┌─────────────────────────────────────────┐
│   PCT Policy (底层执行)                  │
│   - 输入: 单个item + bin状态             │
│   - 输出: 最佳放置位置                    │
└─────────────────────────────────────────┘
```

### 数据流

1. **观测生成**: 
   - 环境 → Buffer观测 (10×4)
   
2. **先验估计**:
   - Buffer观测 → Set Transformer → (Prior, Value)
   
3. **MCTS搜索**:
   - 使用Prior初始化
   - 模拟时调用PCT执行放置
   - 搜索200次后得到改进的策略
   
4. **策略蒸馏**:
   - MCTS策略作为soft target
   - Set Transformer学习逼近MCTS
   - 通过KL散度优化

## 训练策略

### 自我对弈循环

```python
for episode in range(num_episodes):
    state = env.reset()
    while not done:
        # 1. MCTS搜索得到改进策略
        pi_mcts = mcts_planner.search(state)
        
        # 2. 采样动作
        action = sample(pi_mcts)
        
        # 3. 执行
        state, reward, done = env.step(action)
        
        # 4. 存储 (state, pi_mcts, reward)
        
    # 5. 策略蒸馏更新
    update_network(states, mcts_policies, rewards)
```

### 损失函数

```
Total Loss = Policy Loss + Value Loss

Policy Loss = -Σ π_mcts(a|s) log π_θ(a|s)  # KL散度
Value Loss = (V_θ(s) - z)²                   # MSE
```

## 与原始PCT的区别

| 特性 | 原始PCT | MCTS-Enhanced |
|-----|---------|---------------|
| **决策层级** | 单层（直接选item+位置） | 三层（Set Trans → MCTS → PCT） |
| **观测** | 单个item | Buffer (N=10 items) |
| **搜索** | 无 | MCTS 200次模拟 |
| **训练** | A2C/ACKTR | Policy Distillation |
| **前瞻** | 无 | 完整buffer前瞻 |

## 预期改进

1. **空间利用率**: +5-10%（通过前瞻选择更优物品顺序）
2. **物品排序**: 学习重物在下、大物优先等策略
3. **泛化能力**: 对不规则物品组合更鲁棒
4. **推理速度**: 训练后Set Transformer单独使用接近MCTS质量

## 文件依赖关系

```
main_mcts.py
├── set_transformer.py
├── mcts_planner.py
│   ├── set_transformer.py (先验)
│   └── model.py (PCT策略)
├── train_mcts.py
│   ├── set_transformer.py
│   ├── mcts_planner.py
│   └── model.py
└── pct_envs/PctDiscrete0/bin3D_buffer.py
    ├── space.py
    └── binCreator.py
```

## 监控指标

### TensorBoard

启动：
```bash
tensorboard --logdir=./logs/mcts_runs
```

监控指标：
- `Episode/Reward`: Episode累积奖励
- `Episode/SpaceRatio`: 空间利用率
- `Episode/NumPacked`: 成功装箱数量
- `Loss/Policy`: 策略损失（KL散度）
- `Loss/Value`: 价值损失（MSE）

### 控制台输出

```
Episode 100/10000
  Steps: 12500, FPS: 42.3
  Last 100 episodes:
    Mean reward: 6.234
    Mean ratio: 0.623
    Max ratio: 0.742
  Losses:
    Policy: 0.234
    Value: 0.156
```

## 下一步优化

1. **经验回放**: 添加replay buffer提高样本效率
2. **并行MCTS**: 多线程MCTS加速
3. **自适应模拟次数**: 根据不确定性调整模拟次数
4. **课程学习**: 逐步增加buffer大小和物品复杂度
5. **模型压缩**: 知识蒸馏到更小网络用于部署

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少`--mcts-simulations`
   - 减少`--buffer-size`
   - 使用`--no-cuda`

2. **训练不收敛**
   - 降低学习率
   - 检查MCTS是否正常工作
   - 先用预训练PCT

3. **速度太慢**
   - 减少MCTS模拟次数
   - 禁用TensorBoard频繁写入
   - 使用多GPU

## 引用

如果使用此代码，请引用：

```bibtex
@article{zhao2021learning,
  title={Learning Efficient Online 3D Bin Packing on Packing Configuration Trees},
  author={Zhao, Hang and She, Qijin and Zhu, Chenyang and Yang, Yin and Xu, Kai},
  journal={ICLR},
  year={2022}
}
```

以及AlphaZero：

```bibtex
@article{silver2017mastering,
  title={Mastering the game of go without human knowledge},
  author={Silver, David and others},
  journal={Nature},
  volume={550},
  number={7676},
  pages={354--359},
  year={2017}
}
```
