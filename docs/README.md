# MEV-Detect 实验文档

> **课程项目：** 区块链技术与应用  
> **课题：** 基于 MEVisor 的 L1-L2 跨层 MEV 检测与可视化系统

---

## 文档导航

| 文档 | 说明 |
|------|------|
| [项目说明.md](./项目说明.md) | 课题背景、MEVisor 对照、**§7 老师反馈版（课题重定位）** |
| [最小闭环实施方案.md](./最小闭环实施方案.md) | **老师框架下的 MVP 开发步骤、文件清单、E1–E4 重跑计划** |
| [实验清单.md](./实验清单.md) | 四项实验的输入、命令、产出、验收标准（待与 §7 同步） |
| [开发指南.md](./开发指南.md) | 分阶段开发步骤、里程碑、联调流程 |
| [模块设计说明.md](./模块设计说明.md) | 架构、接口、数据结构、待实现模块 |
| [环境配置指南.md](./环境配置指南.md) | 本地环境、RPC、**多 GPU 云主机租用** |
| [数据获取指南.md](./数据获取指南.md) | Mock / 实时 / 历史数据规范 |
| [实验报告模板.md](./实验报告模板.md) | 课程报告与 PPT 撰写模板 |

---

## 推荐阅读顺序

### 第一次接触

1. [项目说明.md](./项目说明.md) — 弄清要做什么；**必读 §7 老师反馈版**  
2. [最小闭环实施方案.md](./最小闭环实施方案.md) — 按阶段推进的具体任务与验收  
3. [实验清单.md](./实验清单.md) — 明确四项实验各自交付什么  
4. [环境配置指南.md](./环境配置指南.md) — 搭建本地开发环境  

### 开始编码

4. [模块设计说明.md](./模块设计说明.md) — 接口与设计  
5. [开发指南.md](./开发指南.md) — 按阶段推进  
6. [数据获取指南.md](./数据获取指南.md) — 配置数据源  

### 实验收尾

7. [实验报告模板.md](./实验报告模板.md) — 整理图表与结论  

---

## 项目目录结构

```text
MEV-Detect/
├── config/
│   └── settings.py              # RPC、池子、PSO、监听阈值
├── src/
│   ├── listener.py              # 双链异步监听
│   ├── optimizer.py             # 向量化 PSO + GA 对照
│   ├── models.py                # 跨层成本模型
│   ├── pool_utils.py            # Swap 解码与价格换算
│   ├── rpc_utils.py             # RPC 鉴权与同步客户端
│   ├── data_loader.py           # Mock 数据加载
│   └── types.py                 # 核心数据结构
├── scripts/
│   ├── check_env.py             # 环境自检
│   ├── test_rpc.py              # RPC 连通测试
│   ├── run_ddp.py               # 【待实现】多 GPU 入口
│   └── benchmark_multi_gpu.py   # 【待实现】多卡性能实验
├── viz/
│   └── dashboard.py             # 【待实现】Streamlit 可视化面板
├── tests/
│   ├── test_pso_vs_ga.py        # PSO vs GA 对比
│   ├── test_listener.py         # 监听与 RPC 测试
│   └── test_models.py           # 成本模型测试
├── data/
│   ├── mock_mempool.json        # 离线测试数据
│   └── figures/                 # 实验图表输出
├── docs/                        # 本目录
└── main.py                      # 检测主入口
```

---

## 核心公式

$$\text{Fitness} = \text{Profit}_{L2} - \text{Gas}_{L1} - \text{Gas}_{L2} - \text{Bridge\_Fee}$$

---

## 参考论文

- **MEVisor:** High-Throughput MEV Discovery in DEXs with GPU Parallelism (NDSS 2025)

---

## 快速命令

```bash
# 环境自检
python scripts/check_env.py

# Mock 端到端检测
python main.py --mock

# 实时监听（国内 ETH WebSocket 不稳定时加 --http）
python main.py --live --http --duration 60

# 多 GPU 对比实验
python tests/test_pso_vs_ga.py

# Phase 1：拉取跨层多池快照（需 .env RPC）
python scripts/fetch_multi_pool_snapshot.py --latest --out data/snapshots/snap_latest.json

# 单元测试
python -m pytest tests/ -v
```
