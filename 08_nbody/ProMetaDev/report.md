# CUDA N-Body Gravity Simulation 项目总结报告

> 作者：Fairy / 自动搭建 & 调优  环境：WSL2 + CUDA 12.2 + RTX 5070 Ti Laptop GPU（CC=12.0, 46 SMs, 12GB VRAM）

## 1. 项目结构

```
CudaNBodyGravitySim_2026summer/
├── src/
│   ├── main.cpp              # CLI 解析 + 主循环 + 计时 + 能量校验
│   ├── file_io.cpp           # particles.txt / params.txt 解析、二进制轨迹写出、CSV 轨迹、性能日志
│   ├── analysis.cpp          # CPU 参考积分、动能 / 势能 / 总能量 / 动量计算
│   └── nbody_kernels.cu      # CUDA 内核：simple / shared_tiling、Euler / Leapfrog、Host 端包装
├── include/                  # 头文件（SimParams, KernelMode, Integrator 枚举、IO/分析接口）
├── scripts/
│   ├── detect_sm.sh          # CUDA SM 代号探测（供 build.sh 生成 -gencode）
│   ├── generate_particles.py # 多模型粒子生成：Plummer 球 / 均匀位力球 / 开普勒盘
│   ├── scan_drift.py         # dt × eps 参数扫描工具，找能量漂移 <10% 最优组合
│   ├── visualize.py          # 轨迹可视化：binary/CSV/demo；2D/3D；GIF/MP4
│   ├── check_bin.py          # 二进制轨迹文件格式校验（N, R, R*N*3*4 字节完整性）
│   ├── perf_to_csv.py        # perf.log(JSON) → CSV
│   └── regression_assert.py  # 回归断言 helper
├── data/                     # particles.txt / params.txt（含 2body / 3body / 4096 / 65536）
├── outputs/                  # trajectory.bin / trajectory.csv / perf.log / GIF
├── build.sh                  # 一键 CMake 构建（自动探测 CUDA SM）
└── run_test.sh / regression_test.sh  # 2body / 3body / 4096 三步回归测试
```

## 2. 算法与实现

### 2.1 N 体问题与软化因子
经典牛顿引力 $F_{ij}=G m_i m_j \frac{r_j-r_i}{|r_{ij}|^3}$，在 `r_{ij}→0` 时数值发散，引入**软化因子 ε**：

$$F_{ij} = G m_i m_j \frac{r_j-r_i}{(|r_{ij}|^2 + \varepsilon^2)^{3/2}}$$

### 2.2 数据布局 (SoA)
采用 **Structure-of-Arrays**（符合题面要求）：
```cpp
struct ParticleSet { int64_t N; float *x,*y,*z,*vx,*vy,*vz,*mass; };
```
全局内存合并访问，与 shared memory tile 粒度（128 float / tile）对齐，获得最大带宽利用率。

### 2.3 CUDA 内核
| 内核 | 实现 | 特点 |
|-----|------|-----|
| `k_accel_simple`       | 每个 thread 管 1 粒子，遍历全 $O(N^2)$ 对 | 基准实现，代码简洁 |
| `k_accel_shared_tiling`| 以 tile=128 为块把 j 侧粒子 load 到 `__shared__`，再 i×tile 计算 | 减少重复全局内存访问，L2 / SMEM 复用 |
| `k_integrate_euler`    | $x ← x + v·dt, v ← v + a·dt$ | 一阶显式，能量漂移大 |
| `k_integrate_leapfrog` | $(x,v)$ 半步交错：$v_{t+1/2}=v_{t-1/2}+a_t·dt,\; x_{t+1}=x_t+v_{t+1/2}·dt$ | 辛积分，能量有界波动、长期稳定 |

### 2.4 输出格式
**二进制轨迹文件**（严格按题面）：
```
Header : int32 N, int32 R              (总 8 字节)
Data   : R × N × 3 float32 (x,y,z)     (R 帧，每帧 N 粒子)
```
大小验证：`8 + R*N*12 = 实际 bytes` ✓

**性能日志 `perf.log`**（JSON 行）：
```
{ mode, integrator, N, steps, dt, eps, record_interval, total_ms, avg_step_ms,
  throughput_particles_per_s, gpu_mem_MB, cpu_time_ms, speedup }
```

**CSV 轨迹**（启用 `--csv` 或 `write_csv=1`）：
```
frame,particle_id,x,y,z,vx,vy,vz
```

## 3. CLI / 配置优先级
`main.cpp` 支持两种配置来源：
1. 位置参数：`./nbody particles.txt params.txt trajectory.bin perf.log`
2. `--kernel {simple|tiling}`、`--integrator {euler|leapfrog}`、`--dt`、`--steps`、`--record`、`--G`、`--softening`、`--block-size`、`--no-cpu`、`--check-energy`、`--csv`、`--info`、`--analyze N`

**优先级规则（已修补）**：`CLI 显式参数 > params.txt > 默认值`（之前存在 params 覆盖 CLI 的瑕疵，已通过保存 sentinel 在 `parse_params_file` 之后再次应用覆盖的方式修复，见 `src/main.cpp` 133~146 行）。

## 4. 功能验证与性能

### 4.1 2 体（Kepler 圆轨道，N=2）
- 参数：dt=0.002, steps=1000, G=1, eps=0.01, Leapfrog
- 能量漂移：|ΔE/E₀| ≈ **0.02%**（<1% 验收线 ✓）
- 可视化：`outputs/two_body.gif` 稳定 1000 步不发散 ✓

### 4.2 3 体（经典小质量三体）
- 3 体轨迹正常生成，二进制 / CSV 格式通过 `check_bin.py`，可视化生成 `rep3csv.gif` ✓

### 4.3 N=4096 能量漂移调优
使用 `scripts/generate_particles.py plummer` 生成的 **Plummer 球**（位力平衡，冷球无整体漂移），在 `scripts/scan_drift.py` 扫描 `dt ∈ {1e-4..3e-3} × eps ∈ {0.005..0.1}` 共 **25 组参数**，**全部通过 <10% 验收**：

| 配置 | 能量漂移 | 平均步时 | 吞吐 |
|------|---------|---------|------|
| dt=1e-4, eps=0.1 | **6.33e-6 %** | 0.342 ms | 11.97 M particle-steps/s |
| dt=5e-4, eps=0.05 | 0.092 % | 0.342 ms | 11.96 M /s |
| dt=1e-3, eps=0.02 | **0.469 %** | 0.331 ms | 12.36 M /s（推荐平衡档） |
| dt=2e-3, eps=0.05 | 2.154 % | 0.337 ms | 12.14 M /s |
| dt=3e-3, eps=0.1  | 4.965 % | 0.324 ms | 12.64 M /s |

**关键调优点**：
1. **模型升级**：从“冷均匀球”（v=0，开始坍缩、能量剧烈变化）改为 **Plummer 分布 + Aarseth 速度抽样**，系统处于近似位力平衡，天然能量漂移降低 ~2 个量级。
2. **积分器**：Leapfrog（辛）优于 Euler，dt=1e-3 下漂移 0.469% vs 之前 Euler 的 74.6%。
3. **软化因子 ε**：过小会导致近距离粒子加速度剧变，需 ≥ 0.01 保证数值稳定。

### 4.4 N=65536 进阶规模（题面「进阶版」要求）
- 参数：Plummer 65536 粒子，dt=0.001，ε=0.02，Leapfrog，shared_tiling 内核，record_interval=100 → R=11 帧
- **实跑 1000 步通过**：
  - 总耗时：17.35 s
  - 平均步时：**17.35 ms / step**
  - 吞吐：**3.777 M particle-steps / s**
  - GPU 显存占用（SoA+中间量估算）：2.5 MB（远小于 12GB）
  - 二进制输出校验：8650760 bytes = 8 + 11×65536×12 ✓

### 4.5 多内核 / 多格式支持
| 能力 | 状态 |
|------|------|
| simple / shared_tiling 两种内核 | ✓（`--kernel simple/tiling`，4096/65536 均实测） |
| Euler / Leapfrog 两种积分 | ✓（`--integrator euler/leapfrog`） |
| 二进制轨迹输出 | ✓（check_bin.py 校验通过，2/3/4096/65536） |
| CSV 轨迹输出 | ✓（`--csv`，3 体已生成 rep3.csv） |
| perf.log JSON 行 | ✓（所有回归测试均生成） |
| 能量/动量校验 | ✓（`--check-energy`，`analysis.cpp` CPU 参考实现） |
| CLI 覆盖优先 | ✓（2025-08-28 修复，已实参 tiling + euler 验证） |

## 5. 可视化
`scripts/visualize.py` 支持：
- 输入：**二进制轨迹 / CSV**（自动识别），或 `--demo {2body|3body|cluster}` 生成示例动画
- 模式：2D xy 投影、3D 3D 视角
- 输出：`.gif`（默认）或 `.mp4`（`--fmt mp4`）
- 已实机生成：`outputs/two_body.gif`（2 体 1000 步稳定圆轨道）、`outputs/rep3csv.gif`（3 体 CSV 回放）

示例用法：
```bash
python3 scripts/visualize.py outputs/65k.bin --view 3d --frames 11 --out outputs/65k.gif
python3 scripts/visualize.py demo 3body
python3 scripts/visualize.py outputs/rep3.csv  --out outputs/rep3csv_fromcsv.gif
```

## 6. 回归测试
`bash run_test.sh` 顺序执行：
1. **Test 1**：2 体 1000 步 Leapfrog（要求能量漂移 <1%，实际 <0.03%）
2. **Test 2**：3 体 500 步 + CSV 输出（要求文件存在且可 check_bin）
3. **Test 3**：4096 粒子 1000 步 tiling kernel（要求二进制正确 + 生成 perf.log）
4. **Build Check**：`bash build.sh` 增量编译通过

所有三个用例在 RTX 5070 Ti Laptop 上均通过。

## 7. Profiling（NSYS / NCU）
WSL2 默认 CUDA Toolkit 12.2 基础安装未包含 `nsys` / `ncu` 命令行工具（需安装 NSight Systems / Compute 完整包），当前环境不可用；已保留接口与文档，用户在本地安装后可执行：
```bash
nsys profile -o outputs/65k_nsys  ./build/nbody data/particles_65536.txt data/params_default.txt outputs/65k.bin outputs/p.log --kernel tiling --no-cpu --softening 0.02 --steps 1000
ncu --set full --section SpeedOfLight -o outputs/ncu_65k ./build/nbody ...
```

## 8. 构建与使用
```bash
# 1. 搭建
bash build.sh

# 2. 生成进阶 65536 粒子（一次即可）
python3 scripts/generate_particles.py plummer 65536 data/particles_65536.txt --seed 2026 --Rscale 2.0 --Mtot 10.0

# 3. 跑 2 体（验证）
./build/nbody data/particles_two_body.txt data/params_default.txt outputs/two_body.bin outputs/perf.log --check-energy --integrator leapfrog --dt 0.002 --G 1 --softening 0.01 --steps 1000 --record 10

# 4. 跑进阶 65536
./build/nbody data/particles_65536.txt data/params_default.txt outputs/65k.bin outputs/perf_65k.log --kernel tiling --no-cpu --softening 0.02 --steps 1000 --record 100

# 5. 可视化
python3 scripts/visualize.py outputs/two_body.bin --out outputs/two_body.gif

# 6. 参数扫描（找 4096 最佳 dt, eps）
python3 scripts/scan_drift.py
```

## 9. 已知限制 / 未来工作
- 当前只有 O(N²) 直接求和，未实现 Barnes-Hut / FMM 树算法（N≥1M 时建议引入）。
- WSL2 默认未装 `nsys` / `ncu`，可在 Windows 侧 Nsight 软件中对同一 GPU 进行 profiling。
- 可视化暂未实现分块渲染超大 N 轨迹预览，可加 downsample 参数。
