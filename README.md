# Entanglement Entropy Tutorial

计算量子多体系统纠缠熵的教学代码库，配套教程《纠缠熵的计算：从偏迹到高效算法与二维推广》。

## 快速开始

```bash
# 安装依赖
pip install numpy scipy scikit-learn pytest

# 运行演示
python demos/demo_1d_cft.py        # 1D CFT 验证
python demos/demo_2d_area_law.py   # 2D 面积律对比
python demos/demo_compare_methods.py  # 三种方法对比

# 运行测试
pytest tests/ -v
```

## 项目结构

```
code/entanglement-entropy-tutorial/
├── README.md                   # 本文件
├── requirements.txt            # Python 依赖
├── ee/                         # 核心计算模块
│   ├── __init__.py
│   ├── core.py                # 三种纠缠熵计算方法
│   │   ├── reshape_psi        # 波函数重塑为系数矩阵
│   │   ├── ee_method1         # 密度矩阵本征分解
│   │   ├── ee_method2         # 直接 SVD（推荐）
│   │   ├── ee_method3         # 随机化 SVD
│   │   ├── rsvd               # 手写随机 SVD
│   │   ├── rsvd_sklearn       # sklearn 随机 SVD
│   │   ├── compute_entanglement_spectrum  # 纠缠谱
│   │   └── compute_renyi_entropy          # Rényi 熵
│   └── free_fermion.py        # 自由费米子关联矩阵方法
│       ├── ee_corr_matrix     # 关联矩阵计算纠缠熵
│       ├── ee_corr_matrix_renyi
│       └── build_corr_matrix_from_hamiltonian
├── lattices/                   # 格点构造模块
│   ├── __init__.py
│   ├── chain_1d.py            # 1D 链
│   ├── square_2d.py           # 2D 方格子
│   ├── honeycomb_2d.py        # 蜂窝格点
│   └── subsystems.py          # 子系统构造工具
├── demos/                      # 演示脚本
│   ├── demo_1d_cft.py         # 1D CFT 验证
│   ├── demo_2d_area_law.py    # 2D 面积律对比
│   └── demo_compare_methods.py # 三种方法全面比较
├── tests/                      # 单元测试
│   ├── test_methods.py        # 计算方法测试
│   └── test_lattices.py       # 格点测试
├── data/                       # 数据输出目录
└── figs/                       # 图形输出目录
```

## 核心功能

### 1. 通用计算方法（任意量子态）

```python
import numpy as np
from ee import ee_method1, ee_method2, ee_method3

# Bell 态: (|00⟩ + |11⟩)/√2
psi = np.array([1, 0, 0, 1]) / np.sqrt(2)

# 三种方法计算纠缠熵
S1 = ee_method1(psi, N=2, NA=1)  # 密度矩阵本征分解
S2 = ee_method2(psi, N=2, NA=1)  # 直接 SVD（推荐）
S3 = ee_method3(psi, N=2, NA=1, k=2)  # 随机化 SVD

print(f"S = {S2:.4f}")  # Should be ln(2) ≈ 0.6931
```

### 2. 随机化 SVD：手写 vs sklearn

```python
from ee import rsvd, rsvd_sklearn
import numpy as np

C = np.random.randn(100, 200)

# 手写实现
s_custom = rsvd(C, k=10, p=5, n_iter=2)

# sklearn 实现
s_sklearn = rsvd_sklearn(C, k=10, n_oversamples=5, n_iter=2)
```

### 3. 自由费米子关联矩阵方法

```python
from ee import ee_corr_matrix
from lattices import chain_1d

# 构造 1D 链的关联矩阵
L = 100
G = chain_1d(L, pbc=False)

# 计算半链纠缠熵
S = ee_corr_matrix(G, list(range(L // 2)))
print(f"Half-chain EE for L={L}: S = {S:.4f}")
```

### 4. 2D 格点系统

```python
from ee import ee_corr_matrix
from lattices import square_2d, honeycomb_2d, subsystem_left_half

# 方格子
G_sq = square_2d(10, 10, pbc=False)
sub_sq = subsystem_left_half(10, 10, sites_per_cell=1)
S_sq = ee_corr_matrix(G_sq, sub_sq)

# 蜂窝格点
G_hc = honeycomb_2d(10, 10, pbc=False)
sub_hc = subsystem_left_half(10, 10, sites_per_cell=2)
S_hc = ee_corr_matrix(G_hc, sub_hc)
```

## 与教程的对应关系

| 教程章节 | 代码位置 |
|---------|---------|
| §2-3: 偏迹与 Schmidt 分解 | `ee/core.py::reshape_psi` |
| §4: 手算练习 | `tests/test_methods.py` |
| §5.1: 密度矩阵本征分解 | `ee/core.py::ee_method1` |
| §5.2: 直接 SVD | `ee/core.py::ee_method2` |
| §5.3: 方法二优势 | `demos/demo_compare_methods.py` |
| §5.4: 随机化 SVD | `ee/core.py::rsvd`, `rsvd_sklearn` |
| §6: 关联矩阵方法 | `ee/free_fermion.py::ee_corr_matrix` |
| §7: 1D CFT 验证 | `demos/demo_1d_cft.py` |
| §8: 2D 推广 | `demos/demo_2d_area_law.py` |

## 三种方法对比

| 方法 | 时间复杂度 | 空间复杂度 | 稳定性 | 适用场景 |
|------|-----------|-----------|--------|---------|
| Method 1 (ρ 本征值) | O(d_A³) | O(d_A²) | 差（条件数平方） | 需要 ρ_A 时 |
| Method 2 (SVD) | O(d_A² d_B) | O(d_A d_B) | 好 | **通用默认** |
| Method 3 (rSVD) | O(d_A d_B k) | O(d_A k) | 好 | 面积律态 |

## 验证结果

### 1D CFT 验证（Calabrese-Cardy）

对开边界 XX 链，半链纠缠熵应满足：
```
S = (c/6) ln(2L/π) + const,  c = 1
```

数值结果：拟合系数 ≈ 0.164，与理论值 1/6 ≈ 0.167 相差 < 2%。

### 2D 面积律对比

- **方格子**（有 Fermi 面）：S/L ~ ln(L) 对数发散
- **蜂窝格点**（Dirac 点）：S/L → 常数（严格面积律）

## 参考文献

1. Eisert, Cramer & Plenio, Rev. Mod. Phys. 82, 277 (2010).
2. Calabrese & Cardy, J. Phys. A 42, 504005 (2009).
3. Vidal et al., PRL 90, 227902 (2003).
4. Peschel, J. Phys. A 36, L205 (2003).
5. Gioev & Klich, PRL 96, 100503 (2006).
6. Halko, Martinsson & Tropp, SIAM Rev. 53, 217 (2011).
7. D'Emidio et al., PRL 132, 076502 (2024).

## 作者

基于教程《纠缠熵的计算：从偏迹到高效算法与二维推广》实现。
