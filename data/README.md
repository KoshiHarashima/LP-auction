# Data Directory

このディレクトリには、最適オークション機構の計算結果がNumPy形式で保存されます。

## ファイル形式

- `results_*.npz`: 計算結果（NumPy圧縮形式）
- `results_*_metadata.txt`: メタデータ（テキスト形式）

## 対応している問題タイプ

1. **2財1シナジー** (`save_results`, `load_results`)
2. **2人2財1シナジー** (`save_results_2agents`, `load_results_2agents`)
3. **3財4シナジー** (`save_results_3goods`, `load_results_3goods`)

## 使用方法

### 1. 2財1シナジーの場合

#### 結果の保存

```python
from code import (
    solve_mechanism_iterative,
    save_results,
    make_tensor_grid_3d
)

# グリッド点と重みを生成
points, weights = make_tensor_grid_3d(10, 10, 10, BETA_PARAMS)

# 最適オークション機構を求解
status, obj_val, u_sol, p_sol, n_iter = solve_mechanism_iterative(
    points, weights, grid_sizes=(10, 10, 10), solver=SOLVER
)

# 結果を保存
save_results(
    points, weights, u_sol, p_sol, obj_val, status,
    grid_sizes=(10, 10, 10),
    n_iter=n_iter
)
```

#### 結果の読み込みと可視化

```python
from code import load_results, plot_polyhedral_regions

# 結果を読み込む
data = load_results('data/results_2goods_1synergy_20240101_120000.npz')

# データにアクセス
print(f"Status: {data['status']}")
print(f"Objective value: {data['obj_val']}")
print(f"u_sol shape: {data['u_sol'].shape}")  # (J,)
print(f"p_sol shape: {data['p_sol'].shape}")  # (3, J)

# 可視化
points_arr = data['points']
p_arr = data['p_sol']  # shape: (3, J)
q_arr = data['p_sol'][2, :]  # シナジーαの配分確率

plot_polyhedral_regions(
    points_arr, p_arr, q_arr,
    title="Allocation Regions (from saved data)"
)
```

### 2. 2人2財1シナジーの場合

#### 結果の保存

```python
from code import (
    solve_mechanism_2agents_iterative,
    save_results_2agents,
    make_tensor_grid_3d
)

# グリッド点と重みを生成
points1, weights1 = make_tensor_grid_3d(10, 10, 10, BETA_PARAMS)
points2, weights2 = make_tensor_grid_3d(10, 10, 10, BETA_PARAMS)

# 最適オークション機構を求解
status, obj_val, u1_sol, u2_sol, p1_sol, p2_sol, n_iter = \
    solve_mechanism_2agents_iterative(
        points1, weights1, (10, 10, 10),
        points2, weights2, (10, 10, 10),
        solver=SOLVER
    )

# 結果を保存
save_results_2agents(
    points1, weights1, points2, weights2,
    u1_sol, u2_sol, p1_sol, p2_sol,
    obj_val, status,
    grid_sizes1=(10, 10, 10),
    grid_sizes2=(10, 10, 10),
    n_iter=n_iter
)
```

#### 結果の読み込み

```python
from code import load_results_2agents

# 結果を読み込む
data = load_results_2agents('data/results_2agents_20240101_120000.npz')

# データにアクセス
print(f"Status: {data['status']}")
print(f"Objective value: {data['obj_val']}")
print(f"u1_sol shape: {data['u1_sol'].shape}")  # (J1, J2)
print(f"p1_sol shape: {data['p1_sol'].shape}")  # (3, J1, J2)
```

### 3. 3財4シナジーの場合

#### 結果の保存

```python
from code import (
    solve_mechanism_3goods_4synergy_iterative,
    save_results_3goods,
    make_tensor_grid_7d
)

# グリッド点と重みを生成
points, weights = make_tensor_grid_7d(N, BETA_PARAMS)

# 最適オークション機構を求解
status, obj_val, u_sol, p_sol, n_iter = \
    solve_mechanism_3goods_4synergy_iterative(
        points, weights, grid_size=N, solver=SOLVER
    )

# 結果を保存
save_results_3goods(
    points, weights, u_sol, p_sol, obj_val, status,
    grid_size=N,
    n_iter=n_iter
)
```

#### 結果の読み込み

```python
from code import load_results_3goods

# 結果を読み込む
data = load_results_3goods('data/results_3goods_4synergy_20240101_120000.npz')

# データにアクセス
print(f"Status: {data['status']}")
print(f"Objective value: {data['obj_val']}")
print(f"u_sol shape: {data['u_sol'].shape}")  # (J,)
print(f"p_sol shape: {data['p_sol'].shape}")  # (7, J)
```

## 保存されるデータ

### 2財1シナジー
- `points`: 型空間の点（NumPy配列、shape: (J, 3)）
- `weights`: 各点の重み（NumPy配列、shape: (J,)）
- `u_sol`: 効用（NumPy配列、shape: (J,)）
- `p_sol`: 配分確率（NumPy配列、shape: (3, J)）
- `obj_val`: 目的関数値
- `status`: LPステータス
- `grid_sizes`: グリッドサイズ（オプション）
- `n_iter`: 反復回数（オプション）

### 2人2財1シナジー
- `points1`, `points2`: 型空間の点（NumPy配列）
- `weights1`, `weights2`: 各点の重み（NumPy配列）
- `u1_sol`, `u2_sol`: 効用（NumPy配列、shape: (J1, J2)）
- `p1_sol`, `p2_sol`: 配分確率（NumPy配列、shape: (3, J1, J2)）
- `obj_val`: 目的関数値
- `status`: LPステータス
- `grid_sizes1`, `grid_sizes2`: グリッドサイズ（オプション）
- `n_iter`: 反復回数（オプション）

### 3財4シナジー
- `points`: 型空間の点（NumPy配列、shape: (J, 7)）
- `weights`: 各点の重み（NumPy配列、shape: (J,)）
- `u_sol`: 効用（NumPy配列、shape: (J,)）
- `p_sol`: 配分確率（NumPy配列、shape: (7, J)）
- `obj_val`: 目的関数値
- `status`: LPステータス
- `grid_size`: グリッドサイズ（オプション）
- `n_iter`: 反復回数（オプション）

## 注意事項

- すべての戻り値はNumPy配列形式です（リストではありません）
- `plot_polyhedral_regions`関数はNumPy配列を直接受け取れます
- ファイル名を指定しない場合、タイムスタンプが自動的に追加されます

