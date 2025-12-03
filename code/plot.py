"""
多面体領域の可視化関数
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def classify_region(p_a, p_b, p_alpha, thresholds=[0.1, 0.5, 0.9]):
    """割り当て確率の値に基づいて領域を分類"""
    def get_level(p, thresh):
        if p < thresh[0]:
            return 0
        elif p < thresh[1]:
            return 1
        else:
            return 2
    
    level_a = get_level(p_a, thresholds)
    level_b = get_level(p_b, thresholds)
    level_alpha = get_level(p_alpha, thresholds)
    
    if level_a == 2 and level_b == 2 and level_alpha == 2:
        return 'U{1,2,3}'
    elif level_a == 2 and level_b == 2:
        return 'U{1,2}'
    elif level_a == 2 and level_alpha == 2:
        return 'U{1,3}'
    elif level_b == 2 and level_alpha == 2:
        return 'U{2,3}'
    elif level_a == 2:
        return 'U{1}'
    elif level_b == 2:
        return 'U{2}'
    elif level_alpha == 2:
        return 'U{3}'
    else:
        return 'U{}'


def plot_polyhedral_regions(points_arr, p_arr, q_arr, title="Allocation Regions", 
                           figsize=(16, 12), thresholds=[0.1, 0.5, 0.9]):
    """
    多面体領域を可視化する関数
    
    パラメータ:
        points_arr: 型空間の点 (np.ndarray, shape: (J, 3))
        p_arr: 配分確率 (np.ndarray, shape: (3, J) または (J, 3))
        q_arr: シナジーαの配分確率 (np.ndarray, shape: (J,))
        title: プロットのタイトル (str)
        figsize: 図のサイズ (tuple)
        thresholds: 領域分類の閾値 (list)
    """
    # NumPy配列に変換
    points_arr = np.array(points_arr, dtype=np.float64)
    p_arr = np.array(p_arr, dtype=np.float64)
    q_arr = np.array(q_arr, dtype=np.float64)
    
    # p_arrの形状を確認して調整
    if p_arr.ndim == 2:
        if p_arr.shape[0] == 3 and p_arr.shape[1] == len(points_arr):
            # shape: (3, J) -> (J, 3)に変換
            p_arr = p_arr.T
        elif p_arr.shape[1] == 3 and p_arr.shape[0] == len(points_arr):
            # shape: (J, 3) そのまま
            pass
        else:
            raise ValueError(f"p_arrの形状が不正です: {p_arr.shape}, 期待される形状: (3, J) または (J, 3)")
    
    # 色のマッピング
    region_colors = {
        'U{1,2,3}': (0.2, 0.2, 0.2, 0.8),
        'U{1,2}': (0.4, 0.4, 0.4, 0.7),
        'U{1,3}': (0.4, 0.4, 0.4, 0.7),
        'U{2,3}': (0.4, 0.4, 0.4, 0.7),
        'U{1}': (0.7, 0.7, 0.7, 0.6),
        'U{2}': (0.7, 0.7, 0.7, 0.6),
        'U{3}': (0.7, 0.7, 0.7, 0.5),
        'U{}': (0.9, 0.9, 0.9, 0.4)
    }
    
    # 各点の領域を分類
    regions = []
    for i in range(len(points_arr)):
        region = classify_region(p_arr[i, 0], p_arr[i, 1], q_arr[i], thresholds)
        regions.append(region)
    
    # 領域ごとに点をグループ化
    region_groups = {}
    for i, region in enumerate(regions):
        if region not in region_groups:
            region_groups[region] = []
        region_groups[region].append(i)
    
    # 3Dプロットを作成
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 各領域の凸包を計算して多面体として描画
    for region_name, indices in region_groups.items():
        if len(indices) < 4:
            continue
        
        region_points = points_arr[indices]
        
        try:
            hull = ConvexHull(region_points)
            faces = []
            for simplex in hull.simplices:
                faces.append([region_points[simplex[0]], 
                             region_points[simplex[1]], 
                             region_points[simplex[2]]])
            
            poly3d = Poly3DCollection(faces, 
                                      alpha=region_colors.get(region_name, (0.5, 0.5, 0.5, 0.5))[3],
                                      facecolor=region_colors.get(region_name, (0.5, 0.5, 0.5, 0.5))[:3],
                                      edgecolor='black',
                                      linewidths=0.5)
            ax.add_collection3d(poly3d)
            
            center = region_points.mean(axis=0)
            ax.text(center[0], center[1], center[2], region_name, 
                   fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        except:
            ax.scatter(region_points[:, 0], region_points[:, 1], region_points[:, 2],
                      c=[region_colors.get(region_name, (0.5, 0.5, 0.5, 0.5))[:3]],
                      s=50, alpha=region_colors.get(region_name, (0.5, 0.5, 0.5, 0.5))[3],
                      label=region_name, edgecolors='black', linewidths=0.5)
    
    # 軸の設定
    ax.set_xlabel('x_a (Item a)', fontsize=12)
    ax.set_ylabel('x_b (Item b)', fontsize=12)
    ax.set_zlabel('x_α (Synergy)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # 軸の範囲を設定
    x_range = [points_arr[:, 0].min(), points_arr[:, 0].max()]
    y_range = [points_arr[:, 1].min(), points_arr[:, 1].max()]
    z_range = [points_arr[:, 2].min(), points_arr[:, 2].max()]
    
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    
    ax.grid(True, alpha=0.3)
    
    # 凡例を追加
    handles = []
    for region_name in sorted(region_groups.keys()):
        color = region_colors.get(region_name, (0.5, 0.5, 0.5, 0.5))
        handles.append(plt.Rectangle((0,0),1,1, facecolor=color[:3], alpha=color[3], 
                                     edgecolor='black', label=region_name))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    plt.tight_layout()
    plt.show()
    
    print(f"領域の統計:")
    for region_name in sorted(region_groups.keys()):
        print(f"  {region_name}: {len(region_groups[region_name])} 点")

