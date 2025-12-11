import cv2
import csv
import sys
import argparse
import numpy as np
from pathlib import Path
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def preprocess_image(img, save_path=None):
    """预处理图像，去除干扰"""
    # 使用双边滤波保持边缘的同时去噪
    denoised = cv2.bilateralFilter(img, 9, 75, 75)

    # 使用中值滤波进一步去除椒盐噪声
    denoised = cv2.medianBlur(denoised, 3)

    # 使用形态学操作去除小的干扰
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)

    if save_path is not None:
        cv2.imwrite(str(save_path), morphed)

    return morphed

def ransac_circle_fit(points, max_iterations=1000, threshold=3.0, min_inliers_ratio=0.6):
    """使用RANSAC算法拟合圆
    
    Args:
        points: Nx2 array, (y, x)
        max_iterations: RANSAC最大迭代次数
        threshold: 内点距离阈值（像素）
        min_inliers_ratio: 最小内点比例
    
    Returns:
        (x_center, y_center, radius, inliers_mask)
    """
    if len(points) < 3:
        return None
    
    best_circle = None
    best_inliers = None
    best_inlier_count = 0
    
    n_points = len(points)
    min_inliers = int(n_points * min_inliers_ratio)
    
    for iteration in range(max_iterations):
        # 随机选择3个点
        idx = np.random.choice(n_points, 3, replace=False)
        sample_points = points[idx]
        
        # 用3个点拟合圆
        circle = fit_circle_from_3points(sample_points)
        if circle is None:
            continue
        
        xc, yc, r = circle
        
        # 计算所有点到圆的距离
        distances = np.abs(np.sqrt((points[:, 1] - xc)**2 + (points[:, 0] - yc)**2) - r)
        
        # 找出内点
        inliers_mask = distances < threshold
        inlier_count = np.sum(inliers_mask)
        
        # 更新最佳模型
        if inlier_count > best_inlier_count:
            # 使用所有内点重新拟合圆
            inlier_points = points[inliers_mask]
            refined_circle = fit_circle_least_squares(inlier_points)
            
            if refined_circle is not None:
                best_circle = refined_circle
                best_inliers = inliers_mask
                best_inlier_count = inlier_count
        
        # 早停：如果找到足够好的模型
        if best_inlier_count > n_points * 0.8:
            break
    
    if best_circle is None or best_inlier_count < min_inliers:
        return None
    
    return (*best_circle, best_inliers)

def fit_circle_from_3points(points):
    """从3个点拟合圆"""
    if len(points) != 3:
        return None
    
    # 提取坐标 (y, x)
    y1, x1 = points[0]
    y2, x2 = points[1]
    y3, x3 = points[2]
    
    # 检查是否共线
    area = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    if area < 1e-6:
        return None
    
    # 计算圆心
    A = x2 - x1
    B = y2 - y1
    C = x3 - x1
    D = y3 - y1
    E = A * (x1 + x2) + B * (y1 + y2)
    F = C * (x1 + x3) + D * (y1 + y3)
    G = 2 * (A * (y3 - y2) - B * (x3 - x2))
    
    if abs(G) < 1e-6:
        return None
    
    xc = (D * E - B * F) / G
    yc = (A * F - C * E) / G
    r = np.sqrt((x1 - xc)**2 + (y1 - yc)**2)
    
    return xc, yc, r

def fit_circle_least_squares(points):
    """最小二乘法拟合圆"""
    y = points[:, 0]
    x = points[:, 1]
    
    A = np.column_stack([x, y, np.ones_like(x)])
    b = x**2 + y**2
    
    try:
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc = params[0] / 2
        yc = params[1] / 2
        r = np.sqrt(params[2] + xc**2 + yc**2)
        return xc, yc, r
    except:
        return None

def detect_pipe_circles(img, visualization=True, output_dir=None, save_intermediate=False):
    """检测金属管的内外圆

    Args:
        img: 输入图像（单通道或三通道）
        visualization: 是否调用可视化函数（一般用于交互）
        output_dir: 若给定，作为中间文件与可视化保存目录（Path 或 str）
        save_intermediate: 是否保存中间结果（processed, edges）到 output_dir

    Returns:
        outer_circle, inner_circle, wall_thickness, extras(dict)
        extras 包含: processed, edges, edge_points, hough_circles, candidate_circles
    """

    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = None

    processed = preprocess_image(img, save_path=(outdir / "processed.png") if (outdir and save_intermediate) else None)
    
    # 自适应边缘检测
    high_thresh = np.percentile(processed, 90)
    low_thresh = high_thresh * 0.3
    edges = cv2.Canny(processed, low_thresh, high_thresh)
    
    # 连接断裂的边缘
    kernel_line = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_line)
    
    # 去除小的连通域（噪点）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    min_size = 50  # 最小连通域大小
    cleaned_edges = np.zeros_like(edges)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_edges[labels == i] = 255
    
    edges = cleaned_edges

    if outdir is not None and save_intermediate:
        cv2.imwrite(str(outdir / "edges.png"), edges)
    
    # 获取所有边缘点
    edge_points = np.column_stack(np.where(edges > 0))
    
    if len(edge_points) < 100:
        print("边缘点太少，无法检测")
        extras = {
            'processed': processed,
            'edges': edges,
            'edge_points': edge_points,
            'hough_circles': [],
            'candidate_circles': None
        }
        return None, None, None, extras
    
    print(f"检测到 {len(edge_points)} 个边缘点")
    
    # 1. 估计圆心 (使用霍夫变换作为初始猜测)
    circles_hough = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=15,
        minRadius=int(min(img.shape[:2]) * 0.1),
        maxRadius=int(min(img.shape[:2]) * 0.8)
    )
    
    center_x, center_y = 0, 0
    if circles_hough is None:
        print("霍夫变换未找到任何圆，尝试使用边缘重心作为中心估计")
        center_y, center_x = np.mean(edge_points, axis=0)
    else:
        circles = circles_hough[0, :]
        k = min(len(circles), 5)
        center_x = np.median(circles[:k, 0])
        center_y = np.median(circles[:k, 1])
        print(f"估计圆心: ({center_x:.1f}, {center_y:.1f})")

    # 准备霍夫变换检测到的圆列表，传给可视化函数
    hough_circles = []
    if circles_hough is not None:
        for c in circles_hough[0]:
            hough_circles.append((float(c[0]), float(c[1]), float(c[2])))

    # 2. 径向直方图分析 (利用同心圆特性)
    distances = np.sqrt((edge_points[:, 1] - center_x)**2 + (edge_points[:, 0] - center_y)**2)
    
    max_dist = np.max(distances)
    hist, bin_edges = np.histogram(distances, bins=int(max_dist/2))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    hist_smooth = ndimage.gaussian_filter1d(hist, sigma=2)
    
    peaks, _ = find_peaks(hist_smooth, height=np.max(hist_smooth)*0.2, distance=10)
    
    outer_circle = None
    inner_circle = None
    
    if len(peaks) >= 2:
        print(f"检测到 {len(peaks)} 个潜在半径峰值: {bin_centers[peaks]}")
        
        candidate_circles = []
        for peak_idx in peaks:
            radius_guess = bin_centers[peak_idx]
            mask = np.abs(distances - radius_guess) < 10
            peak_points = edge_points[mask]
            
            if len(peak_points) < 50:
                continue
                
            result = ransac_circle_fit(
                peak_points,
                max_iterations=2000,
                threshold=3.0,
                min_inliers_ratio=0.3
            )
            
            if result is not None:
                xc, yc, r, inliers = result
                dist_to_center = np.sqrt((xc - center_x)**2 + (yc - center_y)**2)
                if dist_to_center < 30: # 允许一定的中心偏差
                    score = np.sum(inliers)
                    candidate_circles.append({
                        'params': (xc, yc, r),
                        'score': score
                    })
        
        candidate_circles.sort(key=lambda x: x['score'], reverse=True)
        
        if len(candidate_circles) >= 2:
            # 筛选半径差异明显的两个圆
            final_circles = [candidate_circles[0]]
            for cand in candidate_circles[1:]:
                r1 = final_circles[0]['params'][2]
                r2 = cand['params'][2]
                if abs(r1 - r2) > 15:
                    final_circles.append(cand)
                    if len(final_circles) == 2:
                        break
            
            if len(final_circles) == 2:
                c1 = final_circles[0]['params']
                c2 = final_circles[1]['params']
                
                if c1[2] > c2[2]:
                    outer_circle = (int(c1[0]), int(c1[1]), int(c1[2]))
                    inner_circle = (int(c2[0]), int(c2[1]), int(c2[2]))
                else:
                    outer_circle = (int(c2[0]), int(c2[1]), int(c2[2]))
                    inner_circle = (int(c1[0]), int(c1[1]), int(c1[2]))
                print(f"最终结果 - 外圆半径: {outer_circle[2]}, 内圆半径: {inner_circle[2]}")
    else:
        print("未检测到足够的圆边缘峰值")

    # 可视化：只有在交互展示模式时调用
    if visualization:
        vis_save = outdir / "visualization.png" if outdir is not None else None
        visualize_results(img, edges, outer_circle, inner_circle, edge_points, hough_circles=hough_circles, save_path=vis_save, show=True)
    
    # 计算管壁厚度
    wall_thickness = None
    if outer_circle is not None and inner_circle is not None:
        wall_thickness = outer_circle[2] - inner_circle[2]
    
    extras = {
        'processed': processed,
        'edges': edges,
        'edge_points': edge_points,
        'hough_circles': hough_circles,
        'candidate_circles': candidate_circles if 'candidate_circles' in locals() else None
    }

    return outer_circle, inner_circle, wall_thickness, extras

def visualize_results(original, edges, outer_circle, inner_circle, edge_points, hough_circles=None, save_path=None, show=True):
    """可视化结果"""
    plt.figure(figsize=(20, 5))
    
    # 原始图像
    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # 边缘检测结果
    plt.subplot(1, 4, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Detection')
    plt.axis('off')
    
    # 边缘点分布及RANSAC拟合结果
    ax = plt.subplot(1, 4, 3)
    h, w = original.shape[:2]
    ax.scatter(edge_points[:, 1], edge_points[:, 0], s=1, c='blue', alpha=0.5)
    # 绘制霍夫检测到的圆（品红色，细线）
    if hough_circles:
        for hc in hough_circles:
            xc, yc, r = hc
            circ_h = plt.Circle((xc, yc), r, edgecolor='magenta', facecolor='none', linewidth=1.2, alpha=0.9)
            ax.add_patch(circ_h)
    if outer_circle is not None:
        circ = plt.Circle((outer_circle[0], outer_circle[1]), outer_circle[2],
                          edgecolor='green', facecolor='none', linewidth=2)
        ax.add_patch(circ)
    if inner_circle is not None:
        circ = plt.Circle((inner_circle[0], inner_circle[1]), inner_circle[2],
                          edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circ)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect('equal')
    ax.set_title('Edge Points & RANSAC Fit')
    ax.axis('off')
    
    # 计算壁厚并在子图与结果图上标注
    thickness = None
    if outer_circle is not None and inner_circle is not None:
        thickness = outer_circle[2] - inner_circle[2]
        ax.text(0.02, 0.06, f"Wall thickness: {thickness}px", transform=ax.transAxes,
                color='yellow', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))

    # 检测结果图
    ax4 = plt.subplot(1, 4, 4)
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR) if len(original.shape) == 2 else original.copy()

    if outer_circle is not None:
        cv2.circle(result, (outer_circle[0], outer_circle[1]), outer_circle[2], (0, 128, 0), 25)
        cv2.drawMarker(result, (outer_circle[0], outer_circle[1]), (0, 128, 0), markerType=cv2.MARKER_CROSS, markerSize=300, thickness=20)

    if inner_circle is not None:
        cv2.circle(result, (inner_circle[0], inner_circle[1]), inner_circle[2], (0, 0, 255), 25)
        cv2.drawMarker(result, (inner_circle[0], inner_circle[1]), (0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=300, thickness=20)
    
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Final Result (Green=Outer, Red=Inner)')
    plt.axis('off')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(str(save_path), bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close()

def calculate_pixel_to_mm(known_diameter_mm, measured_diameter_pixel):
    """计算像素到毫米的转换比例"""
    return known_diameter_mm / measured_diameter_pixel


def process_folder(input_path, output_root):
    """处理文件夹中的所有图像，保存中间结果与可视化，生成汇总 CSV

    输出结构:
      output_root/
        <image_path>/
          processed.png
          edges.png
          visualization.png
          result.jpg
        results.csv
    """
    input_path = Path(input_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    files = []
    if input_path.is_dir():
        exts = ('*.jpg',)
        for e in exts:
            files.extend(input_path.glob(e))
    elif input_path.is_file():
        files = [input_path]
    else:
        raise ValueError('输入路径不存在')

    results = []
    for f in sorted(files):
        print(f"处理: {f}")
        # 为每张图创建独立子目录
        per_image_out = output_root / f.stem
        res = process_single_image(f, per_image_out, show_visualization=False)
        if res is None:
            results.append({
                'filepath': str(f),
                'outer_x': '',
                'outer_y': '',
                'outer_r': '',
                'inner_x': '',
                'inner_y': '',
                'inner_r': '',
                'thickness': ''
            })
            continue

        outer, inner, thickness = res
        results.append({
            'filepath': str(f),
            'outer_x': outer[0] if outer else '',
            'outer_y': outer[1] if outer else '',
            'outer_r': outer[2] if outer else '',
            'inner_x': inner[0] if inner else '',
            'inner_y': inner[1] if inner else '',
            'inner_r': inner[2] if inner else '',
            'thickness': thickness if thickness is not None else ''
        })

    # 写 CSV
    csv_path = output_root / 'results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filepath', 'outer_x', 'outer_y', 'outer_r', 'inner_x', 'inner_y', 'inner_r', 'thickness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"批量处理完成。结果保存在: {output_root}")

def process_single_image(img_path, output_dir=None, show_visualization=False):
    """处理单张图像
    
    Args:
        img_path: 图像文件路径
        output_dir: 输出目录（可选）
        show_visualization: 是否显示可视化窗口
    
    Returns:
        (outer_circle, inner_circle, thickness) 或 None（如果失败）
    """
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"错误: 图像文件不存在: {img_path}")
        return None
    
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误: 无法读取图像: {img_path}")
        return None
    
    if output_dir is not None:
        outdir = Path(output_dir)
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        outdir = None
    
    # 检测圆
    outer, inner, thickness, extras = detect_pipe_circles(
        img, 
        visualization=show_visualization,
        output_dir=outdir,
        save_intermediate=(outdir is not None)
    )
    
    # 输出结果
    print("\n" + "="*50)
    print("检测结果")
    print("="*50)
    
    if outer is not None:
        print(f"外圆: 中心({outer[0]}, {outer[1]}), 半径={outer[2]} pixels")
        print(f"      外径={outer[2] * 2} pixels")
    else:
        print("未检测到外圆")
    
    if inner is not None:
        print(f"内圆: 中心({inner[0]}, {inner[1]}), 半径={inner[2]} pixels")
        print(f"      内径={inner[2] * 2} pixels")
    else:
        print("未检测到内圆")
    
    print(f"\n管壁厚度: {thickness} pixels")
    print("="*50)
    
    # 保存结果图像（如果处于展示模式则不保存）
    if outer is not None or inner is not None:
        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if outer is not None:
            cv2.circle(result_img, (outer[0], outer[1]), outer[2], (0, 128, 0), 4)
            cv2.drawMarker(result_img, (outer[0], outer[1]), (0, 128, 0), 
                          markerType=cv2.MARKER_CROSS, markerSize=300, thickness=4)
            cv2.putText(result_img, f"Outer R={outer[2]}", 
                       (outer[0]+outer[2]+10, outer[1]-100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 0), 3)
        if inner is not None:
            cv2.circle(result_img, (inner[0], inner[1]), inner[2], (0, 0, 255), 4)
            cv2.drawMarker(result_img, (inner[0], inner[1]), (0, 0, 255), 
                          cv2.MARKER_TILTED_CROSS, 300, 4)
            cv2.putText(result_img, f"Inner R={inner[2]}", 
                       (inner[0]+inner[2]+10, inner[1]+100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        if thickness is not None:
            cv2.putText(result_img, f"Wall Thickness={thickness}px", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        # 如果是展示模式，则跳过保存最终结果图片
        if not show_visualization:
            if outdir is not None:
                save_path = outdir / "result.jpg"
            else:
                save_path = Path("result.jpg")
            cv2.imwrite(str(save_path), result_img)
            print(f"\n结果已保存到: {save_path}")
            print("*" * 70)
    
    return outer, inner, thickness


def main():
    """主函数：解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(
        description='金属管道内外圆检测与壁厚测量工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单张图像（显示可视化窗口）
  python main.py -i 20pipe/2019_10_23_13_44_44.jpg --show
  
  # 处理单张图像并保存到指定目录
  python main.py -i 20pipe/2019_10_23_13_44_44.jpg -o output_2019_10_23_13_44_44/
  
  # 批量处理文件夹（不显示窗口）
  python main.py -i 20pipe/ -o output/ --batch
        """
    )
    
    parser.add_argument('-i', '--input', required=True,
                       help='输入图像文件路径或文件夹路径')
    parser.add_argument('-o', '--output', default=None,
                       help='输出目录（可选，默认保存到当前目录）')
    parser.add_argument('--batch', action='store_true',
                       help='批量处理模式（处理文件夹中所有图像）')
    parser.add_argument('--show', action='store_true',
                       help='显示可视化窗口（仅在单图模式下有效）')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 验证输入路径
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        sys.exit(1)
    
    # 判断处理模式
    if args.batch or input_path.is_dir():
        # 批量处理模式
        if not input_path.is_dir():
            print(f"错误: 批量模式需要输入文件夹路径")
            sys.exit(1)
        
        output_root = Path(args.output) if args.output else Path('output')
        print(f"批量处理模式")
        print(f"输入目录: {input_path}")
        print(f"输出目录: {output_root}")
        print("-" * 50)
        
        process_folder(input_path, output_root)
    else:
        # 单图处理模式
        if not input_path.is_file():
            print(f"错误: 输入路径不是有效的图像文件")
            sys.exit(1)
        
        print(f"单图处理模式")
        print(f"输入图像: {input_path}")
        if args.output:
            print(f"输出目录: {args.output}")
        print("-" * 50)
        
        process_single_image(input_path, args.output, show_visualization=args.show)


if __name__ == "__main__":
    main()