# 金属管道内外圆检测与壁厚测量工具

基于计算机视觉的金属管道内外圆自动检测与壁厚测量系统，使用霍夫变换和RANSAC算法进行圆拟合。

## 功能特性

- ✅ 自动检测金属管道的内外圆
- ✅ 精确计算管壁厚度
- ✅ 支持单张图像处理
- ✅ 支持批量文件夹处理
- ✅ 生成可视化结果图
- ✅ 导出CSV格式检测结果
- ✅ 保存中间处理步骤

## 环境要求

```bash
# Python 3.7+
pip install opencv-python numpy matplotlib scipy
```

## 使用方法

### 1. 单图处理（显示可视化窗口）

```bash
python main.py -i image.jpg --show
```

### 2. 单图处理（保存到指定目录）

```bash
python main.py -i image.jpg -o output/
```

### 3. 批量处理文件夹

```bash
python main.py -i input_folder/ -o output/ --batch
```

### 4. 递归处理所有子文件夹

```bash
python main.py -i input_folder/ -o output/ --batch --recursive
```

## 命令行参数

| 参数 | 说明 | 必需 |
|------|------|------|
| `-i, --input` | 输入图像文件或文件夹路径 | 是 |
| `-o, --output` | 输出目录（默认：当前目录或./output） | 否 |
| `--batch` | 启用批量处理模式 | 否 |
| `--recursive` | 递归搜索子文件夹（仅批量模式） | 否 |
| `--show` | 显示可视化窗口（仅单图模式） | 否 |
| `--save-intermediate` | 保存中间处理结果 | 否 |

## 输出结构

### 单图处理输出

```
output/
  <image_name>/
    processed.png          # 预处理后的图像
    edges.png              # 边缘检测结果
    visualization.png      # 可视化图（包含检测过程）
    result_pipe_detection.jpg  # 最终结果标注图
```

### 批量处理输出

```
output/
  <image1>/
    processed.png
    edges.png
    visualization.png
    result_pipe_detection.jpg
  <image2>/
    ...
  results.csv              # 汇总所有检测结果
```

### CSV结果格式

| 列名 | 说明 |
|------|------|
| filename | 文件名 |
| outer_x, outer_y | 外圆圆心坐标 |
| outer_r | 外圆半径（像素） |
| inner_x, inner_y | 内圆圆心坐标 |
| inner_r | 内圆半径（像素） |
| thickness | 管壁厚度（像素） |

## 算法流程

1. **图像预处理**：双边滤波 → 中值滤波 → 形态学操作
2. **边缘检测**：自适应Canny边缘检测 → 连通域过滤
3. **圆心估计**：霍夫圆变换初始估计
4. **径向分析**：直方图峰值检测识别候选半径
5. **RANSAC拟合**：对每个候选半径进行鲁棒圆拟合
6. **结果筛选**：选择得分最高且半径差异明显的两个圆

## 示例

```bash
# 处理单张图片并查看结果
python main.py -i 20pipe/image.jpg --show

# 批量处理整个文件夹
python main.py -i 20pipe/ -o results/ --batch

# 递归处理所有图片
python main.py -i dataset/ -o output/ --batch --recursive
```

## 注意事项

- 输入图像应为清晰的金属管道截面图
- 支持的图像格式：PNG, JPG, JPEG, BMP, TIF, TIFF
- 建议图像分辨率不低于 800x800
- 管道应占据图像的主要部分
- 批量处理时会自动跳过无法读取的文件

## 故障排查

**问题：检测不到圆**
- 检查图像质量和对比度
- 调整霍夫变换参数（`minRadius`, `maxRadius`）
- 确保管道边缘清晰

**问题：检测到的圆不准确**
- 增加边缘点数量（调低Canny阈值）
- 调整RANSAC参数（`threshold`, `max_iterations`）
- 检查图像中是否有干扰物

## 许可证

MIT License
