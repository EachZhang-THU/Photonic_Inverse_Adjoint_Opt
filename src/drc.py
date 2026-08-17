import os
import re
import subprocess
import xml.etree.ElementTree as ET

import klayout.db as db
import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull

from src import environment as env
from src import setting


def extract_drc_results(root):
    """提取 DRC 检查结果。"""
    results = []
    for item in root.findall(".//item"):
        category = item.find("category").text
        cell = item.find("cell").text
        values = item.find("values/value").text
        results.append({
            "category": category,
            "cell": cell,
            "values": values
        })
    return results


def classify_by_category(results):
    """按 category 分类。"""
    classified_results = {}
    for result in results:
        category = result['category']
        if category not in classified_results:
            classified_results[category] = []
        classified_results[category].append(result)
    return classified_results


def convert_line_segment_sides(index_py, x_min, x_max, y_min, y_max, x_points, y_points,
                               start_point, end_point, index):

    index_convert = index_py.copy()
    x0, y0 = start_point
    x1, y1 = end_point

    # 区域参数
    max_i = index_py.shape[1] - 1
    max_j = index_py.shape[0] - 1
    dx_pixel, dy_pixel = (x_max - x_min) / x_points, (y_max - y_min) / y_points

    def get_pixel_index(x, y):
        """将物理坐标转换为像素索引。"""
        i = int((x - x_min) // dx_pixel)
        j = int((y - y_min) // dy_pixel)
        return (
            max(0, min(i, max_i)),
            max(0, min(j, max_j))
        )

    # 获取起点终点所在像素
    i0, j0 = get_pixel_index(x0, y0)
    i1, j1 = get_pixel_index(x1, y1)

    # 处理单像素情况
    if i0 == i1 and j0 == j1:
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i0 + di, j0 + dj
                if 0 <= ni <= max_i and 0 <= nj <= max_j:
                    index_convert[nj, ni] = index
        return index_convert

    # Bresenham 算法参数初始化
    dx, dy = x1 - x0, y1 - y0
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0

    x_edge = x_min + ((i0 + 1) * dx_pixel if step_x > 0 else i0 * dx_pixel) if dx != 0 else None
    y_edge = y_min + ((j0 + 1) * dy_pixel if step_y > 0 else j0 * dy_pixel) if dy != 0 else None

    tMaxX = (x_edge - x0) / dx if dx != 0 else float('inf')
    tMaxY = (y_edge - y0) / dy if dy != 0 else float('inf')

    tDeltaX = dx_pixel / abs(dx) if dx != 0 else 0
    tDeltaY = dy_pixel / abs(dy) if dy != 0 else 0

    current_i, current_j = i0, j0

    # 遍历线段路径
    while True:
        # 标记当前像素的八邻域
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = current_i + di, current_j + dj
                if 0 <= ni <= max_i and 0 <= nj <= max_j:
                    index_convert[nj, ni] = index

        # 终止条件
        if current_i == i1 and current_j == j1:
            break

        # 步进逻辑
        if tMaxX < tMaxY:
            current_i += step_x
            if current_i < 0 or current_i > max_i:
                break
            tMaxX += tDeltaX
        else:
            current_j += step_y
            if current_j < 0 or current_j > max_j:
                break
            tMaxY += tDeltaY

        # 终止条件：到达终点像素
        if current_i == i1 and current_j == j1:
            break

        # 步进逻辑
        if tMaxX < tMaxY:
            current_i += step_x
            tMaxX += tDeltaX
        else:
            current_j += step_y
            tMaxY += tDeltaY

    return index_convert


def segment_distance(seg1, seg2):
    a = np.array(seg1[0])
    b = np.array(seg1[1])
    c = np.array(seg2[0])
    d = np.array(seg2[1])

    ab = b - a
    cd = d - c
    ac = c - a

    def dot(v1, v2):
        return np.dot(v1, v2)

    denominator = dot(ab, ab) * dot(cd, cd) - dot(ab, cd) ** 2

    if denominator == 0:
        dist = min(np.linalg.norm(a - c), np.linalg.norm(a - d),
                   np.linalg.norm(b - c), np.linalg.norm(b - d))
        return dist
    else:
        t = (dot(ac, ab) * dot(cd, cd) - dot(ac, cd) * dot(ab, cd)) / denominator
        s = (dot(ac, cd) * dot(ab, ab) - dot(ac, ab) * dot(ab, cd)) / denominator
        t = max(0, min(t, 1))
        s = max(0, min(s, 1))

        p = a + t * ab
        q = c + s * cd
        dist_pq = np.linalg.norm(p - q)

        dist_endpoints = min(np.linalg.norm(a - c), np.linalg.norm(a - d),
                             np.linalg.norm(b - c), np.linalg.norm(b - d))
        return min(dist_pq, dist_endpoints)


def single_pixel_process(x_points, y_points, x_min, x_max, y_min, y_max,
                         x1, x2, params, param_index):
    params_opt = params.copy()
    # 区域参数
    max_i = params.shape[1] - 1
    max_j = params.shape[0] - 1

    def get_pixel_index(x, y):
        dx_pixel, dy_pixel = (x_max - x_min) / x_points, (y_max - y_min) / y_points
        """将物理坐标转换为像素索引。"""
        i = int((x - x_min) // dx_pixel)
        j = int((y - y_min) // dy_pixel)
        return (
            max(0, min(i, max_i)),
            max(0, min(j, max_j))
        )

    # 获取起点终点所在像素
    i0, j0 = get_pixel_index(x1[0], x1[1])
    i1, j1 = get_pixel_index(x2[0], x2[1])
    # 处理单像素情况
    if i0 == i1 and j0 == j1:
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i0 + di, j0 + dj
                if 0 <= ni <= max_i and 0 <= nj <= max_j:
                    params_opt[nj, ni] = param_index
    return params_opt


def process_correct_index(params, x_min, x_max, y_min, y_max, x_points, y_points,
                          param_min, param_max, param_delta, x1, x2, y1, y2,
                          threshold, global_updated_mask):
    params_opt = params.copy()

    x_pos = np.linspace(x_min, x_max, x_points)
    y_pos = np.linspace(y_min, y_max, y_points)

    X, Y = np.meshgrid(x_pos, y_pos, indexing='xy')

    distance = segment_distance((x1, x2), (y1, y2))

    # 创建局部更新标记矩阵
    local_updated_mask = np.zeros_like(params_opt, dtype=bool)

    if distance < threshold:
        params_opt = single_pixel_process(x_points, y_points, x_min, x_max, y_min, y_max,
                                          x1, x2, params_opt, param_min)
        params_opt = single_pixel_process(x_points, y_points, x_min, x_max, y_min, y_max,
                                          y1, y2, params_opt, param_min)
        points = np.array([x1, x2, y1, y2])
        hull = ConvexHull(points)
        polygon = points[hull.vertices]
        points = np.vstack((X.ravel(), Y.ravel())).T
        path = Path(polygon)
        mask = path.contains_points(points).reshape(params_opt.shape)
        # 更新参数矩阵
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not global_updated_mask[i, j]:
                    if params_opt[i, j] > 0:
                        params_opt[i, j] = 0
                    else:
                        params_opt[i, j] -= param_delta
                    local_updated_mask[i, j] = True

    else:
        params_opt = single_pixel_process(x_points, y_points, x_min, x_max, y_min, y_max,
                                          x1, x2, params_opt, param_max)
        params_opt = single_pixel_process(x_points, y_points, x_min, x_max, y_min, y_max,
                                          y1, y2, params_opt, param_max)
        lines = [np.array([x1, x2]), np.array([y1, y2])]
        for line in lines:
            p1, p2 = line
            direction = p2 - p1
            direction = direction / np.linalg.norm(direction)
            normal = np.array([-direction[1], direction[0]])
            expand_dist = (x_max - x_min) / x_points
            expanded_line1 = line + normal * expand_dist * 2
            expanded_line2 = line - normal * expand_dist * 2
            polygon_points = np.array([expanded_line1[0], expanded_line1[1],
                                       expanded_line2[1], expanded_line2[0]])
            path = Path(polygon_points)
            points = np.vstack((X.ravel(), Y.ravel())).T
            mask = path.contains_points(points).reshape(params_opt.shape)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i, j] and not global_updated_mask[i, j]:
                        if params_opt[i, j] < 1:
                            params_opt[i, j] = 1
                        else:
                            params_opt[i, j] += param_delta
                        local_updated_mask[i, j] = True

    # 更新全局更新标记矩阵
    global_updated_mask |= local_updated_mask

    return params_opt, global_updated_mask


def dfm(cfg, region, obj, params, input_gds_path, drc_file_path, gds_contour_script):
    """调用 Lumerical 脚本实现仿真区域的轮廓提取。"""
    obj.model.eval(gds_contour_script)

    # 对提取结果进行修改，删除中间的空洞
    ly = db.Layout()
    ly.read(input_gds_path)
    top_cell = ly.top_cell()
    l1 = ly.layer(1, 0)
    r1 = db.Region(top_cell.begin_shapes_rec(l1))
    r2 = r1.merged(2)
    if len(r2) != 0:
        r3 = r1 - r2
        ly.clear_layer(l1)
        top_cell.shapes(l1).insert(r3)
    ly.write(input_gds_path)

    # 调用 KLayout 命令行工具运行 DRC 脚本
    drc_script_path = os.path.join(env.DRC_PATH, cfg.drc.drc_script)
    command = [
        cfg.drc.klayout_path.replace("\\", "/"),
        "-r",
        drc_script_path.replace("\\", "/"),
        "-i",
        input_gds_path.replace("\\", "/"),
    ]
    subprocess.run(command, check=True)

    # 读取 Klayout 软件生成的 DRC 报告，提取相应的数据
    tree = ET.parse(drc_file_path)
    root = tree.getroot()
    drc_results = extract_drc_results(root)
    classified_results = classify_by_category(drc_results)

    # 提取并存储每个类别的坐标
    classified_coordinates = {}
    for category, items in classified_results.items():
        coordinates_list = []
        for item in items:
            values_str = re.sub(r'^[^:]+:', '', item['values'])
            # 使用正则表达式提取坐标，忽略分隔符
            coordinates = re.findall(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", values_str)
            coordinates_list.append([(float(x), float(y)) for x, y in coordinates])
        classified_coordinates[category] = coordinates_list

    params_dfm = params
    global_updated_mask = np.zeros_like(params_dfm, dtype=bool)

    # 从 region 与 cfg 推导 DRC 修正所需参数
    x_min, x_max = -region.size_x / 2 / 1e-6, region.size_x / 2 / 1e-6
    y_min, y_max = -region.size_y / 2 / 1e-6, region.size_y / 2 / 1e-6
    x_points, y_points = region.x_points, region.y_points
    params_min, params_max = cfg.optimizer.params_min, cfg.optimizer.params_max
    param_delta = cfg.drc.param_delta
    threshold = cfg.drc.drc_threshold

    print("\n================================")
    if "'HM.2 Min. exclusion of HM with same layer violation'" in classified_coordinates:
        num_HM_2 = len(classified_coordinates["'HM.2 Min. exclusion of HM with same layer violation'"])
        print(f"HM.2 Min. exclusion of HM with same layer violation:{num_HM_2}")

        for i in range(num_HM_2):
            line1 = classified_coordinates["'HM.2 Min. exclusion of HM with same layer violation'"][i][
                    :2]  # 第一条线段的起点和终点
            line2 = classified_coordinates["'HM.2 Min. exclusion of HM with same layer violation'"][i][
                    2:]  # 第二条线段的起点和终点

            x1, x2 = line1[0], line1[1]
            y1, y2 = line2[0], line2[1]

            params_dfm, global_updated_mask = process_correct_index(
                params_dfm, x_min, x_max, y_min, y_max, x_points, y_points,
                params_max, params_min, param_delta, x1, x2, y1, y2,
                threshold, global_updated_mask)

    if "'HM.1 Min. width of HM violation'" in classified_coordinates:
        num_HM_1 = len(classified_coordinates["'HM.1 Min. width of HM violation'"])
        print(f"HM.1 Min. width of HM violation:{num_HM_1}")
        for i in range(num_HM_1):
            line1 = classified_coordinates["'HM.1 Min. width of HM violation'"][i][:2]
            line2 = classified_coordinates["'HM.1 Min. width of HM violation'"][i][2:]
            x1, x2 = line1[0], line1[1]
            y1, y2 = line2[0], line2[1]

            params_dfm, global_updated_mask = process_correct_index(
                params_dfm, x_min, x_max, y_min, y_max, x_points, y_points,
                params_min, params_max, param_delta, x1, x2, y1, y2,
                threshold, global_updated_mask)

    print("================================")

    return params_dfm, classified_coordinates


def run_drc_iteration(cfg, ws, region, obj, params, iteration):
    """执行一次 DRC 检测与修复，返回修复后的参数以及是否通过 DRC。"""

    # 构造本次迭代的 GDS 和 DRC 结果路径
    input_gds_path = os.path.join(ws.sub_path, f"{cfg.drc.gds_name}_{iteration}.gds")

    gds_contour_script = setting.load_from_lsf("gds_contour.lsf")

    gds_contour_script = gds_contour_script.replace(
        "f = gdsopen('contours.gds');",
        "f = gdsopen('{:s}');".format(input_gds_path.replace("\\", "/"))
    )
    drc_file_path = os.path.join(ws.sub_path, f"{cfg.drc.gds_name}_{iteration}_drc_result")

    # 执行 DRC 修复
    params_opt, classified_coordinates = dfm(cfg, region, obj, params,
                                             input_gds_path, drc_file_path, gds_contour_script)

    # 统计两类违例数量并写入日志
    hm_1 = "'HM.1 Min. width of HM violation'"
    hm_2 = "'HM.2 Min. exclusion of HM with same layer violation'"
    num_hm_1 = len(classified_coordinates.get(hm_1, []))
    num_hm_2 = len(classified_coordinates.get(hm_2, []))

    with open(os.path.join(ws.save_path, "dfm.txt"), "a") as f:
        f.write(f"Iteration {iteration}\t{num_hm_1}\t{num_hm_2}\n")

    # 判断是否已经通过 DRC，通过则保存当前仿真模型
    drc_passed = hm_1 not in classified_coordinates and hm_2 not in classified_coordinates
    if drc_passed:
        obj.model.save(os.path.join(ws.save_path, f"interation{iteration}"))

    return params_opt, drc_passed
