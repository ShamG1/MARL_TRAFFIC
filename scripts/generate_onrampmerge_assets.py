import numpy as np
from PIL import Image, ImageDraw
import os

WIDTH, HEIGHT = 1000, 1000
LANE_WIDTH_PX = 42.0


def draw_dashed_line(draw, p1, p2, color, width=2, dash_len=20, gap_len=20):
    dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if dist < 1e-6:
        return
    dx = (p2[0] - p1[0]) / dist
    dy = (p2[1] - p1[1]) / dist
    curr = 0.0
    while curr < dist:
        start = (p1[0] + dx * curr, p1[1] + dy * curr)
        end_dist = min(curr + dash_len, dist)
        end = (p1[0] + dx * end_dist, p1[1] + dy * end_dist)
        draw.line([start, end], fill=color, width=width)
        curr += dash_len + gap_len


def generate_onramp(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    yc = HEIGHT // 2
    lw = LANE_WIDTH_PX

    # 坐标定义（左侧双车道 → 匝道汇入后三车道 → 掉道回双车道）
    main_upper = yc - 1.5 * lw      # 最上边界（左侧车道上边缘）
    main_lower = yc + 0.5 * lw      # 主路下边界（中间车道下边缘）
    added_lower = yc + 1.5 * lw     # 临时第三车道下边界
    y_div_main = yc - 0.5 * lw      # 主路内部车道分隔线（一直存在）
    y_div_added = yc + 0.5 * lw     # 临时车道分隔线（匝道+三车道段）

    # 匝道（严格固定宽度 lw，无缝接入）
    merge_start_x = 400
    drop_x = 800
    
    # 中心线 y = k*x + b
    # 起点 (x=0, y=yc+3.8*lw), 终点 (x=400, y=yc+1.0*lw)
    # 斜率 k = -2.8 * lw / 400 = -0.007 * lw
    def get_ramp_y_center(x):
        return (yc + 3.8 * lw) + (-0.007 * lw) * x

    # 1. drivable 区域（可行驶mask）
    drivable = Image.new('L', (WIDTH, HEIGHT), 0)
    d = ImageDraw.Draw(drivable)

    # 主路部分
    d.rectangle([0, main_upper, merge_start_x, main_lower], fill=255)
    d.rectangle([merge_start_x, main_upper, drop_x, added_lower], fill=255)
    d.rectangle([drop_x, main_upper, WIDTH, main_lower], fill=255)

    # 匝道部分：通过沿中心线偏移绘制，确保宽度严格为 lw
    # 构造平行四边形/多边形，边界严格平行于中心线
    # 上边界: get_ramp_y_center(x) - 0.5 * lw
    # 下边界: get_ramp_y_center(x) + 0.5 * lw
    d.polygon([
        (0, get_ramp_y_center(0) - 0.5 * lw),
        (0, get_ramp_y_center(0) + 0.5 * lw),
        (merge_start_x, added_lower),
        (merge_start_x, main_lower)
    ], fill=255)

    drivable.save(os.path.join(out_dir, 'drivable.png'))

    # 2. yellowline
    yellow = Image.new('L', (WIDTH, HEIGHT), 0)
    yellow.save(os.path.join(out_dir, 'yellowline.png'))

    # 3. lane_dashes（虚线车道线）
    dashes = Image.new('L', (WIDTH, HEIGHT), 0)
    dd = ImageDraw.Draw(dashes)

    draw_dashed_line(dd, (0, y_div_main), (merge_start_x, y_div_main), 255)
    draw_dashed_line(dd, (drop_x, y_div_main), (WIDTH, y_div_main), 255)
    draw_dashed_line(dd, (merge_start_x, y_div_main), (drop_x, y_div_main), 255)
    draw_dashed_line(dd, (merge_start_x, y_div_added), (drop_x, y_div_added), 255)

    dashes.save(os.path.join(out_dir, 'lane_dashes.png'))

    # 4. lane_id
    lane_id = Image.new('L', (WIDTH, HEIGHT), 0)
    lid = ImageDraw.Draw(lane_id)

    # Lane 1 & 2
    lid.rectangle([0, main_upper, WIDTH, y_div_main], fill=1)
    lid.rectangle([0, y_div_main, merge_start_x, main_lower], fill=2)
    lid.rectangle([merge_start_x, y_div_main, WIDTH, main_lower], fill=2)

    # Lane 3 (Ramp + Merge segment)
    lid.polygon([
        (0, get_ramp_y_center(0) - 0.5 * lw),
        (0, get_ramp_y_center(0) + 0.5 * lw),
        (merge_start_x, added_lower),
        (merge_start_x, main_lower)
    ], fill=3)
    lid.rectangle([merge_start_x, main_lower, drop_x, added_lower], fill=3)

    lane_id.save(os.path.join(out_dir, 'lane_id.png'))


if __name__ == '__main__':
    # 示例保存路径（可自行修改）
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scenarios_dir = os.path.join(base_dir, 'scenarios')

    generate_onramp(os.path.join(scenarios_dir, 'onrampmerge_3lane'))