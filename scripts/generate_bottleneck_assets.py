import numpy as np
from PIL import Image, ImageDraw
import os

# 配置常量
WIDTH, HEIGHT = 1000, 1000
LANE_WIDTH_PX = 42.0
TOTAL_LEN = WIDTH

# 重新分配比例：
# 左右两端三车道各占 30% (300px)
# 过渡段各占 15% (150px)
# 中间瓶颈段占 10% (100px)
X1 = 300.0
TRANS_LEN = 150.0
BNECK_LEN = 100.0

X2 = X1 + TRANS_LEN
X3 = X2 + BNECK_LEN
X4 = X3 + TRANS_LEN

def _piecewise_boundaries(x, x1, x2, x3, x4, full_upper, mid_upper, mid_lower, full_lower):
    """返回给定 x 处的(upper_outer, upper_inner, lower_inner, lower_outer).

    - 三车道段: outer=full, inner=mid
    - 单车道段: outer=inner=mid
    - 过渡段: outer 在 full 与 mid 之间线性插值
    """
    if x <= x1:
        upper_outer = full_upper
        lower_outer = full_lower
    elif x <= x2:
        t = (x - x1) / max(1.0, (x2 - x1))
        upper_outer = full_upper + t * (mid_upper - full_upper)
        lower_outer = full_lower + t * (mid_lower - full_lower)
    elif x <= x3:
        upper_outer = mid_upper
        lower_outer = mid_lower
    elif x <= x4:
        t = (x - x3) / max(1.0, (x4 - x3))
        upper_outer = mid_upper + t * (full_upper - mid_upper)
        lower_outer = mid_lower + t * (full_lower - mid_lower)
    else:
        upper_outer = full_upper
        lower_outer = full_lower

    upper_inner = mid_upper
    lower_inner = mid_lower
    return upper_outer, upper_inner, lower_inner, lower_outer


def _sample_polyline(x_start, x_end, x1, x2, x3, x4, full_upper, mid_upper, mid_lower, full_lower, n=200):
    xs = np.linspace(x_start, x_end, n)
    upper_outer = []
    lower_outer = []
    for x in xs:
        uo, _, _, lo = _piecewise_boundaries(x, x1, x2, x3, x4, full_upper, mid_upper, mid_lower, full_lower)
        upper_outer.append((float(x), float(uo)))
        lower_outer.append((float(x), float(lo)))
    return upper_outer, lower_outer


def generate_bottleneck(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    yc = HEIGHT // 2
    lw = LANE_WIDTH_PX
    
    # 使用统一的全局常量 X1, X2, X3, X4
    x1, x2, x3, x4 = X1, X2, X3, X4
    
    # Y 坐标（中间车道的上下边界）
    mid_upper = yc - 0.5 * lw
    mid_lower = yc + 0.5 * lw
    
    # 完整三车道的上下边界
    full_upper = yc - 1.5 * lw
    full_lower = yc + 1.5 * lw

    # 1. drivable 区域
    drivable = Image.new('L', (WIDTH, HEIGHT), 0)
    d = ImageDraw.Draw(drivable)

    # 绘制中间车道（始终存在）
    d.rectangle([0, mid_upper, WIDTH, mid_lower], fill=255)

    # 绘制两侧车道的收缩与扩张
    # 上侧车道 (Lane 0)
    d.polygon([
        (0, full_upper), (x1, full_upper), (x2, mid_upper), 
        (x3, mid_upper), (x4, full_upper), (WIDTH, full_upper),
        (WIDTH, mid_upper), (0, mid_upper)
    ], fill=255)
    
    # 下侧车道 (Lane 2)
    d.polygon([
        (0, mid_lower), (WIDTH, mid_lower), (WIDTH, full_lower),
        (x4, full_lower), (x3, mid_lower), (x2, mid_lower),
        (x1, full_lower), (0, full_lower)
    ], fill=255)

    drivable.save(os.path.join(out_dir, 'drivable.png'))

    # 2. lane_dashes (虚线)
    dashes = Image.new('L', (WIDTH, HEIGHT), 0)
    dd = ImageDraw.Draw(dashes)
    
    def draw_dashed(p1, p2):
        dist = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
        if dist < 1: return
        dx, dy = (p2[0]-p1[0])/dist, (p2[1]-p1[1])/dist
        curr = 0
        while curr < dist:
            s = (p1[0]+dx*curr, p1[1]+dy*curr)
            e = (p1[0]+dx*min(curr+15, dist), p1[1]+dy*min(curr+15, dist))
            dd.line([s, e], fill=255, width=2)
            curr += 30

    # 只有在三车道区域画虚线
    draw_dashed((0, mid_upper), (x1, mid_upper))
    draw_dashed((0, mid_lower), (x1, mid_lower))
    draw_dashed((x4, mid_upper), (WIDTH, mid_upper))
    draw_dashed((x4, mid_lower), (WIDTH, mid_lower))
    
    dashes.save(os.path.join(out_dir, 'lane_dashes.png'))

    # 4. lane_id
    lane_id = Image.new('L', (WIDTH, HEIGHT), 0)
    lid = ImageDraw.Draw(lane_id)

    # 用采样折线保证过渡段严格连续
    uo_poly, lo_poly = _sample_polyline(0, WIDTH, x1, x2, x3, x4, full_upper, mid_upper, mid_lower, full_lower, n=400)

    # Lane 1 (Middle): ID=1（全程存在）
    lid.polygon([(0, mid_upper), (WIDTH, mid_upper), (WIDTH, mid_lower), (0, mid_lower)], fill=1)

    # Lane 0 (Top): ID=2（上外边界到中线）
    lid.polygon(uo_poly + [(WIDTH, mid_upper), (0, mid_upper)], fill=2)

    # Lane 2 (Bottom): ID=3（中线到下外边界）
    lid.polygon([(0, mid_lower), (WIDTH, mid_lower)] + lo_poly[::-1], fill=3)

    lane_id.save(os.path.join(out_dir, 'lane_id.png'))
    
    # 5. yellowline (占位：全黑图，C++ 后端仍会要求加载)
    yellowline = Image.new('L', (WIDTH, HEIGHT), 0)
    yellowline.save(os.path.join(out_dir, 'yellowline.png'))

    # 6. 可视化预览图 (用于 README 展示)
    preview = Image.merge("RGB", (drivable, dashes, lane_id))
    preview.save(os.path.join(out_dir, 'preview.png'))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(base_dir, 'scenarios', 'bottleneck')
    generate_bottleneck(target)
    print(f"Bottleneck assets generated in: {target}")
