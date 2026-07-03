from fmr.data.regions import Region


def test_iou_identical():
    r = Region(0.1, 0.1, 0.5, 0.5)
    assert abs(r.iou(r) - 1.0) < 1e-9


def test_iou_disjoint():
    a = Region(0.0, 0.0, 0.2, 0.2)
    b = Region(0.5, 0.5, 0.9, 0.9)
    assert a.iou(b) == 0.0


def test_iou_known_overlap():
    # Two unit-half boxes overlapping in a quarter: inter=0.25*0.25... compute:
    a = Region(0.0, 0.0, 0.5, 0.5)     # area 0.25
    b = Region(0.25, 0.25, 0.75, 0.75)  # area 0.25, inter = 0.25^2 = 0.0625
    expect = 0.0625 / (0.25 + 0.25 - 0.0625)
    assert abs(a.iou(b) - expect) < 1e-9


def test_coordinate_normalization():
    r = Region(0.9, 0.8, 0.1, 0.2)  # reversed corners
    assert r.x0 <= r.x1 and r.y0 <= r.y1
    assert r.area > 0


def test_grid_cell_partition():
    # 4x4 grid cells tile the unit square: areas sum to 1, no pairwise overlap.
    cells = [Region.from_grid_cell(r, c, 4, 4) for r in range(4) for c in range(4)]
    assert abs(sum(c.area for c in cells) - 1.0) < 1e-9
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            assert a.iou(b) < 1e-9
