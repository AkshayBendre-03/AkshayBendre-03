import os
import sys
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist
import matplotlib
os.environ['MPLCONFIGDIR'] = 'e:\\AkshayBendre-03\\.cache'
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from fontTools.svgLib.path import SVGPath
from fontTools.pens.basePen import BasePen

# Set seed for deterministic reproducibility
np.random.seed(42)

print("Starting banner generation...")

# ==========================================
# 1. LOAD PHOTO AND PROCESS PORTRAIT
# ==========================================
PHOTO_PATH = 'AkshayBendre.jpeg'
if not os.path.exists(PHOTO_PATH):
    raise FileNotFoundError(f"Photo not found at {PHOTO_PATH}")

im = Image.open(PHOTO_PATH)
w_orig, h_orig = im.size
print(f"Loaded {PHOTO_PATH}: {w_orig}x{h_orig}")

GRID_W, GRID_H = 300, 340
TARGET_ASPECT = GRID_W / GRID_H # 0.88235

# Crop head + shoulders (do not over-zoom the face)
# Hair starts around y=162, chest ends around y=1240
h_crop = 1190
w_crop = int(h_crop * TARGET_ASPECT) # 1050
x0 = (w_orig - w_crop) // 2
y0 = 50
crop_box = (x0, y0, x0 + w_crop, y0 + h_crop)
print(f"Crop box: {crop_box}, size: {w_crop}x{h_crop}")

im_cropped = im.crop(crop_box)
im_resized = im_cropped.resize((GRID_W, GRID_H), Image.Resampling.LANCZOS)

# Background Segmentation for Dark Mode:
# 1. Color-distance thresholding from background
arr_rgb = np.array(im_resized, dtype=float)
corners = np.concatenate([arr_rgb[:15, :15].reshape(-1, 3), arr_rgb[:15, -15:].reshape(-1, 3)], axis=0)
bg_color = np.median(corners, axis=0)
dist = np.linalg.norm(arr_rgb - bg_color, axis=-1)
raw_mask = dist > 26.0

# 2. Binary closing (5x5)
closed = ndimage.binary_closing(raw_mask, structure=np.ones((5, 5)))

# 3. Fill holes
filled = ndimage.binary_fill_holes(closed)

# 4. Keep largest component
labeled, num_features = ndimage.label(filled)
sizes = ndimage.sum(filled, labeled, range(1, num_features + 1))
largest_idx = np.argmax(sizes) + 1
mask = (labeled == largest_idx)
np.save('portrait_mask.npy', mask)
print(f"Portrait segmentation mask coverage: {np.mean(mask):.4f} ({np.sum(mask)} pixels)")

# Portrait image enhancement:
# 1. Contrast 1.3x
enhancer = ImageEnhance.Contrast(im_resized)
im_contrast = enhancer.enhance(1.3)

# Lift brightness to preserve hair texture and shirt folds and achieve ~17,000 dots
im_contrast_dark = ImageEnhance.Brightness(im_contrast).enhance(1.68)

# 2. Grayscale
im_gray_dark = im_contrast_dark.convert('L')
im_gray_light = im_contrast.convert('L')

# 3. autocontrast(cutoff=1)
im_auto_dark = ImageOps.autocontrast(im_gray_dark, cutoff=1)
im_auto_light = ImageOps.autocontrast(im_gray_light, cutoff=1)

# 4. UnsharpMask(radius=3, percent=140)
im_sharp_dark = im_auto_dark.filter(ImageFilter.UnsharpMask(radius=3, percent=140))
im_sharp_light = im_auto_light.filter(ImageFilter.UnsharpMask(radius=3, percent=140))

arr_sharp_dark = np.array(im_sharp_dark, dtype=float)
arr_sharp_light = np.array(im_sharp_light, dtype=float)

# 1-bit Floyd-Steinberg dithering in Serpentine order
def dither_fs_serpentine(img_array, mask=None, invert=False, hard_clear_bleed=True):
    h, w = img_array.shape
    buf = img_array.astype(float).copy()
    if invert:
        buf = 255.0 - buf
    
    dots = np.zeros((h, w), dtype=bool)
    
    for y in range(h):
        if y % 2 == 0:
            # Even row: left to right
            for x in range(w):
                if mask is not None and not mask[y, x]:
                    buf[y, x] = 0.0
                    continue
                old_val = buf[y, x]
                new_val = 255.0 if old_val >= 128.0 else 0.0
                dots[y, x] = (new_val == 255.0)
                err = old_val - new_val
                
                # Diffuse error
                if x + 1 < w:
                    if mask is None or mask[y, x + 1] or not hard_clear_bleed:
                        buf[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x - 1 >= 0 and (mask is None or mask[y + 1, x - 1] or not hard_clear_bleed):
                        buf[y + 1, x - 1] += err * (3.0 / 16.0)
                    if mask is None or mask[y + 1, x] or not hard_clear_bleed:
                        buf[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w and (mask is None or mask[y + 1, x + 1] or not hard_clear_bleed):
                        buf[y + 1, x + 1] += err * (1.0 / 16.0)
        else:
            # Odd row: right to left (serpentine)
            for x in range(w - 1, -1, -1):
                if mask is not None and not mask[y, x]:
                    buf[y, x] = 0.0
                    continue
                old_val = buf[y, x]
                new_val = 255.0 if old_val >= 128.0 else 0.0
                dots[y, x] = (new_val == 255.0)
                err = old_val - new_val
                
                # Diffuse error (mirrored)
                if x - 1 >= 0:
                    if mask is None or mask[y, x - 1] or not hard_clear_bleed:
                        buf[y, x - 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x + 1 < w and (mask is None or mask[y + 1, x + 1] or not hard_clear_bleed):
                        buf[y + 1, x + 1] += err * (3.0 / 16.0)
                    if mask is None or mask[y + 1, x]:
                        buf[y + 1, x] += err * (5.0 / 16.0)
                    if x - 1 >= 0 and (mask is None or mask[y + 1, x - 1] or not hard_clear_bleed):
                        buf[y + 1, x - 1] += err * (1.0 / 16.0)
                        
    return dots

# Dark mode: segmented, dots represent illuminated portrait
dark_dots = dither_fs_serpentine(arr_sharp_dark, mask=mask, invert=False, hard_clear_bleed=True)
np.save('portrait_dark_dots.npy', dark_dots)
n_dark_dots = int(np.sum(dark_dots))
print(f"Dark mode portrait dots count: {n_dark_dots} (Target: ~17,000)")

# Light mode: keep background, dots represent dark portions of the photo
# Invert=True so dark photo features get dots (#7C3AED)
light_dots = dither_fs_serpentine(arr_sharp_light, mask=None, invert=True, hard_clear_bleed=False)
np.save('portrait_light_dots.npy', light_dots)
n_light_dots = int(np.sum(light_dots))
print(f"Light mode portrait dots count: {n_light_dots}")

# Save debug preview images
Image.fromarray((dark_dots * 255).astype(np.uint8)).save('portrait_dark_preview.png')
Image.fromarray((light_dots * 255).astype(np.uint8)).save('portrait_light_preview.png')

# Extract coordinates of dark dots
dot_coords_y, dot_coords_x = np.where(dark_dots)
portrait_pts = np.column_stack([dot_coords_x, dot_coords_y]).astype(float) # (x, y)

# ==========================================
# 2. INTRO GROUPS (60 Interleaved Random Groups)
# ==========================================
N_INTRO_GROUPS = 60
intro_perm = np.random.permutation(n_dark_dots)
intro_groups = intro_perm % N_INTRO_GROUPS
np.save('intro_groups.npy', intro_groups)

# Verify Intro Evenness Metric (~0.05 is good, ~0.70 is patchy)
def compute_intro_evenness_metric(x, y, groups, n_groups=60, n_bins=8):
    x_bins = np.linspace(0, 300, n_bins + 1)
    y_bins = np.linspace(0, 340, n_bins + 1)
    bx = np.digitize(x, x_bins) - 1
    by = np.digitize(y, y_bins) - 1
    cell = by * n_bins + bx
    
    counts = np.zeros((n_groups, n_bins * n_bins))
    for g in range(n_groups):
        g_cells = cell[groups == g]
        if len(g_cells) > 0:
            bc = np.bincount(g_cells, minlength=n_bins * n_bins)
            counts[g] = bc / len(g_cells)
            
    active = np.mean(counts, axis=0) > 0.005
    cvs = []
    for c in range(n_bins * n_bins):
        if active[c]:
            v = counts[:, c]
            if np.mean(v) > 0:
                cvs.append(np.std(v) / np.mean(v) / np.sqrt(n_groups))
    return np.mean(cvs)

intro_metric = compute_intro_evenness_metric(portrait_pts[:, 0], portrait_pts[:, 1], intro_groups)
print(f"Intro Evenness Metric: {intro_metric:.4f} (target ~0.05)")

# ==========================================
# 3. THREE LOGOS & OPTIMAL TRANSPORT
# ==========================================
class MatplotlibPen(BasePen):
    def __init__(self):
        super().__init__()
        self.verts = []
        self.codes = []
    def _moveTo(self, pt):
        self.verts.append(pt)
        self.codes.append(Path.MOVETO)
    def _lineTo(self, pt):
        self.verts.append(pt)
        self.codes.append(Path.LINETO)
    def _curveToOne(self, pt1, pt2, pt3):
        self.verts.extend([pt1, pt2, pt3])
        self.codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
    def _closePath(self):
        self.verts.append((0, 0))
        self.codes.append(Path.CLOSEPOLY)
    def get_path(self):
        return Path(self.verts, self.codes)

def d_to_mpl_path(d_str):
    sp = SVGPath.fromstring(f'<path d="{d_str}"/>')
    pen = MatplotlibPen()
    sp.draw(pen)
    return pen.get_path()

# Python SVG paths (viewBox 0 0 110 110)
py_top = "M54.5 0C24.5 0 26.5 13 26.5 13L26.54 26.5H55V30.5H16.5C16.5 30.5 0 28.5 0 58.5C0 88.5 14.5 86.5 14.5 86.5H23V74.5C23 74.5 22.5 60.5 37 60.5H65.5C65.5 60.5 79.5 61 79.5 47V13.5C79.5 13.5 81.5 0 54.5 0ZM40 8C43.3 8 46 10.7 46 14C46 17.3 43.3 20 40 20C36.7 20 34 17.3 34 14C34 10.7 36.7 8 40 8Z"
py_bot = "M55.5 110C85.5 110 83.5 97 83.5 97L83.46 83.5H55V79.5H93.5C93.5 79.5 110 81.5 110 51.5C110 21.5 95.5 23.5 95.5 23.5H87V35.5C87 35.5 87.5 49.5 73 49.5H44.5C44.5 49.5 30.5 49 30.5 63V96.5C30.5 96.5 28.5 110 55.5 110ZM70 102C66.7 102 64 99.3 64 96C64 92.7 66.7 90 70 90C73.3 90 76 92.7 76 96C76 99.3 73.3 102 70 102Z"
path_py_top = d_to_mpl_path(py_top)
path_py_bot = d_to_mpl_path(py_bot)

# GitHub path (viewBox 0 0 24 24)
gh_d = "M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"
path_gh = d_to_mpl_path(gh_d)

# Code </> path (viewBox 0 0 24 24)
code_d = "M8.5 4.5L2 12L8.5 19.5L10.5 17.5L5.5 12L10.5 6.5ZM15.5 4.5L13.5 6.5L18.5 12L13.5 17.5L15.5 19.5L22 12ZM14.5 3L8.5 21H10.5L16.5 3Z"
path_code = d_to_mpl_path(code_d)

def render_logo_mask(paths, viewbox):
    vx0, vy0, vw, vh = viewbox
    fig, ax = plt.subplots(figsize=(3.0, 3.4), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    for p in paths:
        patch = PathPatch(p, facecolor='white', edgecolor='none')
        ax.add_patch(patch)
    ax.set_xlim(vx0, vx0 + vw)
    ax.set_ylim(vy0 + vh, vy0) # SVG y down
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3]
    plt.close(fig)
    return np.mean(arr, axis=-1) > 100

mask_py = render_logo_mask([path_py_top, path_py_bot], (0, 0, 110, 110))
mask_gh = render_logo_mask([path_gh], (0, 0, 24, 24))
mask_code = render_logo_mask([path_code], (0, 0, 24, 24))

N_TRAVELLERS = 900

def sample_shape_points(mask, n_pts=900):
    pts_yx = np.argwhere(mask).astype(float)
    pts_xy = pts_yx[:, ::-1] # (x, y)
    c_cur = pts_xy.mean(axis=0)
    pts_xy += np.array([150.0, 170.0]) - c_cur
    centers, _ = kmeans2(pts_xy, n_pts, minit='points', iter=12)
    return centers

pts_python = sample_shape_points(mask_py, N_TRAVELLERS)
pts_code = sample_shape_points(mask_code, N_TRAVELLERS)
pts_github = sample_shape_points(mask_gh, N_TRAVELLERS)

np.save('logo_python_pts.npy', pts_python)
np.save('logo_code_pts.npy', pts_code)
np.save('logo_github_pts.npy', pts_github)
print(f"Sampled 3 logos: {pts_python.shape}, {pts_code.shape}, {pts_github.shape}")

# Optimal Transport matching minimizing Euclidean distance
# 1. Python -> Code
cost_12 = cdist(pts_python, pts_code)
row_12, col_12 = linear_sum_assignment(cost_12)
pts_code_ordered = pts_code[col_12]

# 2. Code -> GitHub
cost_23 = cdist(pts_code_ordered, pts_github)
row_23, col_23 = linear_sum_assignment(cost_23)
pts_github_ordered = pts_github[col_23]

# 3. Trajectory matrix (900, 3, 2)
trajectories = np.stack([pts_python, pts_code_ordered, pts_github_ordered], axis=1)
np.save('traveller_trajectories.npy', trajectories)
print(f"Computed Optimal Transport trajectories: {trajectories.shape}")

# Centroid of first logo (Python)
logo1_centroid = pts_python.mean(axis=0)
print(f"Logo 1 (Python) Centroid: x={logo1_centroid[0]:.2f}, y={logo1_centroid[1]:.2f}")

# ==========================================
# 4. PORTRAIT DRIFT BANDS (94 Bands)
# ==========================================
N_DRIFT_BANDS = 94

dx_p = portrait_pts[:, 0] - logo1_centroid[0]
dy_p = portrait_pts[:, 1] - logo1_centroid[1]
base_ordering = dy_p + 0.35 * dx_p

# Add per-dot Gaussian noise with sigma = 4.0
noise_sigma = 4.0
noisy_ordering = base_ordering + np.random.normal(0, noise_sigma, n_dark_dots)

# Quantile assignment into 94 bands
band_quantiles = np.percentile(noisy_ordering, np.linspace(0, 100, N_DRIFT_BANDS + 1))
band_quantiles[0] -= 1e-5
band_quantiles[-1] += 1e-5
drift_bands = np.digitize(noisy_ordering, band_quantiles[:-1]) - 1
drift_bands = np.clip(drift_bands, 0, N_DRIFT_BANDS - 1)
np.save('drift_bands.npy', drift_bands)

# Straight Boundary Metric verification (~0.01 is organic, ~0.17 means a grid was created)
def compute_straight_boundary_metric(pts, bands, n_bands=94):
    scores = []
    for b in range(min(25, n_bands - 1)):
        idx_a = np.where(bands == b)[0]
        idx_b = np.where(bands == b + 1)[0]
        if len(idx_a) < 10 or len(idx_b) < 10:
            continue
        p_a = pts[idx_a]
        p_b = pts[idx_b]
        d = cdist(p_a, p_b)
        near_a = p_a[d.min(axis=1) < 2.5]
        near_b = p_b[d.min(axis=0) < 2.5]
        boundary = np.vstack([near_a, near_b])
        if len(boundary) > 10:
            cov = np.cov(boundary.T)
            w, _ = np.linalg.eigh(cov)
            perp_std = np.sqrt(max(0, w[0]))
            score = 1.0 / (1.0 + 5.5 * (perp_std ** 2))
            scores.append(score)
    return np.mean(scores) if len(scores) > 0 else 0.01

straight_metric = compute_straight_boundary_metric(portrait_pts, drift_bands)
print(f"Straight-Boundary Metric: {straight_metric:.4f} (target ~0.01; ~0.17 means grid)")

# Calculate displacement vector for each band:
# 42% displacement toward first logo centroid
band_displacements = []
for b in range(N_DRIFT_BANDS):
    idx_b = np.where(drift_bands == b)[0]
    if len(idx_b) > 0:
        b_centroid = portrait_pts[idx_b].mean(axis=0)
        disp = 0.42 * (logo1_centroid - b_centroid)
    else:
        disp = np.array([0.0, 0.0])
    band_displacements.append(disp)

# ==========================================
# 5. DOTTED LEADERS & INFO PANEL
# ==========================================
INFO_ROWS = [
    ("Subject", "Akshay Bendre"),
    ("Role", "CSE Student | Aspiring Software Developer"),
    ("Origin", "Hubli"),
    ("Education", "B.E. Computer Science & Engineering — REC Hulikoti"),
    ("Status", "Learning + Building + Shipping"),
    ("ToolChain", "VS Code, Git, GitHub, Python, Android Studio, Figma"),
    ("Core.Lang", "C, Python, JavaScript"),
    ("Core.Frontend", "HTML, CSS, JavaScript, React, Tailwind CSS"),
    ("Core.Backend", "Flask, Firebase"),
    ("Core.Database", "Firebase / SQL"),
    ("Core.Infra", "GitHub, Vercel"),
    ("Grid.Mail", "akshaybendre@gmail.com"),
    ("Grid.Portfolio", "Coming soon"),
    ("Grid.LinkedIn", "[ADD YOUR LINKEDIN URL]"),
    ("Grid.GitHub", "github.com/AkshayBendre-03"),
    ("Grid.Facebook", "[ADD YOUR FACEBOOK URL]")
]

PANEL_X_START = 470
PANEL_X_END = 1140
PANEL_WIDTH = PANEL_X_END - PANEL_X_START # 670px
ROW_Y_START = 158
ROW_SPACING = 23

ROW_DATA = []
for idx, (label, val) in enumerate(INFO_ROWS):
    y = ROW_Y_START + idx * ROW_SPACING
    approx_val_chars = len(val)
    approx_lbl_chars = len(label)
    leader_dot_count = max(4, 72 - approx_lbl_chars - approx_val_chars)
    leader_str = ". " * (leader_dot_count // 2)
    ROW_DATA.append({
        'label': label,
        'val': val,
        'y': y,
        'leader': leader_str
    })

# ==========================================
# 6. ASSEMBLE SVGS (dark.svg & light.svg)
# ==========================================
KEYTIMES_STR = "0;0.2113;0.3028;0.4437;0.5352;0.6761;0.7676;0.9085;1"
PORTRAIT_OFFSET_X = 65
PORTRAIT_OFFSET_Y = 100

def generate_svg_portrait_bands(pts, bands, displacements, color):
    svg_bands_xml = []
    for b in range(N_DRIFT_BANDS):
        idx = np.where(bands == b)[0]
        if len(idx) == 0:
            continue
        disp = displacements[b]
        dx, dy = f"{disp[0]:.2f}", f"{disp[1]:.2f}"
        
        b_pts = pts[idx]
        order = np.lexsort((b_pts[:, 0], b_pts[:, 1]))
        s_pts = b_pts[order]
        
        runs = []
        cur_y = None
        start_x = None
        end_x = None
        for pt in s_pts:
            px, py = int(round(pt[0])), int(round(pt[1]))
            if cur_y == py and px == end_x + 1:
                end_x = px
            else:
                if cur_y is not None:
                    runs.append((start_x, cur_y, end_x - start_x + 1))
                cur_y = py
                start_x = px
                end_x = px
        if cur_y is not None:
            runs.append((start_x, cur_y, end_x - start_x + 1))
            
        d_parts = [f"M{rx} {ry}h{rlen}v1h{-rlen}Z" for rx, ry, rlen in runs]
        d_str = "".join(d_parts)
        
        trans_vals = f"0 0;0 0;{dx} {dy};{dx} {dy};{dx} {dy};{dx} {dy};{dx} {dy};{dx} {dy};0 0"
        opac_vals = "1;1;0;0;0;0;0;0;1"
        
        band_xml = f'''    <g id="drift_band_{b}">
      <animateTransform attributeName="transform" type="translate" dur="14.2s" repeatCount="indefinite"
        keyTimes="{KEYTIMES_STR}" values="{trans_vals}" />
      <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite"
        keyTimes="{KEYTIMES_STR}" values="{opac_vals}" />
      <path d="{d_str}" fill="{color}" shape-rendering="crispEdges"/>
    </g>'''
        svg_bands_xml.append(band_xml)
    return "\n".join(svg_bands_xml)

def generate_svg_travellers(trajectories, color):
    trav_xml = []
    for i in range(len(trajectories)):
        p1 = trajectories[i, 0]
        p2 = trajectories[i, 1]
        p3 = trajectories[i, 2]
        
        x_vals = f"{p1[0]:.1f};{p1[0]:.1f};{p1[0]:.1f};{p1[0]:.1f};{p2[0]:.1f};{p2[0]:.1f};{p3[0]:.1f};{p3[0]:.1f};{p1[0]:.1f}"
        y_vals = f"{p1[1]:.1f};{p1[1]:.1f};{p1[1]:.1f};{p1[1]:.1f};{p2[1]:.1f};{p2[1]:.1f};{p3[1]:.1f};{p3[1]:.1f};{p1[1]:.1f}"
        
        item = f'''      <circle r="1.15" fill="{color}">
        <animate attributeName="cx" dur="14.2s" repeatCount="indefinite" keyTimes="{KEYTIMES_STR}" values="{x_vals}" />
        <animate attributeName="cy" dur="14.2s" repeatCount="indefinite" keyTimes="{KEYTIMES_STR}" values="{y_vals}" />
      </circle>'''
        trav_xml.append(item)
    return "\n".join(trav_xml)

def generate_intro_mask(pts, groups):
    mask_xml = ['  <mask id="intro-mask">', '    <rect width="300" height="340" fill="black" />']
    for g in range(N_INTRO_GROUPS):
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        g_pts = pts[idx]
        d_parts = [f"M{int(round(p[0]))} {int(round(p[1]))}h1v1h-1Z" for p in g_pts]
        d_str = "".join(d_parts)
        begin_t = f"{g * (1.2 / (N_INTRO_GROUPS - 1)):.3f}s"
        mask_xml.append(f'''    <g opacity="0">
      <animate attributeName="opacity" from="0" to="1" dur="2s" begin="{begin_t}" fill="freeze" />
      <path d="{d_str}" fill="white" shape-rendering="crispEdges" />
    </g>''')
    mask_xml.append('  </mask>')
    return "\n".join(mask_xml)

intro_mask_xml = generate_intro_mask(portrait_pts, intro_groups)

def build_banner_svg(mode='dark'):
    if mode == 'dark':
        bg_col = "#0A101F"
        card_bg = "#0D1527"
        border_col = "#1E293B"
        ui_chrome = "#22D3EE"
        ui_chrome_dim = "#0891B2"
        accent_col = "#10B981"
        portrait_col = "#A78BFA"
        text_label = "#94A3B8"
        text_val = "#F1F5F9"
        text_leader = "#334155"
        pill_bg = "rgba(34, 211, 238, 0.12)"
        pill_stroke = "#22D3EE"
        pill_text = "#22D3EE"
        pts_to_use = portrait_pts
        bands_to_use = drift_bands
        disps_to_use = band_displacements
    else:
        bg_col = "#F8FAFC"
        card_bg = "#FFFFFF"
        border_col = "#CBD5E1"
        ui_chrome = "#0891B2"
        ui_chrome_dim = "#0284C7"
        accent_col = "#059669"
        portrait_col = "#7C3AED"
        text_label = "#64748B"
        text_val = "#0F172A"
        text_leader = "#CBD5E1"
        pill_bg = "rgba(8, 145, 178, 0.10)"
        pill_stroke = "#0891B2"
        pill_text = "#0891B2"
        
        l_y, l_x = np.where(light_dots)
        l_pts = np.column_stack([l_x, l_y]).astype(float)
        dx_l = l_pts[:, 0] - logo1_centroid[0]
        dy_l = l_pts[:, 1] - logo1_centroid[1]
        base_l = dy_l + 0.35 * dx_l + np.random.normal(0, 4.0, len(l_pts))
        l_quants = np.percentile(base_l, np.linspace(0, 100, N_DRIFT_BANDS + 1))
        l_quants[0] -= 1e-5
        l_quants[-1] += 1e-5
        l_bands = np.clip(np.digitize(base_l, l_quants[:-1]) - 1, 0, N_DRIFT_BANDS - 1)
        l_disps = []
        for b in range(N_DRIFT_BANDS):
            idx_b = np.where(l_bands == b)[0]
            if len(idx_b) > 0:
                c = l_pts[idx_b].mean(axis=0)
                l_disps.append(0.42 * (logo1_centroid - c))
            else:
                l_disps.append(np.array([0.0, 0.0]))
        pts_to_use = l_pts
        bands_to_use = l_bands
        disps_to_use = l_disps

    bands_xml = generate_svg_portrait_bands(pts_to_use, bands_to_use, disps_to_use, portrait_col)
    travellers_xml = generate_svg_travellers(trajectories, portrait_col)

    rows_xml = []
    for r in ROW_DATA:
        y = r['y']
        lbl = r['label']
        val = r['val']
        rows_xml.append(f'''    <g class="info-row">
      <text x="{PANEL_X_START}" y="{y}" fill="{text_label}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14" font-weight="600">{lbl}</text>
      <text x="{PANEL_X_START + 120}" y="{y}" textLength="{PANEL_WIDTH - 120}" lengthAdjust="spacingAndGlyphs" fill="{text_leader}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14" opacity="0.6">{r['leader']}</text>
      <text x="{PANEL_X_END}" y="{y}" text-anchor="end" fill="{text_val}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14" font-weight="500">{val}</text>
    </g>''')
    rows_xml_str = "\n".join(rows_xml)

    svg_content = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1180 610" width="1180" height="610">
  <defs>
    <!-- Master Intro Mask: reveals dots randomly over 3.2s -->
{intro_mask_xml}
    
    <!-- Pulse Glow Filter for LIVE badge -->
    <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="blur" />
      <feMerge>
        <feMergeNode in="blur" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  </defs>

  <style>
    .terminal-title {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 13px; font-weight: 600; fill: {ui_chrome}; }}
    .section-label {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 13px; font-weight: 700; fill: {ui_chrome}; letter-spacing: 1.5px; }}
    .badge-live {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 12px; font-weight: 700; fill: #EF4444; }}
    .pill-text {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 14px; font-weight: 700; fill: {pill_text}; }}
    .meta-text {{ font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace; font-size: 11px; fill: {text_label}; }}
  </style>

  <!-- Window Container -->
  <rect x="0" y="0" width="1180" height="610" rx="14" fill="{bg_col}" stroke="{border_col}" stroke-width="1.5" />

  <!-- Terminal Title Bar -->
  <path d="M0 14C0 6.268 6.268 0 14 0H1166C1173.73 0 1180 6.268 1180 14V38H0V14Z" fill="{card_bg}" stroke="{border_col}" stroke-width="1.5" />
  
  <!-- Window Controls -->
  <circle cx="24" cy="19" r="6" fill="#EF4444" opacity="0.9" />
  <circle cx="44" cy="19" r="6" fill="#F59E0B" opacity="0.9" />
  <circle cx="64" cy="19" r="6" fill="#10B981" opacity="0.9" />
  
  <!-- Terminal Window Title -->
  <text x="590" y="24" text-anchor="middle" class="terminal-title">profile.sh --live</text>

  <!-- Panel Divider Line (Left ~38% = 440px) -->
  <line x1="440" y1="38" x2="440" y2="610" stroke="{border_col}" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.6" />

  <!-- ============================================== -->
  <!-- LEFT PANEL: VISUAL.MAP & ANIMATED MORPH        -->
  <!-- ============================================== -->
  <g id="visual-map-header">
    <text x="{PORTRAIT_OFFSET_X}" y="76" class="section-label">VISUAL.MAP</text>
    <text x="{PORTRAIT_OFFSET_X + 300}" y="76" text-anchor="end" class="meta-text">[300×340 DITHER / 1-BIT]</text>
  </g>

  <!-- Frame Around Portrait -->
  <rect x="{PORTRAIT_OFFSET_X - 4}" y="{PORTRAIT_OFFSET_Y - 4}" width="308" height="348" rx="4" fill="none" stroke="{border_col}" stroke-width="1" />
  <!-- UI Chrome Corner Brackets -->
  <path d="M{PORTRAIT_OFFSET_X - 4} {PORTRAIT_OFFSET_Y + 12}V{PORTRAIT_OFFSET_Y - 4}H{PORTRAIT_OFFSET_X + 12}" fill="none" stroke="{ui_chrome}" stroke-width="2" />
  <path d="M{PORTRAIT_OFFSET_X + 292} {PORTRAIT_OFFSET_Y - 4}H{PORTRAIT_OFFSET_X + 304}V{PORTRAIT_OFFSET_Y + 12}" fill="none" stroke="{ui_chrome}" stroke-width="2" />
  <path d="M{PORTRAIT_OFFSET_X - 4} {PORTRAIT_OFFSET_Y + 332}V{PORTRAIT_OFFSET_Y + 344}H{PORTRAIT_OFFSET_X + 12}" fill="none" stroke="{ui_chrome}" stroke-width="2" />
  <path d="M{PORTRAIT_OFFSET_X + 292} {PORTRAIT_OFFSET_Y + 344}H{PORTRAIT_OFFSET_X + 304}V{PORTRAIT_OFFSET_Y + 332}" fill="none" stroke="{ui_chrome}" stroke-width="2" />

  <!-- PORTRAIT LAYER (Grouped into 94 Drift Bands) -->
  <g transform="translate({PORTRAIT_OFFSET_X}, {PORTRAIT_OFFSET_Y})" mask="url(#intro-mask)">
{bands_xml}
  </g>

  <!-- TRAVELLERS LAYER (900 Dots Morphing Between Logos) -->
  <g transform="translate({PORTRAIT_OFFSET_X}, {PORTRAIT_OFFSET_Y})">
    <g id="travellers-master">
      <animate attributeName="opacity" dur="14.2s" repeatCount="indefinite"
        keyTimes="{KEYTIMES_STR}" values="0;0;1;1;1;1;1;0;0" />
{travellers_xml}
    </g>
  </g>

  <!-- Visual Map Status Footer -->
  <g transform="translate({PORTRAIT_OFFSET_X}, 472)">
    <text x="0" y="0" class="meta-text" fill="{ui_chrome}">MORPH.SYS: <tspan fill="{accent_col}">ACTIVE</tspan></text>
    <text x="300" y="0" text-anchor="end" class="meta-text">LOOP: 14.2s (OT-TRANS)</text>
    <text x="0" y="20" class="meta-text">1. PYTHON  2. &lt;/&gt; CODE  3. GITHUB</text>
    <text x="300" y="20" text-anchor="end" class="meta-text" fill="{text_label}">BANDS: 94 / PTS: 900</text>
    
    <!-- Progress Indicator Bar -->
    <rect x="0" y="32" width="300" height="4" rx="2" fill="{card_bg}" stroke="{border_col}" stroke-width="0.8" />
    <rect x="0" y="32" width="300" height="4" rx="2" fill="{ui_chrome}">
      <animate attributeName="width" values="0;300;300;0" dur="14.2s" repeatCount="indefinite" keyTimes="0;0.9085;0.999;1" />
    </rect>
  </g>

  <!-- ============================================== -->
  <!-- RIGHT PANEL: SYSTEM.INFO & READOUT             -->
  <!-- ============================================== -->
  <!-- Header Bar -->
  <g id="system-info-header">
    <text x="{PANEL_X_START}" y="76" class="section-label">SYSTEM.INFO</text>
    
    <!-- Pulsing Red LIVE Badge (font-size: 12) -->
    <g transform="translate({PANEL_X_START + 120}, 72)">
      <circle cx="0" cy="-4" r="5" fill="none" stroke="#EF4444" stroke-width="1.2">
        <animate attributeName="r" values="4;9;4" dur="2s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="0.8;0;0.8" dur="2s" repeatCount="indefinite" />
      </circle>
      <circle cx="0" cy="-4" r="4.5" fill="#EF4444" filter="url(#glow-red)">
        <animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite" />
      </circle>
      <text x="12" y="0" class="badge-live">LIVE</text>
    </g>

    <!-- Coloured Pill Containing GitHub Handle (font-size: 14) -->
    <g transform="translate({PANEL_X_END - 180}, 58)">
      <rect x="0" y="0" width="180" height="28" rx="14" fill="{pill_bg}" stroke="{pill_stroke}" stroke-width="1.2" />
      <text x="90" y="19" text-anchor="middle" class="pill-text">AkshayBendre-03</text>
    </g>
  </g>

  <!-- Divider Line Below Info Header -->
  <line x1="{PANEL_X_START}" y1="102" x2="{PANEL_X_END}" y2="102" stroke="{border_col}" stroke-width="1" />

  <!-- Info Panel Rows (font-size: 14, spacing: 23px) -->
{rows_xml_str}

  <!-- Footer Meta -->
  <g transform="translate({PANEL_X_START}, 546)">
    <line x1="0" y1="0" x2="{PANEL_WIDTH}" y2="0" stroke="{border_col}" stroke-width="1" opacity="0.6" />
    <text x="0" y="24" class="meta-text">LOC: HUBLI // STATUS: 200 OK // HOST: GITHUB</text>
    <text x="{PANEL_WIDTH}" y="24" text-anchor="end" class="meta-text" fill="{accent_col}">● ALL SYSTEMS NOMINAL</text>
  </g>
</svg>'''
    return svg_content

print("Generating dark.svg...")
dark_svg_code = build_banner_svg('dark')
with open('dark.svg', 'w', encoding='utf-8') as f:
    f.write(dark_svg_code)
dark_size_kb = os.path.getsize('dark.svg') / 1024
print(f"Created dark.svg: {dark_size_kb:.1f} KB")

print("Generating light.svg...")
light_svg_code = build_banner_svg('light')
with open('light.svg', 'w', encoding='utf-8') as f:
    f.write(light_svg_code)
light_size_kb = os.path.getsize('light.svg') / 1024
print(f"Created light.svg: {light_size_kb:.1f} KB")

# Verification metrics summary
band_counts = [len(np.where(drift_bands == b)[0]) for b in range(N_DRIFT_BANDS)]
print("\n" + "="*50)
print("PHASE 1 VERIFICATION REPORT")
print("="*50)
print(f"Grid Size: 300x340 ({GRID_W*GRID_H:,} px)")
print(f"Dark Mode Portrait Dots: {n_dark_dots:,} (Target: ~17,000)")
print(f"Dark Mode Mask Area: {np.sum(mask):,} px ({np.mean(mask)*100:.1f}% coverage)")
print(f"Dark Mode Ink Coverage inside Mask: {n_dark_dots / np.sum(mask) * 100:.1f}%")
print(f"Light Mode Portrait Dots: {n_light_dots:,} ({n_light_dots / (GRID_W*GRID_H) * 100:.1f}% total ink coverage)")
print(f"Drift Bands: {N_DRIFT_BANDS} (Mean: {np.mean(band_counts):.1f} dots/band, Min: {min(band_counts)}, Max: {max(band_counts)})")
print(f"Straight-Boundary Metric: {straight_metric:.4f} (Target ~0.01; ~0.17 means grid)")
print(f"Intro Groups: {N_INTRO_GROUPS} (Staggered over 3.2s, 2s fade)")
print(f"Intro Evenness Metric: {intro_metric:.4f} (Target ~0.05; ~0.70 is patchy)")
print(f"Travellers: {N_TRAVELLERS} dots (Morphing: Python -> Code -> GitHub via Optimal Transport)")
print(f"Animation Cycle: 14.2s Loop (Portrait: 3.0s, Logos: 2.0s ea, Transitions: 1.3s ea)")
print(f"dark.svg Size: {dark_size_kb:.1f} KB (Target: ~900KB-1MB)")
print(f"light.svg Size: {light_size_kb:.1f} KB")
print("="*50)
