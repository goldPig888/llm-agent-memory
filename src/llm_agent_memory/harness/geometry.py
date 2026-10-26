"""Calibrated overlays and pixel/ray conversions. No hidden object information."""
import numpy as np
from PIL import Image, ImageDraw


def project(point, camera):
    relative = np.asarray(camera['R']).T @ (np.asarray(point) - camera['position'])
    if relative[2] <= 1e-5:
        return None
    pixel = np.asarray(camera['K']) @ relative
    return pixel[:2] / pixel[2]


def deproject(u, v, z_plane, camera, crop=(0, 0, 512, 512)):
    if not (0 <= u <= 1 and 0 <= v <= 1 and np.isfinite(z_plane)):
        raise ValueError('normalized pixels must be in [0,1] and height finite')
    px = crop[0] + u * (crop[2] - crop[0] - 1)
    py = crop[1] + v * (crop[3] - crop[1] - 1)
    direction = np.asarray(camera['R']) @ np.linalg.solve(np.asarray(camera['K']), [px, py, 1])
    center = np.asarray(camera['position'])
    if abs(direction[2]) < 1e-7:
        raise ValueError('ray is parallel to requested plane')
    distance = (z_plane - center[2]) / direction[2]
    if distance <= 0:
        raise ValueError('plane is behind camera')
    return tuple((center + distance * direction)[:2])


def annotate(rgb, camera, state, grid_z, crop=(0, 0, 512, 512), jaw_gap=.08):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    def line(a, b, color, width=1):
        a, b = project(a, camera), project(b, camera)
        if a is not None and b is not None and np.max(np.abs(np.r_[a, b])) < 10000:
            draw.line([tuple(a), tuple(b)], fill=color, width=width)
    for c in np.arange(-.6, .601, .02):
        major = abs(c * 10 - round(c * 10)) < 1e-5
        line([c, -.6, grid_z], [c, .6, grid_z], (200, 200, 200) if major else (90, 90, 90))
        line([-.6, c, grid_z], [.6, c, grid_z], (200, 200, 200) if major else (90, 90, 90))
    occupied=[]
    for x in np.arange(-.6, .601, .1):
        for y in np.arange(-.6, .601, .1):
            uv = project([x, y, grid_z], camera)
            if uv is not None and np.all(uv >= 0) and np.all(uv < 490):
                x0,y0=uv; box=(x0,y0,x0+52,y0+11)
                if not any(not (box[2]<b[0] or b[2]<box[0] or box[3]<b[1] or b[3]<box[1]) for b in occupied):
                    draw.text(tuple(uv), f'{x:.1f},{y:.1f}', fill=(255, 225, 40))
                    occupied.append(box)
    position = np.asarray(state[:3]); rotation = np.asarray(state[3:9]).reshape(2, 3).T
    z_axis = np.cross(rotation[:, 0], rotation[:, 1])
    line(position, position - z_axis * .08, 'orange', 3)
    # Panda jaws close along grip-site local X (YAM uses local Y).
    line(position - rotation[:, 0] * jaw_gap/2, position + rotation[:, 0] * jaw_gap/2, 'lime', 3)
    line(position, [position[0], position[1], grid_z], 'magenta', 2)
    uv = project(position, camera)
    if uv is not None and np.all(np.abs(uv) < 10000):
        x, y = uv; draw.ellipse((x-4, y-4, x+4, y+4), outline='lime', width=2)
    return image.crop(crop).resize((512, 512))
