"""Headless RGB of the E80 asset and rack for the single-station physics bench."""
import cv2
import numpy as np
from pxr import Gf, Usd, UsdGeom

from physics import rotation


class CpuCamera:
    def __init__(self, sim, index):
        self.sim, self.index = sim, index
        stage = sim.stage
        station = f"/World/Sets/Set_{index:02d}"
        self.cargo = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(station + "/Load")):
            # Fine printing is omitted in the CPU renderer.
            if not prim.IsA(UsdGeom.Mesh) or prim.GetName() in ("grid_ink", "logo_white"):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get())
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            edges = np.cumsum(np.asarray(mesh.GetFaceVertexCountsAttr().Get()))
            color = tuple(round(255 * c) for c in mesh.GetDisplayColorAttr().Get()[0])
            self.cargo.append((points[indices], edges[:-1], color))
        self.static = []
        cache = UsdGeom.XformCache()
        square_faces = ((0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4),
                        (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5))
        for root in ("/World/Floor", "/World/CenterDivider", station + "/Rack"):
            prim = stage.GetPrimAtPath(root)
            if not prim:
                continue
            for obj in Usd.PrimRange(prim):
                if not obj.IsA(UsdGeom.Cube):
                    continue
                cube = UsdGeom.Cube(obj)
                transform = np.asarray(cache.GetLocalToWorldTransform(obj))
                size = cube.GetSizeAttr().Get() / 2
                corners = np.array([[x * size, y * size, z * size, 1.]
                                    for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
                vertices = (corners @ transform)[:, :3]
                colors = cube.GetDisplayColorAttr().Get()
                color = tuple(round(255 * c) for c in colors[0]) if colors else (80, 80, 80)
                self.static.extend((vertices[list(face)], color) for face in square_faces)

    @staticmethod
    def clip_near(poly, near=.025):
        if np.all(poly[:, 2] >= near):
            return poly
        if np.all(poly[:, 2] < near):
            return None
        result = []
        for a, b in zip(np.roll(poly, 1, axis=0), poly):
            if (a[2] >= near) != (b[2] >= near):
                result.append(a + (b-a) * ((near-a[2]) / (b[2]-a[2])))
            if b[2] >= near:
                result.append(b)
        return np.asarray(result) if len(result) >= 3 else None

    def render(self, width, height, intrinsics):
        camera = self.sim.link_pose(self.index, "camera_link_optical")
        camera_rotation = np.asarray(Gf.Matrix3d(rotation(camera)))
        load_pose = self.sim.loads.get_transforms()[self.sim.load_rows[self.index]]
        load_rotation = np.asarray(Gf.Matrix3d(rotation(load_pose)))
        camera_to_world = camera_rotation.T
        colored = [((poly-camera[:3]) @ camera_to_world, color) for poly, color in self.static]
        for vertices, cuts, color in self.cargo:
            optical = (vertices @ load_rotation + load_pose[:3] - camera[:3]) @ camera_to_world
            colored.extend((poly, color) for poly in np.split(optical, cuts))
        fx, fy, cx, cy = intrinsics
        image = np.full((height, width, 3), 38, dtype=np.uint8)
        depth = np.full((height, width), np.inf)
        ray_x = (np.arange(width) - cx) / fx
        ray_y = (np.arange(height) - cy) / fy
        for poly, color in colored:
            clipped = self.clip_near(poly)
            if clipped is None:
                continue
            uv = clipped[:, :2] / clipped[:, 2, None] * (fx, fy) + (cx, cy)
            x0, y0 = np.maximum(0, np.ceil(uv.min(axis=0))).astype(int)
            x1, y1 = np.minimum((width-1, height-1), np.floor(uv.max(axis=0))).astype(int)
            if x0 > x1 or y0 > y1:
                continue
            normal = np.cross(poly[1]-poly[0], poly[2]-poly[0])
            denominator = normal[0]*ray_x[None, x0:x1+1] + normal[1]*ray_y[y0:y1+1, None] + normal[2]
            projected = np.divide(normal @ poly[0], denominator,
                                  out=np.full(denominator.shape, np.inf),
                                  where=np.abs(denominator) > 1e-12)
            mask = np.zeros(projected.shape, dtype=np.uint8)
            vertices = np.rint(np.clip(uv-(x0, y0), -5000, 5000)).astype(np.int32)
            cv2.fillConvexPoly(mask, vertices, 1)
            sub_depth = depth[y0:y1+1, x0:x1+1]
            front = (mask != 0) & (projected >= .0249) & (projected <= sub_depth + .00002)
            sub_depth[front] = projected[front]
            image[y0:y1+1, x0:x1+1][front] = color
        return image
