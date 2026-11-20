"""
Add WHITE END PANELS to existing rack groups in a manually edited Isaac Sim USD.

Use INSIDE Isaac Sim's Window > Script Editor while your warehouse scene is open.
- Does NOT regenerate or move your warehouse, racks, or blue/yellow boxes.
- Detects CURRENT rack world positions and rotations (including edits to parent Xforms).
- Adds ONE solid panel at EACH SHORT END of each rack bank.
- Re-running replaces ONLY panels previously created under OUTPUT_ROOT.
- Panels have STATIC COLLISION.
- Save a separate backup USD before running.

Expected structure from the earlier warehouse generator:
  /World/WarehouseLab/Racks/<RackBank>/Module*/Shelf1
  /World/WarehouseLab/Racks/<RackBank>/Module*/Post1 ... Post4
"""
import math
import re

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# ====== ADJUSTABLE PARAMETERS (meters) ======
OUTPUT_ROOT = "/World/Ray_Rack_End_Panels"
RACKS_ROOT = "/World/WarehouseLab/Racks"

PANEL_THICKNESS = 0.008   # 8 mm white panel at each rack end
PANEL_OUTER_GAP = 0.002   # 2 mm outside posts/shelf ends (prevents overlaps)
PANEL_WIDTH_EXTRA = 0.0  # Add extra width across the rack depth, if desired
PANEL_HEIGHT_EXTRA = 0.0 # Add extra height above the tallest rack post
GROUND_FALLBACK_Z = 0.0  # Only used when warehouse floor is missing
PANEL_COLOR = (0.90, 0.91, 0.88)  # off-white

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Open your manually edited warehouse USD scene first.")

racks_prim = stage.GetPrimAtPath(RACKS_ROOT)
if not racks_prim.IsValid():
    raise RuntimeError(
        f"Could not find {RACKS_ROOT}. This script expects the original "
        "WarehouseLab rack hierarchy; please share the USD if it was renamed."
    )

cache = UsdGeom.XformCache(Usd.TimeCode.Default())
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
)


def world_corners_of_cube(prim):
    """8 cube corners, in METERS / world coordinates; includes parent edits."""
    cube = UsdGeom.Cube(prim)
    size = cube.GetSizeAttr().Get()
    if size is None:
        size = 2.0
    radius = float(size) / 2.0
    mat = cache.GetLocalToWorldTransform(prim)
    return [
        mat.Transform(Gf.Vec3d(sx * radius, sy * radius, sz * radius))
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]


def cube_center(prim):
    return cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(0, 0, 0))


def shelf_long_direction(prim):
    """Return long-axis direction in the horizontal plane for a single shelf."""
    corners = world_corners_of_cube(prim)
    mat = cache.GetLocalToWorldTransform(prim)
    xvec = mat.TransformDir(Gf.Vec3d(1, 0, 0))
    yvec = mat.TransformDir(Gf.Vec3d(0, 1, 0))
    if xvec.GetLength() >= yvec.GetLength():
        candidate = xvec
    else:
        candidate = yvec
    norm = math.hypot(float(candidate[0]), float(candidate[1]))
    if norm < 1e-8:
        raise ValueError(f"Bad shelf transform: {prim.GetPath()}")
    return (float(candidate[0]) / norm, float(candidate[1]) / norm)


# Detect top of floor, including any user changes to the floor transform.
floor_prim = stage.GetPrimAtPath("/World/WarehouseLab/Floor/Main")
if floor_prim.IsValid():
    floor_z = float(
        bbox_cache.ComputeWorldBound(floor_prim).ComputeAlignedBox().GetMax()[2]
    )
else:
    floor_z = GROUND_FALLBACK_Z
    print(f"WARNING: warehouse floor missing; using Z={floor_z:.3f} m.")


def rack_bank_geometry(bank_prim):
    # Discover Shelf1 prims and rack structure; ignore generated cartons/boxes.
    first_shelves = []
    physical_parts = []
    for prim in Usd.PrimRange(bank_prim):
        if prim.GetTypeName() != "Cube":
            continue
        name = prim.GetName()
        if name == "Shelf1":
            first_shelves.append(prim)
        if re.fullmatch(r"(Post\d+|Shelf\d+)", name):
            physical_parts.append(prim)
    if not first_shelves:
        return None

    first_shelves.sort(key=lambda p: str(p.GetPath()))
    centers = [cube_center(p) for p in first_shelves]

    # Prefer the direction connecting the most distant pair of shelf modules,
    # if this is a multi-module rack. For a single module, use its long axis.
    best_dist_sq = 0.0
    axis = shelf_long_direction(first_shelves[0])
    for i, center_a in enumerate(centers):
        for center_b in centers[i + 1:]:
            dx = float(center_b[0] - center_a[0])
            dy = float(center_b[1] - center_a[1])
            dist_sq = dx*dx + dy*dy
            if dist_sq > best_dist_sq and dist_sq > 0.05**2:
                best_dist_sq = dist_sq
                dist = math.sqrt(dist_sq)
                axis = (dx/dist, dy/dist)

    ux, uy = axis
    vx, vy = -uy, ux  # perpendicular, across rack depth
    all_corners = []
    for prim in first_shelves:
        all_corners.extend(world_corners_of_cube(prim))

    along_vals = [float(p[0])*ux + float(p[1])*uy for p in all_corners]
    across_vals = [float(p[0])*vx + float(p[1])*vy for p in all_corners]
    near_edge = min(along_vals)
    far_edge = max(along_vals)
    across_center = (min(across_vals) + max(across_vals)) / 2.0
    depth = max(across_vals) - min(across_vals)

    # Determine actual rack height using all posts and shelves.
    if not physical_parts:
        raise RuntimeError(
            f"No Post1/Post2/Shelf1/Shelf2 cubes found in {bank_prim.GetPath()}. "
            "Check the rack prim names and hierarchy."
        )
    top_z = max(
        float(p[2])
        for prim in physical_parts
        for p in world_corners_of_cube(prim)
    )
    height = max(0.0, top_z - floor_z + PANEL_HEIGHT_EXTRA)
    if height < 0.01 or depth < 0.01:
        raise ValueError(f"Invalid dimensions for rack {bank_prim.GetPath()}")

    panel_positions = []
    for end_coord in (
        near_edge - PANEL_OUTER_GAP - PANEL_THICKNESS / 2,
        far_edge + PANEL_OUTER_GAP + PANEL_THICKNESS / 2,
    ):
        # Convert coordinates in rack's (along, across) basis back to world X/Y.
        x = end_coord*ux + across_center*vx
        y = end_coord*uy + across_center*vy
        z = floor_z + height/2
        panel_positions.append((x, y, z))

    return dict(
        positions=panel_positions,
        yaw_deg=math.degrees(math.atan2(uy, ux)),
        height=height,
        depth=depth + PANEL_WIDTH_EXTRA,
        module_count=len(first_shelves),
    )


# Compute ALL placement locations before modifying the scene.
banks = []
for bank in racks_prim.GetChildren():
    geom = rack_bank_geometry(bank)
    if geom is not None:
        banks.append((bank.GetName(), geom))
if not banks:
    raise RuntimeError(
        "No Shelf1 modules found under rack banks. "
        "Check rack hierarchy or share the USD to adapt the script."
    )

# Only remove panels produced by THIS script. Existing boxes are untouched.
if stage.GetPrimAtPath(OUTPUT_ROOT).IsValid():
    stage.RemovePrim(OUTPUT_ROOT)
UsdGeom.Xform.Define(stage, OUTPUT_ROOT)
materials_root = f"{OUTPUT_ROOT}/Looks"
UsdGeom.Scope.Define(stage, materials_root)

mat = UsdShade.Material.Define(stage, f"{materials_root}/OffWhite")
shader = UsdShade.Shader.Define(stage, f"{materials_root}/OffWhite/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
    Gf.Vec3f(*PANEL_COLOR)
)
shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

count = 0
for name, geom in banks:
    # Use original bank names; no new rack hierarchy or renamed racks.
    bank_out = f"{OUTPUT_ROOT}/{name}"
    UsdGeom.Xform.Define(stage, bank_out)
    for label, position in zip(("EndA", "EndB"), geom["positions"]):
        cube = UsdGeom.Cube.Define(stage, f"{bank_out}/{label}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position))
        cube.AddRotateZOp().Set(float(geom["yaw_deg"]))
        cube.AddScaleOp().Set(
            Gf.Vec3f(
                PANEL_THICKNESS,
                geom["depth"],
                geom["height"],
            )
        )
        cube.CreateDisplayColorAttr([Gf.Vec3f(*PANEL_COLOR)])
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())  # Static
        count += 1
    print(
        f"{name}: {geom['module_count']} modules, "
        f"panel height={geom['height']:.3f} m, "
        f"depth={geom['depth']:.3f} m."
    )

print(f"Created {count} end panels for {len(banks)} rack groups.")
print(f"Only edited {OUTPUT_ROOT}; existing racks, boxes and warehouse unchanged.")
print("Inspect panels. If correct, press Ctrl+S to save the USD.")