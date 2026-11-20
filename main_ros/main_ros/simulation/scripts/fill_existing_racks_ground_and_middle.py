"""
Fill the GROUND and FIRST SHELF of existing Isaac Sim racks with blue/yellow boxes.
Run INSIDE Isaac Sim's Script Editor while your hand-edited warehouse USD
is open. Does not regenerate, move, or replace racks or walls.

Designed for the rack/module structure from build_warehouse_lab.py:
  /World/WarehouseLab/Racks/<RackName>/Module<N>/Shelf1, Shelf2, ...
The racks have three storage levels: ground, top of Shelf1, top of Shelf2.
This script fills ONLY ground and top of Shelf1; Shelf2 stays EMPTY.
It finds shelves at their CURRENT positions, including hand-adjusted parents.

SAVE A COPY of your USD first (File > Save As).
"""
import math
import re

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

# All values in METERS.
BOX_LENGTH = 0.12  # Along the longer dimension of a shelf
BOX_DEPTH = 0.10   # Across the shorter dimension
BOX_HEIGHT = 0.13
END_MARGIN = 0.010  # Keep boxes away from end posts (1 cm)
GAP = 0.006         # Between neighboring boxes (6 mm)
BOTTOM_GAP = 0.001  # 1 mm above shelf to avoid visual overlap

# Re-running replaces ONLY previously created boxes under this path.
AUTO_BOXES_ROOT = "/World/Ray_Auto_Boxes"

# Original warehouse-building script added small brown placeholder "Carton"
# objects to alternating modules. Leave them alone by default.
# If those old placeholders overlap with new boxes, set True ONLY if
# you want to delete those auto-generated placeholders.
REMOVE_OLD_PLACEHOLDER_CARTONS = False

stage = omni.usd.get_context().get_stage()
if stage is None:
    raise RuntimeError("Open your edited warehouse USD in Isaac Sim first.")

# Collect ONLY FIRST SHELVES BEFORE editing the scene.
# Each Shelf1 supplies the ground-row footprint and the middle-row surface.
shelves = [
    p for p in stage.Traverse()
    if p.GetTypeName() == "Cube"
    and p.GetName() == "Shelf1"
    and "/WarehouseLab/Racks/" in str(p.GetPath())
]
shelves.sort(key=lambda p: str(p.GetPath()))

if not shelves:
    raise RuntimeError(
        "No rack shelves found. Check that the stage contains "
        "/World/WarehouseLab/Racks/.../Shelf1, or send me your USD."
    )

# Optional: remove only the OLD brown sample boxes from the original generator.
legacy = [
    p.GetPath() for p in stage.Traverse()
    if p.GetName() == "Carton"
    and "/WarehouseLab/Racks/" in str(p.GetPath())
]
if REMOVE_OLD_PLACEHOLDER_CARTONS:
    for path in legacy:
        stage.RemovePrim(path)

# The script never deletes or changes the existing shelves, racks or walls.
if stage.GetPrimAtPath(AUTO_BOXES_ROOT).IsValid():
    stage.RemovePrim(AUTO_BOXES_ROOT)
UsdGeom.Xform.Define(stage, AUTO_BOXES_ROOT)
looks_root = f"{AUTO_BOXES_ROOT}/Looks"
UsdGeom.Xform.Define(stage, looks_root)


def make_material(name, rgb):
    mat = UsdShade.Material.Define(stage, f"{looks_root}/{name}")
    shader = UsdShade.Shader.Define(stage, f"{looks_root}/{name}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


BLUE_RGB = (0.05, 0.27, 0.88)
YELLOW_RGB = (0.96, 0.76, 0.06)
blue_material = make_material("Blue", BLUE_RGB)
yellow_material = make_material("Yellow", YELLOW_RGB)

matrix_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
bbox_cache = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
)
created = 0
skipped = []

# Ground level is the TOP face of the real warehouse floor, not the top
# of the bottom shelf. The generated warehouse initially places it at Z=0.
floor_prim = stage.GetPrimAtPath("/World/WarehouseLab/Floor/Main")
if floor_prim.IsValid():
    floor_bound = bbox_cache.ComputeWorldBound(floor_prim).ComputeAlignedBox()
    floor_top_z = float(floor_bound.GetMax()[2])
else:
    floor_top_z = 0.0
    print("WARNING: warehouse floor not found; assuming ground surface Z=0.0 m.")


def place_box(prim_path, pos, yaw_deg, rgb, material):
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
    cube.AddRotateZOp().Set(float(yaw_deg))
    cube.AddScaleOp().Set(
        Gf.Vec3f(BOX_LENGTH, BOX_DEPTH, BOX_HEIGHT)
    )
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    # Static collision; boxes remain in place for initial SLAM tests.
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


for shelf_index, shelf in enumerate(shelves, start=1):
    path = str(shelf.GetPath())
    cube = UsdGeom.Cube(shelf)
    size = cube.GetSizeAttr().Get()
    if size is None:
        size = 2.0  # USD Cube's default size when the attribute is unauthored

    # Includes transformations made by hand in Isaac Sim, even on parent groups.
    matrix = matrix_cache.GetLocalToWorldTransform(shelf)
    xaxis = matrix.TransformDir(Gf.Vec3d(1, 0, 0))
    yaxis = matrix.TransformDir(Gf.Vec3d(0, 1, 0))
    length_x = xaxis.GetLength() * float(size)
    length_y = yaxis.GetLength() * float(size)

    if length_x >= length_y:
        shelf_length, shelf_depth = length_x, length_y
        along = xaxis / xaxis.GetLength()
    else:
        shelf_length, shelf_depth = length_y, length_x
        along = yaxis / yaxis.GetLength()

    if shelf_depth < BOX_DEPTH + 0.002:
        skipped.append((path, "Shelf too narrow"))
        continue

    usable = shelf_length - 2 * END_MARGIN
    nboxes = max(0, math.floor((usable + GAP + 1e-8) / (BOX_LENGTH + GAP)))
    if nboxes < 1:
        skipped.append((path, "Shelf too short"))
        continue

    # The box row is centered on the shelf footprint.
    row_length = nboxes * BOX_LENGTH + (nboxes - 1) * GAP
    start = -row_length / 2 + BOX_LENGTH / 2
    center = matrix.Transform(Gf.Vec3d(0, 0, 0))
    shelf_bound = bbox_cache.ComputeWorldBound(shelf).ComputeAlignedBox()
    first_shelf_top_z = float(shelf_bound.GetMax()[2])
    yaw = math.degrees(math.atan2(float(along[1]), float(along[0])))

    # Use the same X/Y footprint for both levels.
    # Ground (bottom): boxes on actual floor.
    # Middle: boxes on top of Shelf1.
    # Upper: Shelf2 is deliberately left EMPTY.
    for level_name, surface_z in (
        ("Ground", floor_top_z),
        ("Middle", first_shelf_top_z),
    ):
        level_root = f"{AUTO_BOXES_ROOT}/RackModule_{shelf_index:03d}/{level_name}"
        UsdGeom.Xform.Define(stage, level_root)

        for i in range(nboxes):
            offset = start + i * (BOX_LENGTH + GAP)
            x = float(center[0] + along[0] * offset)
            y = float(center[1] + along[1] * offset)
            z = surface_z + BOTTOM_GAP + BOX_HEIGHT / 2

            is_blue = (created % 2 == 0)
            place_box(
                f"{level_root}/Box_{i+1:03d}",
                (x, y, z),
                yaw,
                BLUE_RGB if is_blue else YELLOW_RGB,
                blue_material if is_blue else yellow_material,
            )
            created += 1

print(f"Filled ground and middle levels of {len(shelves)} rack modules.")
print(f"Total boxes: {created} (Blue: {(created + 1)//2}; Yellow: {created//2})")
print(f"Top shelves (Shelf2) intentionally left empty.")
print(f"Generated boxes only under {AUTO_BOXES_ROOT}")
if skipped:
    print("Skipped rack modules:")
    for path, reason in skipped:
        print(f"  {path}: {reason}")
if legacy and not REMOVE_OLD_PLACEHOLDER_CARTONS:
    print(
        f"NOTE: {len(legacy)} original brown 'Carton' placeholders remain. "
        "If they overlap the blue/yellow boxes, you can set "
        "REMOVE_OLD_PLACEHOLDER_CARTONS=True and rerun."
    )
print("Review the scene, then press Ctrl+S in Isaac Sim to save.")
