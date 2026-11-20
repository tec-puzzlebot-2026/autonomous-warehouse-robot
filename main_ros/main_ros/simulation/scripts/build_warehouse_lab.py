"""Procedural scaled warehouse scene for Isaac Sim Script Editor.
Only modifies /World/WarehouseLab within YOUR saved warehouse_lab_v1.usd.
Edit provisional positions after measuring the lab.
"""
from pathlib import Path
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics, UsdLux

OUT = Path("/home/spark-439/data/ray_ws/simulation/worlds/warehouse_lab_v1.usd")
L, W, WALL_H = 4.90, 3.65, 0.41
DOCK_W, DOCK_D, DOCK_H = 0.36, 0.61, 0.25
RACK_L, RACK_D, LEVEL_H = 0.35, 0.12, 0.19
# PROVISIONAL values, NOT measured:
WALL_T, DOCK_Y0, DOCK_GAP = 0.025, 2.05, 0.06
RACK_LEVELS = 2
RACKS = [  # name, first unit center x, first unit center y, number, direction
    ("CenterBack", 1.44, 2.64, 4, "x"),
    ("CenterFront", 1.53, 1.47, 4, "x"),
    ("CenterRight", 3.54, 1.48, 3, "y"),
    ("WallNorth", 2.10, 3.32, 3, "x"),
    ("WallWest", 0.34, 2.34, 2, "y"),
]
stage = omni.usd.get_context().get_stage()
if stage is None or Path(stage.GetRootLayer().realPath).resolve() != OUT.resolve():
    raise RuntimeError(f"Save/open this scene FIRST: {OUT}")
if stage.GetPrimAtPath("/World/GroundPlane").IsValid():
    raise RuntimeError("Delete /World/GroundPlane first; script creates its own floor.")
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
world = UsdGeom.Xform.Define(stage, "/World")
stage.SetDefaultPrim(world.GetPrim())
ROOT = "/World/WarehouseLab"
if stage.GetPrimAtPath(ROOT).IsValid():
    stage.RemovePrim(ROOT)

def grp(path):
    UsdGeom.Xform.Define(stage, path)

def block(path, xyz, dims, color):
    obj = UsdGeom.Cube.Define(stage, path)
    obj.CreateSizeAttr(1.0)
    trans = UsdGeom.XformCommonAPI(obj)
    trans.SetTranslate(Gf.Vec3d(*xyz))
    trans.SetScale(Gf.Vec3f(*dims))
    obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(obj.GetPrim())

def wall(name, axis, position, start, end):
    if end <= start: return
    if axis == "x":
        center = ((start+end)/2, position, WALL_H/2)
        size = (end-start, WALL_T, WALL_H)
    else:
        center = (position, (start+end)/2, WALL_H/2)
        size = (WALL_T, end-start, WALL_H)
    block(f"{ROOT}/Perimeter/{name}", center, size, (0.65,0.65,0.63))

grp(ROOT)
for part in ("Floor","Perimeter","Docks","Racks"):
    grp(f"{ROOT}/{part}")
if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1))
    scene.CreateGravityMagnitudeAttr(9.81)
if not stage.GetPrimAtPath("/World/WarehouseLight").IsValid():
    UsdLux.DomeLight.Define(stage, "/World/WarehouseLight").CreateIntensityAttr(500)

# Main platform: its top is z=0.
block(f"{ROOT}/Floor/Main", (L/2,W/2,-0.01), (L,W,0.02), (0.40,0.42,0.43))
wall("South","x",0,0,L)
wall("North","x",W,0,L)
wall("West","y",0,0,W)

# Three openings on east side; U-shaped delivery docks extend to x>L.
previous = 0
for i in range(3):
    a = DOCK_Y0 + i*(DOCK_W+DOCK_GAP)
    b = a + DOCK_W
    if b >= W: raise ValueError("Dock exceeds warehouse width")
    wall(f"EastSegment{i+1}","y",L,previous,a)
    previous = b
    dock = f"{ROOT}/Docks/Dock{i+1}"
    grp(dock)
    mid_y = (a+b)/2
    mid_x = L+DOCK_D/2
    block(f"{dock}/Floor",(mid_x,mid_y,-0.01),(DOCK_D,DOCK_W,0.02),(0.40,0.42,0.43))
    for edge, y in (("Left",a),("Right",b)):
        block(f"{dock}/{edge}Panel",(mid_x,y,DOCK_H/2),(DOCK_D,0.02,DOCK_H),(0.91,0.91,0.87))
    block(f"{dock}/BackPanel",(L+DOCK_D,mid_y,DOCK_H/2),(0.02,DOCK_W,DOCK_H),(0.91,0.91,0.87))
wall("EastFinal","y",L,previous,W)

# Racks are open frames, not solid cubes: useful for 2D lidar testing.
for bank_name, x0, y0, count, direction in RACKS:
    bank = f"{ROOT}/Racks/{bank_name}"
    grp(bank)
    for i in range(count):
        cx = x0 + (i*RACK_L if direction=="x" else 0)
        cy = y0 + (i*RACK_L if direction=="y" else 0)
        dx,dy = (RACK_L,RACK_D) if direction=="x" else (RACK_D,RACK_L)
        mod = f"{bank}/Module{i+1}"
        grp(mod)
        h = RACK_LEVELS*LEVEL_H
        post_t=0.012
        for j,(sx,sy) in enumerate(((-1,-1),(-1,1),(1,-1),(1,1)),start=1):
            block(f"{mod}/Post{j}",(cx+sx*(dx-post_t)/2,cy+sy*(dy-post_t)/2,h/2),
                  (post_t,post_t,h),(0.06,0.20,0.69))
        for level in range(1,RACK_LEVELS+1):
            z=level*LEVEL_H-0.006
            block(f"{mod}/Shelf{level}",(cx,cy,z),(dx,dy,0.012),(0.61,0.48,0.29))
        if i%2 == 0:
            block(f"{mod}/Carton",(cx,cy,LEVEL_H+0.035),(dx*0.5,dy*0.65,0.07),(0.72,0.55,0.34))

print("Created warehouse geometry inside", OUT)
print("Rack positions, tier count, dock placement and wall thickness are provisional.")
print("Select /World/WarehouseLab and press F; then Ctrl+S to save.")
