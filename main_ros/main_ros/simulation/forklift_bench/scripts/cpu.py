"""Load the same scene and ROS interfaces in the renderer-free Kit application."""
import json
import os
from pathlib import Path
import sys
import traceback

sys.dont_write_bytecode = True

import omni.kit.app
import omni.usd

sys.path.insert(0, os.environ["SIMFORKLIFT_ROS_PYTHON"])
sys.path.insert(0, str(Path(os.environ["ISAAC_SIM_ROOT"])/"exts/omni.pip.compute/pip_prebundle"))
sys.path.insert(0, str(Path(os.environ["SIMFORKLIFT_ROOT"])/"scripts"))
from simulation import options, serve

try:
    args = options().parse_args(json.loads(os.environ["SIMFORKLIFT_ARGS"]))
    args.no_rgb = args.no_rgb or not args.cpu_rgb
    context = omni.usd.get_context()
    if not context.open_stage(str(args.world.resolve())):
        raise RuntimeError(f"Could not open {args.world}")
    serve(context.get_stage(), args)
except KeyboardInterrupt:
    omni.kit.app.get_app().post_quit()
except SystemExit as error:
    omni.kit.app.get_app().post_quit(error.code or 0)
except Exception:
    traceback.print_exc()
    omni.kit.app.get_app().post_quit(1)
else:
    omni.kit.app.get_app().post_quit()
