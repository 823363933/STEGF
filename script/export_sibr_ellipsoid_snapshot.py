from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_sibr_snapshot import build_arg_parser, export_snapshot


def main():
    parser = build_arg_parser(
        "Export a STEGF/STGS frame snapshot for SIBR diagnostics using anisotropic ellipsoid Gaussians."
    )
    parser.set_defaults(scene_ball_radius=0.0, scene_ball_opacity=0.0, scene_scale_multiplier=3.0)
    args = parser.parse_args()
    if float(args.scene_ball_radius) > 0.0:
        raise ValueError("export_sibr_ellipsoid_snapshot.py keeps real ellipsoid scales; do not pass --scene_ball_radius.")
    if float(args.scene_ball_opacity) > 0.0:
        raise ValueError("export_sibr_ellipsoid_snapshot.py keeps time-modulated opacity; do not pass --scene_ball_opacity.")
    export_snapshot(args)


if __name__ == "__main__":
    main()
