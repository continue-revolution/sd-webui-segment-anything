import launch
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(current_dir, "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            launch.run_pip(
                f"install {lib}", f"sd-webui-segment-anything requirement: {lib}")

launch.git_pull_recursive(current_dir)
dino_dir = os.path.join(current_dir, "GroundingDINO")
launch.run_pip(f"install -e {dino_dir}", "sd-webui-segment-anything requirement: GroundingDINO")
