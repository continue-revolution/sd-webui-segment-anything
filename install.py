import launch
import os

current_dir = os.path.dirname(os.path.realpath(__file__))
req_file = os.path.join(current_dir, "requirements.txt")

with open(req_file) as file:
    for lib in file:
        lib = lib.strip()
        if not launch.is_installed(lib):
            if lib == "groundingdino":
                lib = "git+https://github.com/IDEA-Research/GroundingDINO"
            launch.run_pip(
                f"install {lib}", f"sd-webui-segment-anything requirement: {lib}")
