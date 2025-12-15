import json
from pathlib import Path


def load_wireframe_data(json_file):
    """Load wireframe data from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    return data["vertices"], data["edges"]


def load_aruco_annotations(json_file):
    """Load ArUco marker annotations from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    # Return dictionary name so callers can auto-select 4x4 vs 5x5, etc.
    return (
        data["markers"],
        data.get("size", 0.021),
        data.get("border_width", 0.05),
        data.get("aruco_dictionary", "DICT_4X4_50"),
    )


def get_available_models(data_dir):
    """Get list of available models from the data directory."""
    wireframe_dir = Path(data_dir) / "wireframe"
    aruco_dir = Path(data_dir) / "aruco"

    if not wireframe_dir.exists() or not aruco_dir.exists():
        return []

    wireframe_files = list(wireframe_dir.glob("*_wireframe.json"))
    aruco_files = list(aruco_dir.glob("*_aruco.json"))

    wireframe_models = {f.stem.replace("_wireframe", "") for f in wireframe_files}
    aruco_models = {f.stem.replace("_aruco", "") for f in aruco_files}

    available_models = wireframe_models.intersection(aruco_models)
    return sorted(list(available_models))


def select_model_interactive(available_models):
    """Interactive model selection."""
    if not available_models:
        print("No models found in data directory!")
        return None

    print("\nAvailable models:")
    for i, model in enumerate(available_models, 1):
        print(f"  {i}. {model}")

    while True:
        try:
            choice = input(f"\nSelect model (1-{len(available_models)}) or 'q' to quit: ").strip()
            if choice.lower() == "q":
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_models):
                selected_model = available_models[choice_num - 1]
                print(f"Selected model: {selected_model}")
                return selected_model
            else:
                print(f"Please enter a number between 1 and {len(available_models)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")


def load_assembly_data(json_file):
    """Load FMB assembly data from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def parse_assembly_components(assembly_data):
    """Parse assembly data to extract component and marker information."""
    components = {}
    markers = {}

    for component in assembly_data["components"]:
        if component["type"] == "component":
            components[component["name"]] = {
                "id": component["id"],
                "display_name": component["displayName"],
                "position": component["position"],
                "rotation": component["rotation"],
                "parent_id": component["parentId"],
            }
        elif component["type"] == "marker":
            markers[component["name"]] = {
                "id": component["id"],
                "display_name": component["displayName"],
                "position": component["position"],
                "rotation": component["rotation"],
                "parent_id": component["parentId"],
            }

    return components, markers


__all__ = [
    "load_wireframe_data",
    "load_aruco_annotations",
    "get_available_models",
    "select_model_interactive",
    "load_assembly_data",
    "parse_assembly_components",
]

