import argparse
import json
import os
import webbrowser
from pathlib import Path
import time


def generate_graph_js(combined_graphs, js_file):
    """Converts combined graphs into a readable js object.

    Args:
        combined_graphs: Dictionary containing two scene graphs ('im1' and 'im2').
        js_file: The javascript file to write to.
    """
    processed_graphs = []
    for key in ["im1", "im2"]:
        graph = combined_graphs[key]
        scene_graph = {
            "objects": [],
            "attributes": [],
            "relationships": [],
            "url": graph["url"],
            "unique_objects": graph["# of unique objects"],
            "unique_attrs": graph["# of unique attributes"],
            "unique_rels": graph["# of unique relationships"],
            "objects_without_synsets": graph["# of objects with missing synsets"],
            "attributes_without_synsets": graph["# of attributes with missing synsets"],
            "relationships_without_synsets": graph[
                "# of relationships with missing synsets"
            ],
            "SAM_segmentations": graph["# of SAM segmentations"],
            "SAM2_segmentations": graph["# of SAM 2 segmentations"],
            "FC_CLIP_classes": graph["# of FC-CLIP classes"],
            "regions": graph.get("regions", []),
            "dataset": graph.get("dataset", ""),
        }

        for obj in graph["objects"]:
            name = ""
            if "name" in obj:
                name = obj["name"]
            elif "names" in obj and len(obj["names"]) > 0:
                name = obj["names"][0]
            scene_graph["objects"].append({"name": name})

        scene_graph["attributes"] = graph["attributes"]
        scene_graph["relationships"] = graph["relationships"]
        processed_graphs.append(scene_graph)

    with open(js_file, "w") as f:
        f.write("var graphs = " + json.dumps(processed_graphs))


def get_html_path() -> str:
    current_dir = Path(os.getcwd())

    if current_dir.name == "notebooks":
        project_root = current_dir.parent
    elif current_dir.name == "Visual Genome Driver":
        project_root = current_dir
    else:
        raise ValueError(
            f"Script must be run from either 'Visual Genome Driver' or its 'notebooks' subdirectory. "
            f"Current directory: {current_dir}"
        )

    base_name = "comparison"
    filename = f"{base_name}.html"
    file_path = project_root / "graphviz" / filename
    file_path.parent.mkdir(exist_ok=True)

    return str(file_path)


def process_single_graph(graph):
    """Processes a single graph into the required format."""
    scene_graph = {
        "objects": [],
        "attributes": [],
        "relationships": [],
        "url": graph["url"],
        "unique_objects": graph.get("# of unique objects", 0),
        "unique_attrs": graph.get("# of unique attributes", 0),
        "unique_rels": graph.get("# of unique relationships", 0),
        "objects_without_synsets": graph.get("# of objects with missing synsets", 0),
        "attributes_without_synsets": graph.get(
            "# of attributes with missing synsets", 0
        ),
        "relationships_without_synsets": graph.get(
            "# of relationships with missing synsets", 0
        ),
        "SAM_segmentations": graph.get("# of SAM segmentations", "Not available"),
        "SAM2_segmentations": graph.get("# of SAM 2 segmentations", "Not available"),
        "FC_CLIP_classes": graph.get("# of FC-CLIP classes", "Not available"),
        "regions": graph.get("regions", []),
        "dataset": graph.get("dataset", ""),
    }

    for obj in graph["objects"]:
        name = ""
        if "name" in obj:
            name = obj["name"]
        elif "names" in obj and len(obj["names"]) > 0:
            name = obj["names"][0]
        scene_graph["objects"].append({"name": name})

    scene_graph["attributes"] = graph["attributes"]
    scene_graph["relationships"] = graph["relationships"]

    return scene_graph


def visualize_paired_graphs(combined_graphs, js_file):
    """Creates an html visualization of the paired scene graphs.

    Args:
        combined_graphs: Dictionary containing two scene graphs ('im1' and 'im2').
        js_file: The javascript file to write to.
    """
    generate_graph_js(combined_graphs, js_file)
    html_file_path = get_html_path()
    webbrowser.open("file://" + html_file_path)
    time.sleep(1)


def get_path(file: str) -> str:
    current_dir = Path(os.getcwd())

    if current_dir.name == "notebooks":
        project_root = current_dir.parent
    elif current_dir.name == "Visual Genome Driver":
        project_root = current_dir
    else:
        raise ValueError(
            f"Script must be run from either 'Visual Genome Driver' or its 'notebooks' subdirectory. "
            f"Current directory: {current_dir}"
        )

    file_path = project_root / "graphviz" / file

    if not file_path.parent.exists():
        raise FileNotFoundError(
            f"Graphviz directory not found at {file_path.parent}. "
            f"Please ensure the directory structure is correct."
        )

    return str(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-scene-graphs",
        "-n",
        type=int,
        default=1,
        help="Number of scene graph pairs to visualize.",
    )
    args = parser.parse_args()

    for i in range(args.no_scene_graphs):
        # Construct filenames
        if i >= 1:
            graph_filename = f"scene_graph_combined{i}.json"
        else:
            graph_filename = "scene_graph_combined.json"

        js_filename = "scene_graphs.js"

        # Get the paths
        graph_path = get_path(graph_filename)
        js_file_path = get_path(js_filename)

        # Read and process the combined graphs
        with open(graph_path) as f:
            combined_graphs = json.load(f)

        # Visualize the pair
        visualize_paired_graphs(combined_graphs, js_file_path)

        # Clean up
        os.remove(graph_path)
