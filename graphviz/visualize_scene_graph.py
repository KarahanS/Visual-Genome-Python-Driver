"""Visualizes a scene graph stored in a json in a webpage.
"""

import argparse
import json
import os
import webbrowser
from pathlib import Path
import time


def generate_graph_js(graph, js_file):
    """Converts a json into a readable js object.

    Args:
        graph: A scene graph object.
        js_file: The javascript file to write to.
    """
    f = open(js_file, "w")
    f.write("var graph = " + json.dumps(graph))
    f.close()


def get_html_path() -> str:
    """
    Returns the correct path for HTML files in the Visual Genome Driver project,
    handling calls from either the root directory or notebooks subdirectory.


    Returns:
        str: Correct path to the HTML file relative to the project structure

    Example:
        If called from 'Visual Genome Driver/notebooks' or 'Visual Genome Driver/',
        returns the correct path to access HTML files.
    """
    # Convert current working directory to Path object
    current_dir = Path(os.getcwd())

    # Extract the project root directory
    if current_dir.name == "notebooks":
        project_root = current_dir.parent
    elif current_dir.name == "Visual Genome Driver":
        project_root = current_dir
    else:
        raise ValueError(
            f"Script must be run from either 'Visual Genome Driver' or its 'notebooks' subdirectory. "
            f"Current directory: {current_dir}"
        )

    # Construct the filename with optional index
    base_name = "graphviz"
    filename = f"{base_name}.html"

    # Construct the full path
    file_path = project_root / "graphviz" / filename

    # Ensure the graphviz directory exists
    file_path.parent.mkdir(exist_ok=True)

    return str(file_path)


def visualize_scene_graph(graph, js_file):
    """Creates an html visualization of the scene graph.

    Args:
        graph: A scene graph object.
        js_file: The javascript file to write to.
    """
    scene_graph = {
        "objects": [],
        "attributes": [],
        "relationships": [],
        "url": graph["url"],
        "unique_objects": 0,
        "unique_attrs": 0,
        "unique_rels": 0,
        "objects_without_synsets": 0,
        "attributes_without_synsets": 0,
        "relationships_without_synsets": 0,
        "SAM_segmentations": 0,
        "SAM2_segmentations": 0,
        "FC_CLIP_classes": 0,
        "regions": [],
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
    scene_graph["unique_objects"] = graph["# of unique objects"]
    scene_graph["unique_attrs"] = graph["# of unique attributes"]
    scene_graph["unique_rels"] = graph["# of unique relationships"]
    scene_graph["objects_without_synsets"] = graph["# of objects with missing synsets"]
    scene_graph["attributes_without_synsets"] = graph[
        "# of attributes with missing synsets"
    ]
    scene_graph["relationships_without_synsets"] = graph[
        "# of relationships with missing synsets"
    ]
    scene_graph["SAM_segmentations"] = graph["# of SAM segmentations"]
    scene_graph["SAM2_segmentations"] = graph["# of SAM 2 segmentations"]
    scene_graph["FC_CLIP_classes"] = graph["# of FC-CLIP classes"]
    scene_graph["regions"] = graph["regions"]
    scene_graph["dataset"] = graph["dataset"]

    generate_graph_js(scene_graph, js_file)
    html_file_path = get_html_path()
    webbrowser.open("file://" + html_file_path)
    # add wait for 1 seconds
    time.sleep(1)


def get_path(file: str) -> str:
    """
    Returns the correct path for files in the Visual Genome Driver project,
    handling calls from either the root directory or notebooks subdirectory.

    Args:
        file (str): Name of the file to locate

    Returns:
        str: Correct path to the file relative to the project structure

    Example:
        If called from 'Visual Genome Driver/notebooks' or 'Visual Genome Driver/',
        returns the correct path to access project files.
    """
    # Convert current working directory to Path object
    current_dir = Path(os.getcwd())

    # Extract the project root directory
    if current_dir.name == "notebooks":
        project_root = current_dir.parent
    elif current_dir.name == "Visual Genome Driver":
        project_root = current_dir
    else:
        raise ValueError(
            f"Script must be run from either 'Visual Genome Driver' or its 'notebooks' subdirectory. "
            f"Current directory: {current_dir}"
        )

    # Construct the file path relative to the graphviz directory
    file_path = project_root / "graphviz" / file

    # Verify the path exists
    if not file_path.parent.exists():
        raise FileNotFoundError(
            f"Graphviz directory not found at {file_path.parent}. "
            f"Please ensure the directory structure is correct."
        )

    return str(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph",
        type=str,
        default="scene_graph.json",
        help="Location of scene graph file to visualize.",
    )
    parser.add_argument(
        "--js-file",
        type=str,
        default="scene_graph.js",
        help="Temporary file generated to enable visualization.",
    )
    parser.add_argument(
        "--no-scene-graphs",
        "-n",
        type=int,
        default=1,
        help="Number of scene graphs (images) to show in visualization.",
    )
    args = parser.parse_args()

    for i in range(args.no_scene_graphs):
        # Split the filename and extension
        graph_base, graph_ext = os.path.splitext(args.graph)

        # Add index to filenames (use i+1 to start from 1 instead of 0)
        if i >= 1:
            indexed_graph = f"{graph_base}{i}{graph_ext}"
        else:
            indexed_graph = args.graph

        # Get the adjusted paths
        graph_path = get_path(indexed_graph)
        js_file_path = get_path(args.js_file)

        with open(graph_path) as graph_file:
            graph = json.load(graph_file)

        visualize_scene_graph(graph, js_file_path)

        # remove the scene graph json files
        os.remove(graph_path)
