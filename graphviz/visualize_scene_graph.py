"""Visualizes a scene graph stored in a json in a webpage.
"""

import argparse
import json
import os
import webbrowser


def generate_graph_js(graph, js_file):
    """Converts a json into a readable js object.

    Args:
        graph: A scene graph object.
        js_file: The javascript file to write to.
    """
    f = open(js_file, "w")
    f.write("var graph = " + json.dumps(graph))
    f.close()


def get_html_path():
    """Returns the correct path to graphviz.html depending on the current working directory."""
    current_dir = os.getcwd()  # Get current working directory
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script

    if current_dir == script_dir:
        # Case where script is called inside its folder
        return os.path.realpath("graphviz.html")
    else:
        # Case where script is called from outside its folder
        return os.path.realpath(os.path.join("graphviz", "graphviz.html"))


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

    generate_graph_js(scene_graph, js_file)
    html_file_path = get_html_path()
    webbrowser.open("file://" + html_file_path)


def get_path(file):
    """Returns the correct path for the scene graph file."""
    current_dir = os.getcwd()  # Get current working directory
    script_dir = os.path.dirname(os.path.realpath(__file__))  # Directory of the script

    if current_dir == script_dir:
        # Case where script is called inside its folder
        return file
    else:
        # Case where script is called from outside its folder
        return os.path.join("graphviz", file)


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
    args = parser.parse_args()

    graph_path = get_path(args.graph)  # Get the adjusted graph path
    js_file_path = get_path(args.js_file)  # Ensure JS file is in graphviz folder
    print(graph_path, js_file_path)

    with open(graph_path) as graph_file:
        graph = json.load(graph_file)

    visualize_scene_graph(graph, js_file_path)
