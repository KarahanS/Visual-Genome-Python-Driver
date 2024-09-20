import os
import gc
import json
import visual_genome.utils as utils
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PIL_Image
from io import BytesIO
from visual_genome.models import Image, Object, Attribute, Relationship, Graph, Synset
import deprecation
import numpy as np
import matplotlib.ticker as ticker
import random


class VisualGenome:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir

        print("Loading data...")

        self.attribute_synsets = self.get_synset_dictionary(
            "attribute_synsets.json", data_dir
        )

        images = self.get_all_image_data(data_dir)
        objects = self.get_all_objects(data_dir)
        region_descriptions = self.get_all_region_descriptions(data_dir)
        attributes = self.get_all_attributes(data_dir)
        relationships = self.get_all_relationships(data_dir)
        qas = self.get_all_qas(data_dir)

        self.IMAGES = {}
        self.REGIONS = {}
        self.OBJECTS = {}
        self.ATTRIBUTES = {}
        self.RELATIONSHIPS = {}

        for image in images:
            self.IMAGES[image.id] = {"image": image, "regions": [], "objects": []}

        for image_regions, image_objects, image_attributes, image_relationhips in zip(
            region_descriptions, objects, attributes, relationships
        ):
            # Process regions
            first_region = image_regions[0]
            image_id = first_region.image.id
            for region in image_regions:
                self.REGIONS[region.id] = region
            self.IMAGES[image_id]["regions"] = image_regions

            if len(image_objects) == 0:
                continue
            # Process objects
            for obj in image_objects:
                self.OBJECTS[obj.id] = {
                    "object": obj,
                    "attributes": [],
                    "relationships": [],
                }
            self.IMAGES[image_id]["objects"] = image_objects

            for attr in image_attributes:
                self.ATTRIBUTES[attr.id] = attr
                if (
                    attr.object_id in self.OBJECTS
                ):  # ignore attributes on merged objects
                    self.OBJECTS[attr.object_id]["attributes"].append(attr)
            for rel in image_relationhips:
                self.RELATIONSHIPS[rel.id] = rel
                if (
                    rel.subject_id in self.OBJECTS and rel.object_id in self.OBJECTS
                ):  # ignore relationships on merged objects
                    self.OBJECTS[rel.subject_id]["relationships"].append(rel)
                    self.OBJECTS[rel.object_id]["relationships"].append(rel)

        print("Data loaded.")

    def get_images(self):
        """
        Get all images.
        """
        return self.IMAGES

    def get_images_with_n_objects(self, n, at_least=False):
        """
        Get images with exactly n objects or with n or more objects.

        :param n: Number of objects to filter images by.
        :param at_least: If True, filter images with at least n objects. If False, filter images with exactly n objects.
        :return: List of images that meet the criteria.
        """
        if n < 0:
            raise ValueError(
                "The number of objects (n) must be a non-negative integer."
            )

        images = []
        for image_id in self.IMAGES:
            num_objects = len(self.IMAGES[image_id]["objects"])
            if (at_least and num_objects >= n) or (not at_least and num_objects == n):
                images.append(self.IMAGES[image_id]["image"])

        return images

    def get_image(self, id):
        return self.IMAGES[id]["image"]

    def get_object(self, id):
        return self.OBJECTS[id]["object"]

    def get_region(self, id):
        return self.REGIONS[id]

    def get_image_objects(self, im):
        if isinstance(im, Image):
            im = im.id
        return self.IMAGES[im]["objects"]

    def get_image_attributes(self, im):
        # Check if id is an instance of Image
        if isinstance(im, Image):
            im = im.id
            
        attributes = []
        for obj in self.IMAGES[im]["objects"]:
            # Ensure obj.id is valid and has attributes
            if obj.id in self.OBJECTS and self.OBJECTS[obj.id]["attributes"]:
                attributes.extend(self.OBJECTS[obj.id]["attributes"])
    
        return attributes

    def get_image_relationships(self, im):
        if isinstance(im, Image):
            im = im.id

        relationships = []
        id_set = set()
        for obj in self.IMAGES[im]["objects"]:
            rels = self.OBJECTS[obj.id]["relationships"]
            for rel in rels:
                if rel.id not in id_set:
                    relationships.append(rel)
                    id_set.add(rel.id)
        return relationships

    def get_image_regions(self, id):
        if isinstance(id, Image):
            id = id.id
        return self.IMAGES[id]["regions"]

    def get_all_image_data(self, data_dir=None):
        """
        Get all images
        Returns:
            List(Image) objects
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()
        data_file = os.path.join(data_dir, "image_data.json")
        data = json.load(open(data_file))
        return [utils.parse_image_data(image) for image in data]

    def generate_scene_graph_json(self, image_id):
        image = self.get_image(image_id)
        objects = self.get_image_objects(image_id)
        attributes = self.get_image_attributes(image_id)
        relationships = self.get_image_relationships(image_id)

        data = {"url": image.url, "objects": [], "attributes": [], "relationships": []}
        obj_id_to_index_map = {obj.id: i for i, obj in enumerate(objects)}
        for obj in objects:
            data["objects"].append({"name": obj.name})

        for attr in attributes:
            data["attributes"].append(
                {
                    "attribute": attr.attribute,
                    "object": obj_id_to_index_map[attr.object_id],
                }
            )

        for rel in relationships:
            data["relationships"].append(
                {
                    "subject": obj_id_to_index_map[rel.subject_id],
                    "predicate": rel.predicate,
                    "object": obj_id_to_index_map[rel.object_id],
                }
            )

        return data

    def sample_images(self, cnt=1):
        """
        Sample a range of images from the dataset.

        Parameters:
            cnt (int): Number of images to sample.

        Returns:
            List[Image]: A list of Image objects.
        """
        sample = [
            self.IMAGES[id]["image"] for id in np.random.choice(list(self.IMAGES), cnt)
        ]
        if len(sample) == 1:
            return sample[0]

    def sample_image_id(self, start=None, end=None):
        """
        Samples a random image ID from self.IMAGES within the specified range.

        :param start: Starting index (inclusive) of the range. Defaults to the minimum key.
        :param end: Ending index (exclusive) of the range. Defaults to the maximum key + 1.
        :return: Randomly sampled image ID from self.IMAGES.
        """
        # Determine the full range of keys
        all_keys = sorted(self.IMAGES.keys())

        # Set default start and end if not provided
        if start is None:
            start = all_keys[0]  # Set to the minimum key
        if end is None:
            end = all_keys[-1] + 1  # Set to the maximum key + 1 (exclusive)

        # Filter image IDs within the specified range
        valid_ids = [key for key in all_keys if start <= key < end]

        # Sample a random ID from the valid ones
        if valid_ids:
            return random.choice(valid_ids)
        else:
            raise ValueError(f"No image IDs found in the range {start} to {end}.")

    def display_random_image(self):
        """
        Display a random image from the dataset.
        """
        image = self.sample_images()
        self.display_image(image)

    def synset_objects_histogram(self, image_ids=None):
        """
        Plots a horizontal bar histogram of synsets for the specified type of data ('objects', 'attributes', or 'relationships').

        :param y: The type of data to plot ('objects', 'attributes', 'relationships').
        :param image_ids: Optional list of image IDs to include. If None, include all images.
        """

        # Get the synset counts for the specified data type
        if image_ids is None:  # Use all images
            synset_counts = [
                len(obj.synsets)
                for key in self.IMAGES
                for obj in self.IMAGES[key]["objects"]
            ]
        else:
            synset_counts = [
                len(obj.synsets)
                for id in image_ids
                for obj in self.IMAGES[id]["objects"]
            ]

        # Count the frequency of each unique synset count
        unique_counts, counts_frequency = np.unique(synset_counts, return_counts=True)

        # Plot the horizontal bar histogram
        plt.figure(figsize=(10, 6))
        plt.barh(unique_counts, counts_frequency)
        plt.xlabel("Frequency")
        plt.ylabel("Number of Synsets")
        plt.title(f"Histogram of Synset Counts for Objects")

        # Add value labels on bars, formatted with commas
        for index, value in enumerate(counts_frequency):
            plt.text(value, unique_counts[index], f"{value:,}", va="center")

        plt.tight_layout()
        plt.show()

    def get_synset_dictionary(self, synset_json, data_dir=None):
        """
        Get the dictionary of synsets.

        Output: Dict[str, Synset]
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()

        synset_file = os.path.join(data_dir, synset_json)
        synsets = json.load(open(synset_file))
        return synsets

    def histogram(self, y="regions", image_ids=None):
        """
        Plots a histogram for the specified type of data ('regions', 'objects', 'attributes', or 'relationships').

        :param y: The type of data to plot ('regions', 'objects', 'attributes', 'relationships').
        :param image_ids: Optional list of image IDs to include. If None, include all images.
        """
        if y not in ["regions", "objects", "attributes", "relationships"]:
            raise ValueError(
                f"Invalid value for 'y': {y}. Must be one of ['regions', 'objects', 'attributes', 'relationships']."
            )

        if y in ["regions", "objects"]:
            if image_ids is None:  # Use all images
                counts = [len(self.IMAGES[key].get(y, [])) for key in self.IMAGES]
            else:
                counts = [len(self.IMAGES.get(id, {}).get(y, [])) for id in image_ids]
        elif y in ["attributes", "relationships"]:
            counts = []
            if image_ids is None:
                for key in self.IMAGES:
                    cnt = 0
                    for obj in self.IMAGES[key].get("objects", []):
                        cnt += len(self.OBJECTS[obj.id].get(y, []))
                    if y == "relationships":
                        cnt = cnt / 2  # relationships are bidirectional
                    counts.append(cnt)
            else:
                for id in image_ids:
                    cnt = 0
                    for obj in self.IMAGES[id].get("objects", []):
                        cnt += len(self.OBJECTS[obj.id].get(y, []))
                    if y == "relationships":
                        cnt = cnt / 2  # relationships are bidirectional
                    counts.append(cnt)
        # Compute histogram bin edges
        max_count = max(counts, default=0)
        bin_edges = np.arange(
            0, max_count + 5, 5
        )  # Bin edges starting from 0 to max_count + 5
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers

        # Plot the histogram
        plt.hist(counts, bins=bin_edges, edgecolor="black", alpha=0.7)
        plt.axvline(np.mean(counts), color="r", linestyle="dashed", linewidth=1)
        plt.text(
            np.mean(counts),
            plt.ylim()[1] * 0.9,
            f"Average: {np.mean(counts):.2f}",
            rotation=0,
            verticalalignment="center",
        )

        plt.xlabel(f"Number of {y}")
        plt.ylabel("Number of images")
        plt.title(f"Histogram of number of {y} in images")

        # Adjust x-ticks to be centered on bins
        plt.xticks(ticks=bin_centers)
        plt.xticks([])
        plt.show()

    def object_histogram(self, y):
        """
        Plots a histogram of the number of attributes per object.

        :param image_ids: Optional list of image IDs to include. If None, include all images.
        """
        if y not in ["relationships", "attributes"]:
            raise ValueError(
                f"Invalid value for 'y': {y}. Must be one of ['relationships', 'attributes']."
            )
        counts = {}

        for key in self.OBJECTS:
            cnt = len(self.OBJECTS[key][y])
            if cnt in counts:
                counts[cnt] += 1
            else:
                counts[cnt] = 1

        # Create a list of counts and frequencies
        y_counts = list(counts.keys())
        frequencies = list(counts.values())

        # Format numbers with commas
        def format_number(x, _):
            return f"{int(x):,}"  # Format the number as an integer with commas

        if y == "relationships":
            y_counts = y_counts[:20]
            frequencies = frequencies[:20]
        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            y_counts, frequencies, color="blue", edgecolor="black", alpha=0.7
        )
        plt.xlabel(f"Number of {y}")
        plt.ylabel("Number of Objects")
        plt.title(f"Number of Objects vs. Number of {y}")

        # Add value labels on bars with formatted numbers
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                format_number(yval, None),
                ha="center",
                va="bottom",
            )

        # Set y-axis formatter
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_number))

        plt.tight_layout()
        plt.show()

    def visualize_cluster(self, df,  cluster, n=3):
        # Get nxn images from this cluster
        cluster_df = df[df["cluster"] == cluster]
        image_ids = cluster_df["Image_id"].tolist()

        # Randomly sample nxn images from image_ids
        sampled_ids = np.random.choice(image_ids, n * n, replace=False)
        images = [self.IMAGES[image_id]["image"] for image_id in sampled_ids]

        fig, axes = plt.subplots(
            n, n, figsize=(n**2, n**2)
        )  # Create a nxn grid of subplots
        axes = axes.flatten()  # Flatten axes array for easy iteration

        for i, ax in enumerate(axes):
            response = requests.get(images[i].url)
            img = PIL_Image.open(BytesIO(response.content))
            ax.imshow(img)

            # Hide grid lines and axes ticks
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    @deprecation.deprecated(
        details="""This function will iterate through all the images until it finds the one requested.
        Instead, instantiate a `VisualGenome` object. Then you can access the image with `get_image(id)`.""",
    )
    def get_image_data_with_id(self, id=61512, data_dir=None):
        """
        Get a single image with ID
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()

        data_file = os.path.join(data_dir, "image_data.json")
        data = json.load(open(data_file))
        for image in data:
            img_data = utils.parse_image_data(image)
            if img_data.id == id:
                return img_data
        print("Image not found")
        return None

    def display_image(self, image):
        """
        Display an image.

        Parameters:
            image (Image object, int, or str): An Image object, an integer representing an Image ID,
                                                or a string representing a URL.

        Raises:
            TypeError: If the input is not an Image object, an integer, or a string (URL).
            ValueError: If the URL is invalid or there is an issue with the request.
        """
        url = None
        # Check if the input is an Image object, URL string, or integer (ID)
        if isinstance(image, Image):  # Custom Image object
            url = image.url
        elif isinstance(image, str):  # URL string
            url = image
        elif isinstance(image, int):  # Image ID (int)
            if image in self.IMAGES:
                url = self.IMAGES[image]["image"].url
            else:
                raise ValueError(
                    f"Image ID {image} does not exist in the IMAGES collection."
                )
        else:
            raise TypeError(
                "Input should be either an Image object, Image ID (int), or a URL string."
            )

        # Check if the URL is valid and working
        try:
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Failed to retrieve image from URL: {url}\nError: {e}")

        # Display the image
        img = PIL_Image.open(BytesIO(response.content))
        plt.imshow(img)
        plt.axis("off")  # Hide axis
        plt.show()

    def get_all_region_descriptions(self, data_dir=None):
        """
        Get all region descriptions.

        Output: List[List[Region]] --> First list has 108,077 elements (number of images).
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()

        data_file = os.path.join(data_dir, "region_descriptions.json")
        image_data = self.get_all_image_data(data_dir)

        # Create a map of image IDs to image data
        image_map = {d.id: d for d in image_data}

        # Load JSON data using a context manager
        with open(data_file, "r") as file:
            images = json.load(file)

        # Parse region descriptions for each image
        return [
            utils.parse_region_descriptions(image["regions"], image_map[image["id"]])
            for image in images
        ]

    def get_all_relationships(self, data_dir=None):
        """
        Get all relationships.

        Output: List[List[Relationship]] --> First list has 108,077 elements (number of images).
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()

        data_file = os.path.join(data_dir, "relationships.json")
        images = json.load(open(data_file))

        relationships = []
        for image in images:
            image_id = image["image_id"]
            relationships.append(
                utils.parse_relationships(image["relationships"], image_id)
            )

        return relationships

    @deprecation.deprecated(
        details="""This function iterates through all images to find the requested one.
        Instead, instantiate a `VisualGenome` object. Then you can access the image with `get_image_regions(id)`."""
    )
    def get_region_descriptions_of_image_with_id(self, image_id=61512, data_dir=None):
        """
        Get region descriptions of a specific image.

        Args:
            image_id (int): The ID of the image.
            data_dir (str): The directory containing the data file.

        Returns:
            List[Region] or None: A list of region descriptions or None if not found.
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()

        data_file = os.path.join(data_dir, "region_descriptions.json")

        # Use a context manager for safe file handling
        with open(data_file, "r") as file:
            images = json.load(file)

        for image in images:
            if image["id"] == image_id:
                return utils.parse_region_descriptions(
                    image["regions"], self.get_image_data_with_id(image_id)
                )

        print("Image not found.")
        return None

    def get_all_attributes(self, data_dir=None):
        """
        Get all attributes.

        Output: List[List[Attribute]] --> First list has 108,077 elements (number of images).
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()

        data_file = os.path.join(data_dir, "attributes.json")
        images = json.load(open(data_file))

        attributes = []
        for image in images:
            image_id = image["image_id"]
            attributes.append(
                utils.parse_attributes(
                    image["attributes"], image_id, self.attribute_synsets
                )
            )

        return attributes

    def get_all_objects(self, data_dir=None):
        """
        images:  {image_id, objects, image_url}
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()

        data_file = os.path.join(data_dir, "objects.json")
        images = json.load(open(data_file))

        output = []
        for image in images:
            image_url = image["image_url"] if "image_url" in image else None
            output.append(
                utils.parse_objects(image["objects"], image["image_id"], image_url)
            )
        return output

    def get_objects_of_image(self, id=61512, data_dir=None):
        if data_dir is None:
            data_dir = utils.get_data_dir()

        data_file = os.path.join(data_dir, "objects.json")
        images = json.load(open(data_file))
        for image in images:
            if image["image_id"] == id:
                return utils.parse_objects(image["objects"], id, image["image_url"])
        print("Image not found")
        return None

    def get_all_qas(self, data_dir=None):
        """
        Get all question answers.
        """
        if data_dir is None:
            data_dir = utils.get_data_dir()
        data_file = os.path.join(data_dir, "question_answers.json")
        image_data = self.get_all_image_data(data_dir)
        image_map = {}
        for d in image_data:
            image_map[d.id] = d
        images = json.load(open(data_file))
        output = []
        for image in images:
            output.append(utils.parse_QA(image["qas"], image_map))
        return output

    def visualize_regions(self, image_id, end=None):
        """
        Input:
            image:  Image object
            regions:  List[Region] (list of regions for the given image)
        """
        image = self.IMAGES[image_id]["image"]
        regions = self.IMAGES[image_id]["regions"]

        regions = regions[:end] if end else regions  # visualize up to end

        fig = plt.gcf()
        fig.set_size_inches(9, 5)
        response = requests.get(image.url)
        img = PIL_Image.open(BytesIO(response.content))
        plt.imshow(img)
        ax = plt.gca()
        if end != 0:  # if end = 0, don't display anything
            for region in regions:
                ax.add_patch(
                    Rectangle(
                        (region.x, region.y),
                        region.width,
                        region.height,
                        fill=False,
                        edgecolor="red",
                        linewidth=3,
                    )
                )
                ax.text(
                    region.x,
                    region.y,
                    region.phrase,
                    style="italic",
                    bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
                )
        fig = plt.gcf()
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.show()

    def visualize_images_side_by_side(self, images, n):
        # visualize n images side by side
        fig, axes = plt.subplots(
            n, n, figsize=(n**2, n**2)
        )  # Create a 3x3 grid of subplots
        axes = axes.flatten()  # Flatten axes array for easy iteration

        for i, ax in enumerate(axes):
            response = requests.get(images[i].url)
            img = PIL_Image.open(BytesIO(response.content))
            ax.imshow(img)

            # Hide grid lines and axes ticks
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()

    def visualize_objects(
        self, image_id, end=None, desc=True, synsets=True, attributes=True
    ):
        """
        Visualizes objects for a given image.

        Parameters:
        image_id (int): The ID of the image to visualize.
        end (int, optional): The number of objects to display. If None, all objects will be visualized.
                             If 0, no objects will be displayed.
        desc (bool, optional): If True, the object names and descriptions will be displayed. Defaults to True.
        synsets (bool, optional): If True, the synsets of the object will be displayed along with the name.
                                  If False, only the object names will be shown. Defaults to True.
        """
        image = self.IMAGES[image_id]["image"]
        objects = self.IMAGES[image_id]["objects"]

        objects = objects[:end] if end else objects  # visualize up to end

        fig = plt.gcf()
        fig.set_size_inches(9, 5)
        response = requests.get(image.url)
        img = PIL_Image.open(BytesIO(response.content))
        plt.imshow(img)
        ax = plt.gca()
        if end != 0:  # if end = 0, don't display anything
            for obj in objects:
                ax.add_patch(
                    Rectangle(
                        (obj.x, obj.y),
                        obj.width,
                        obj.height,
                        fill=False,
                        edgecolor="red",
                        linewidth=3,
                    )
                )

                # Display the object name and synsets
                if desc is True:
                    to_write = ""
                    if synsets is True:
                        # Handle synsets formatting
                        if obj.synsets:
                            synsets_str = ", ".join(
                                obj.synsets
                            )  # Join the list into a string
                        else:
                            synsets_str = "No synsets"

                        to_write = f"{obj.name}\n{synsets_str}"
                    else:
                        to_write = f"{obj.name}"

                    if attributes is True:
                        # Display the object attributes
                        if self.OBJECTS[obj.id]["attributes"]:
                            attr_str = ", ".join(
                                [
                                    attr.attribute
                                    for attr in self.OBJECTS[obj.id]["attributes"]
                                ]
                            )

                            to_write += f"\nAttributes: {attr_str}"
                    ax.text(
                        obj.x,
                        obj.y,
                        to_write,
                        style="italic",
                        bbox={"facecolor": "white", "alpha": 0.7, "pad": 10},
                    )
        fig = plt.gcf()
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.show()

    # --------------------------------------------------------------------------------------------------
    # get_scene_graphs and sub-methods

    def get_scene_graph(
        self,
        image_id,
        images="data/",
        image_data_dir="data/by-id/",
        synset_file="data/synsets.json",
    ):
        """
        Load a single scene graph from a .json file.
        """
        if type(images) is str:
            # Instead of a string, we can pass this dict as the argument `images`
            images = {img.id: img for img in self.get_all_image_data(images)}

        fname = str(image_id) + ".json"
        image = images[image_id]
        data = json.load(open(image_data_dir + fname, "r"))

        scene_graph = self.parse_graph_local(data, image)
        scene_graph = self.init_synsets(scene_graph, synset_file)
        return scene_graph

    def get_scene_graphs(
        self,
        start_index=0,
        end_index=-1,
        data_dir="data/",
        image_data_dir="data/by-id/",
        min_rels=0,
        max_rels=100,
    ):
        """
        Get scene graphs given locally stored .json files;
        requires `save_scene_graphs_by_id`.

        start_index, end_index : get scene graphs listed by image,
                            from start_index through end_index
        data_dir : directory with `image_data.json` and `synsets.json`
        image_data_dir : directory of scene graph jsons saved by image id
                    (see `save_scene_graphs_by_id`)
        min_rels, max_rels: only get scene graphs with at least / less
                        than this number of relationships
        """
        images = {img.id: img for img in self.get_all_image_data(data_dir)}
        scene_graphs = []

        img_fnames = os.listdir(image_data_dir)
        if end_index < 1:
            end_index = len(img_fnames)

        for fname in img_fnames[start_index:end_index]:
            image_id = int(fname.split(".")[0])
            scene_graph = self.get_scene_graph(
                image_id, images, image_data_dir, data_dir + "synsets.json"
            )
            n_rels = len(scene_graph.relationships)
            if min_rels <= n_rels <= max_rels:
                scene_graphs.append(scene_graph)

        return scene_graphs

    def map_object(self, object_map, obj):
        """
        Use object ids as hashes to `visual_genome.models.Object` instances.
        If item not in table, create new `Object`. Used when building
        scene graphs from json.
        """

        oid = obj["object_id"]
        obj["id"] = oid
        del obj["object_id"]

        if oid in object_map:
            object_ = object_map[oid]

        else:
            if "attributes" in obj:
                attrs = obj["attributes"]
                del obj["attributes"]
            else:
                attrs = []
            if "w" in obj:
                obj["width"] = obj["w"]
                obj["height"] = obj["h"]
                del obj["w"], obj["h"]

            object_ = Object(**obj)

            object_.attributes = attrs
            object_map[oid] = object_

        return object_map, object_

    global count_skips
    count_skips = [0, 0]

    def parse_graph_local(self, data, image, verbose=False):
        """
        Modified version of `utils.ParseGraph`.
        """
        global count_skips
        objects = []
        object_map = {}
        relationships = []
        attributes = []

        for obj in data["objects"]:
            object_map, o_ = self.map_object(object_map, obj)
            objects.append(o_)
        for rel in data["relationships"]:
            if rel["subject_id"] in object_map and rel["object_id"] in object_map:
                object_map, s = self.map_object(
                    object_map, {"object_id": rel["subject_id"]}
                )
                v = rel["predicate"]
                object_map, o = self.map_object(
                    object_map, {"object_id": rel["object_id"]}
                )
                rid = rel["relationship_id"]
                relationships.append(Relationship(rid, s, v, o, rel["synsets"]))
            else:
                # Skip this relationship if we don't have the subject and object in
                # the object_map for this scene graph. Some data is missing in this
                # way.
                count_skips[0] += 1
        if "attributes" in data:
            for attr in data["attributes"]:
                a = attr["attribute"]
                if a["object_id"] in object_map:
                    attributes.append(
                        Attribute(
                            attr["attribute_id"],
                            Object(
                                a["object_id"],
                                a["x"],
                                a["y"],
                                a["w"],
                                a["h"],
                                a["names"],
                                a["synsets"],
                            ),
                            a["attributes"],
                            a["synsets"],
                        )
                    )
                else:
                    count_skips[1] += 1
        if verbose:
            print("Skipped {} rels, {} attrs total".format(*count_skips))
        return Graph(image, objects, relationships, attributes)

    def init_synsets(self, scene_graph, synset_file):
        """
        Convert synsets in a scene graph from strings to Synset objects.
        """
        syn_data = json.load(open(synset_file, "r"))
        syn_class = {
            s["synset_name"]: Synset(s["synset_name"], s["synset_definition"])
            for s in syn_data
        }

        for obj in scene_graph.objects:
            obj.synsets = [syn_class[sn] for sn in obj.synsets]
        for rel in scene_graph.relationships:
            rel.synset = [syn_class[sn] for sn in rel.synset]
        for attr in scene_graph.attributes:
            obj.synset = [syn_class[sn] for sn in attr.synset]

        return scene_graph

    # --------------------------------------------------------------------------------------------------
    # This is a pre-processing step that only needs to be executed once.
    # You can download .jsons segmented with these methods from:
    #     https://drive.google.com/file/d/0Bygumy5BKFtcQ1JrcFpyQWdaQWM

    def save_scene_graphs_by_id(self, data_dir="data/", image_data_dir="data/by-id/"):
        """
        Save a separate .json file for each image id in `image_data_dir`.

        Notes
        -----
        - If we don't save .json's by id, `scene_graphs.json` is >6G in RAM
        - Separated .json files are ~1.1G on disk
        - Run `add_attrs_to_scene_graphs` before `parse_graph_local` will work
        - Attributes are only present in objects, and do not have synset info

        Each output .json has the following keys:
        - "id"
        - "objects"
        - "relationships"
        """
        if not os.path.exists(image_data_dir):
            os.mkdir(image_data_dir)

        all_data = json.load(open(os.path.join(data_dir, "scene_graphs.json")))
        for sg_data in all_data:
            img_fname = str(sg_data["image_id"]) + ".json"
            with open(os.path.join(image_data_dir, img_fname), "w") as f:
                json.dump(sg_data, f)

        del all_data
        gc.collect()  # clear memory

    def add_attrs_to_scene_graphs(self, data_dir="data/"):
        """
        Add attributes to `scene_graph.json`, extracted from `attributes.json`.

        This also adds a unique id to each attribute, and separates individual
        attibutes for each object (these are grouped in `attributes.json`).
        """
        attr_data = json.load(open(os.path.join(data_dir, "attributes.json")))
        with open(os.path.join(data_dir, "scene_graphs.json")) as f:
            sg_dict = {sg["image_id"]: sg for sg in json.load(f)}

        id_count = 0
        for img_attrs in attr_data:
            attrs = []
            for attribute in img_attrs["attributes"]:
                a = img_attrs.copy()
                del a["attributes"]
                a["attribute"] = attribute
                a["attribute_id"] = id_count
                attrs.append(a)
                id_count += 1
            iid = img_attrs["image_id"]
            sg_dict[iid]["attributes"] = attrs

        with open(os.path.join(data_dir, "scene_graphs.json"), "w") as f:
            json.dump(sg_dict.values(), f)
        del attr_data, sg_dict
        gc.collect()

    # --------------------------------------------------------------------------------------------------
    # For info on VRD dataset, see:
    #   http://cs.stanford.edu/people/ranjaykrishna/vrd/

    def get_scene_graphs_VRD(self, json_file="data/vrd/json/test.json"):
        """
        Load VRD dataset into scene graph format.
        """
        scene_graphs = []
        with open(json_file, "r") as f:
            D = json.load(f)

        scene_graphs = [self.parse_graph_VRD(d) for d in D]
        return scene_graphs

    def parse_graph_VRD(self, d):
        image = Image(d["photo_id"], d["filename"], d["width"], d["height"], "", "")

        id2obj = {}
        objs = []
        rels = []
        atrs = []

        for i, o in enumerate(d["objects"]):
            b = o["bbox"]
            obj = Object(i, b["x"], b["y"], b["w"], b["h"], o["names"], [])
            id2obj[i] = obj
            objs.append(obj)

            for j, a in enumerate(o["attributes"]):
                atrs.append(Attribute(j, obj, a["attribute"], []))

        for i, r in enumerate(d["relationships"]):
            s = id2obj[r["objects"][0]]
            o = id2obj[r["objects"][1]]
            v = r["relationship"]
            rels.append(Relationship(i, s, v, o, []))

        return Graph(image, objs, rels, atrs)
