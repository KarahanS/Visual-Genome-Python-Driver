import requests
from os.path import dirname, realpath, join
from visual_genome.models import (
    Image,
    Object,
    Attribute,
    Relationship,
    Region,
    Graph,
    QA,
    QAObject,
    Synset,
)


def get_data_dir():
    """
    Get the local directory where the Visual Genome data is locally stored.
    """
    data_dir = join(dirname(realpath("__file__")), "data")
    return data_dir


def parse_synset(canon):
    """
    Helper to Extract Synset from canon object.
    """
    if len(canon) == 0:
        return None
    return Synset(canon[0]["synset_name"], canon[0]["synset_definition"])


def parse_graph(data, image):
    """
    Helper to parse a Graph object from API data.
    """
    objects = []
    object_map = {}
    relationships = []
    attributes = []
    # Create the Objects
    for obj in data["bounding_boxes"]:
        names = []
        synsets = []
        for bbx_obj in obj["boxed_objects"]:
            names.append(bbx_obj["name"])
            synsets.append(parse_synset(bbx_obj["object_canon"]))
            object_ = Object(
                obj["id"],
                obj["x"],
                obj["y"],
                obj["width"],
                obj["height"],
                names,
                synsets,
            )
            object_map[obj["id"]] = object_
        objects.append(object_)
    # Create the Relationships
    for rel in data["relationships"]:
        relationships.append(
            Relationship(
                rel["id"],
                object_map[rel["subject"]],
                rel["predicate"],
                object_map[rel["object"]],
                parse_synset(rel["relationship_canon"]),
            )
        )
    # Create the Attributes
    for atr in data["attributes"]:
        attributes.append(
            Attribute(
                atr["id"],
                object_map[atr["subject"]],
                atr["attribute"],
                parse_synset(atr["attribute_canon"]),
            )
        )
    return Graph(image, objects, relationships, attributes)


def parse_image_data(data):
    """
    Helper to parse the image data for one image.
    """
    img_id = data["id"] if "id" in data else data["image_id"]
    url = data["url"]
    width = data["width"]
    height = data["height"]
    coco_id = data["coco_id"]
    flickr_id = data["flickr_id"]
    image = Image(img_id, url, width, height, coco_id, flickr_id)
    return image


def parse_region_descriptions(data, image):
    """
    Helper to parse region descriptions.
    """
    regions = []
    if "region_id" in data[0]:
        region_id_key = "region_id"
    else:
        region_id_key = "id"
    for info in data:
        regions.append(
            Region(
                info[region_id_key],
                image,
                info["phrase"],
                info["x"],
                info["y"],
                info["width"],
                info["height"],
            )
        )
    return regions


def parse_attributes(data, image_id, attribute_synsets):
    """
    Helper to parse attributes.
    """
    attributes = []
    for info in data:
        if "attributes" in info:  # it might be just an object without attributes
            for attr in info["attributes"]:
                synset = attribute_synsets.get(
                    attr, None
                )  # it may not be mapped to a specific synset
                attributes.append(
                    Attribute(
                        attribute=attr,
                        object_id=info["object_id"],
                        synset=synset,  # mapping from attr to synset
                        image_id=image_id,
                    )
                )
    return attributes


def parse_relationships(data, image_id, relationship_synsets):
    """
    Helper to parse relationships.
    """
    relationships = []
    for info in data:
        if "names" in info["object"]:
            object_name = info["object"]["names"][0]
        else:
            object_name = info["object"]["name"]

        if "names" in info["subject"]:
            subject_name = info["subject"]["names"][0]
        else:
            subject_name = info["subject"]["name"]
        synset = relationship_synsets.get(info["predicate"].lower(), None)
        if synset is None and len(info["synsets"]) > 0:
            synset = info["synsets"][0]
        relationships.append(
            Relationship(
                info["relationship_id"],
                info["subject"]["object_id"],
                subject_name,
                info["predicate"],
                info["object"]["object_id"],
                object_name,
                synset,
                image_id,
            )
        )
    return relationships


def parse_objects(data, image_id, image_url, object_synsets):
    """
    Helper to parse objects.
    """
    objects = []
    # data = [objects]
    for info in data:
        # experimental
        synset = object_synsets.get(info["names"][0].lower(), None)
        if synset is None and len(info["synsets"]) > 0:
            synset = info["synsets"][0]
        objects.append(
            Object(
                info["object_id"],
                info["x"],
                info["y"],
                info["w"],
                info["h"],
                info["names"],
                synset,
                image_id,
                image_url,
            )
        )

    return objects


def parse_QA(data, image_map):
    """
    Helper to parse a list of question answers.
    """
    qas = []
    for info in data:
        qos = []
        aos = []
        if "question_objects" in info:
            for qo in info["question_objects"]:
                synset = Synset(qo["synset_name"], qo["synset_definition"])
                qos.append(
                    QAObject(
                        qo["entity_idx_start"],
                        qo["entity_idx_end"],
                        qo["entity_name"],
                        synset,
                    )
                )
        if "answer_objects" in info:
            for ao in info["answer_objects"]:
                synset = Synset(ao["synset_name"], ao["synset_definition"])
                aos.append(
                    QAObject(
                        ao["entity_idx_start"],
                        ao["entity_idx_end"],
                        ao["entity_name"],
                        synset,
                    )
                )
        qas.append(
            QA(
                info["qa_id"],
                image_map[info["image_id"]],
                info["question"],
                info["answer"],
                qos,
                aos,
            )
        )
    return qas
