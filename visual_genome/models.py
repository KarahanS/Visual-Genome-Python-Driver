"""
Visual Genome Python API wrapper, models
"""

global ATTRIBUTE_ID
ATTRIBUTE_ID = 0


class Image:
    """
    Image.
      ID         int
      url        hyperlink string
      width      int
      height     int
    """

    def __init__(self, id, url, width, height, coco_id, flickr_id):
        self.id = id
        self.url = url
        self.width = width
        self.height = height
        self.coco_id = coco_id
        self.flickr_id = flickr_id

    def __str__(self):
        return "id: %d, coco_id: %d, flickr_id: %d, width: %d, url: %s" % (
            self.id,
            -1 if self.coco_id is None else self.coco_id,
            -1 if self.flickr_id is None else self.flickr_id,
            self.width,
            self.url,
        )

    def __repr__(self):
        return str(self)


class Region:
    """
    Region.
      image 		   int
      phrase           string
      x                int
      y                int
      width            int
      height           int
    """

    def __init__(self, id, image, phrase, x, y, width, height):
        self.id = id
        self.image = image
        self.phrase = phrase
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def __str__(self):
        stat_str = (
            "region id: {0}, x: {1}, y: {2}, width: {3}, "
            "height: {4}, phrase: {5}, image id: {6}"
        )
        return stat_str.format(
            self.id, self.x, self.y, self.width, self.height, self.phrase, self.image.id
        )

    def __repr__(self):
        return str(self)


class Graph:
    """
    Graphs contain objects, relationships and attributes
      image            Image
      bboxes           Object array
      relationships    Relationship array
      attributes       Attribute array
    """

    def __init__(self, image, objects, relationships, attributes):
        self.image = image
        self.objects = objects
        self.relationships = relationships
        self.attributes = attributes


class Object:
    """
    Objects.
      id         int
      x          int
      y          int
      width      int
      height     int
      name       string
      synsets    list(Synset)
      image_id   int
      image_url  string

    (x, y) refers to the upper left corner of the bounding box
    """

    def __init__(self, id, x, y, width, height, names, synsets, image_id, image_url):
        self.id = id
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        # Raise error if there are multiple names
        if len(names) > 1:
            raise ValueError("Object cannot have multiple names.")

        self.name = (
            names[0] if names else "None"
        )  # Fallback to "None" if no names are provided
        self.synsets = synsets
        self.image_id = image_id
        self.image_url = image_url

    def __str__(self):
        stat_str = (
            "object id: {0}, x: {1}, y: {2}, width: {3}, "
            "height: {4}, name: {5}, image id: {6}, synset: {7}"
        )
        return stat_str.format(
            self.id,
            self.x,
            self.y,
            self.width,
            self.height,
            self.name,
            self.image_id,
            ", ".join(self.synsets) if len(self.synsets) > 0 else "None",
        )

    def __repr__(self):
        return str(self)


class Relationship:
    """
    Relationships. Ex, 'man - jumping over - fire hydrant'.
        subject    int
        predicate  string
        object     int
        rel_canon  Synset
    """

    def __init__(
        self,
        id,
        subject_id,
        subject_name,
        predicate,
        object_id,
        object_name,
        synset,
        image_id,
    ):
        self.id = id
        self.subject_id = subject_id
        self.subject_name = subject_name
        self.predicate = predicate
        self.object_id = object_id
        self.object_name = object_name
        self.synset = synset
        self.image_id = image_id

    def __str__(self):
        return "id: {0}: {1} {2} {3}, synset: {4}".format(
            self.id, self.subject_name, self.predicate, self.object_name, self.synset
        )

    def __repr__(self):
        return str(self)


class Attribute:
    """
    Attributes. Ex, 'man - old'.
      subject    Object
      attribute  string
      synset     Synset
    """

    # Class-level variable to keep track of the ID
    ATTRIBUTE_ID = 0

    def __init__(self, object_id, attribute, synset, image_id):
        self.id = Attribute.ATTRIBUTE_ID  # Assign the current ID
        Attribute.ATTRIBUTE_ID += 1  # Increment the ID for the next instance
        self.object_id = object_id
        self.attribute = attribute
        self.synset = synset
        self.image_id = image_id

    def __str__(self):
        return f"id: {self.id}, object_id: {self.object_id}, attribute: {self.attribute}, synset: {self.synset}, image_id: {self.image_id}"

    def __repr__(self):
        return str(self)


class QA:
    """
    Question Answer Pairs.
      ID         int
      image      int
      question   string
      answer     string
      q_objects  QAObject array
      a_objects  QAObject array
    """

    def __init__(self, id, image, question, answer, question_objects, answer_objects):
        self.id = id
        self.image = image
        self.question = question
        self.answer = answer
        self.q_objects = question_objects
        self.a_objects = answer_objects

    def __str__(self):
        return "id: %d, image: %d, question: %s, answer: %s" % (
            self.id,
            self.image.id,
            self.question,
            self.answer,
        )

    def __repr__(self):
        return str(self)


class QAObject:
    """
    Question Answer Objects are localized in the image and refer to a part
    of the question text or the answer text.
      start_idx          int
      end_idx            int
      name               string
      synset_name        string
      synset_definition  string
    """

    def __init__(self, start_idx, end_idx, name, synset):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.name = name
        self.synset = synset

    def __repr__(self):
        return str(self)


class Synset:
    """
    Wordnet Synsets.
      name       string
      definition string
    """

    def __init__(self, name, definition):
        self.name = name
        self.definition = definition

    def __str__(self):
        return "{} - {}".format(self.name, self.definition)

    def __repr__(self):
        return str(self)
