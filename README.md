# Visual Genome Python Driver

This repository contains a Python driver for the [Visual Genome dataset](https://visualgenome.org/), an extensive dataset and knowledge base designed to connect structured image concepts with language. Visual Genome includes 108,077 images annotated with objects, attributes, relationships, and region descriptions.

The main goal of this repository is to offer a simple, easy-to-use interface for:
- Visualizing images and their annotations, including scene graphs.
- Performing exploratory data analysis.
- Applying sentence clustering to categorize the dataset.

This driver enhances functionality from the original [Visual Genome Python Driver](https://github.com/ranjaykrishna/visual_genome_python_driver) and incorporates a Graph Visualization tool based on [Graphviz](https://github.com/ranjaykrishna/GraphViz).


### Installation

Firstly, clone the repository into your local machine.

```bash
git clone 
```

As the official API is not supported, you have to download the following data files from the Visual Genome website:
- [image_data.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip)
- [objects.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip)
- [region_descriptions.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip)
- [attributes.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip)
- [relationships.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip)
- [attributes_synsets.json](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attribute_synsets.json.zip)

Please locate all your downloaded files in a directory called `data/` in the root of the repository. In total, it should take up ~2.5GB of space on disk. You can directly run the `setup.py` to download the files and locate them in the correct directory.

```bash
python setup.py
```


### Usage

Data loading, querying, and visualization are all done through the `VisualGenome` class. To visualize the images, please have a look at the `demo.ipynb` notebook. There are some statistics about the dataset in the `analysis.ipynb` notebook.


### Clustering

We also offer a clustering approach for the images in the dataset, based on the K-Means algorithm. This method utilizes [sentence transformers](https://sbert.net/) to vectorize the region descriptions of the images. The results can be found in the `clustering.ipynb` notebook.