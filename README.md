# SPECTR: Serum Protein Electrophoresis Computer-Assisted Recognition Artificial Intelligence

Achieving Expert-Level Interpretation of Serum Protein Electrophoresis through Deep Learning Driven by Human Reasoning

<strong>Authors</strong>

Floris Chabrun<sup>*†,1,2</sup>, Xavier Dieu<sup>†,1,2</sup>, Marc Ferré<sup>2</sup>, Olivier Gaillard<sup>3</sup>, Anthony Mery<sup>1</sup>, Juan Manuel Chao de la Barca<sup>1,2</sup>, Audrey Taisne<sup>1</sup>, Geoffrey Urbanski<sup>2,4</sup>, Pascal Reynier<sup>§,1,2</sup>, Delphine Mirebeau-Prunier<sup>§,1,2</sup>

<sup>†</sup>and <sup>§</sup> contributed equally to this manuscript

<strong>Affiliations</strong>

<sup>1</sup>Laboratoire de Biochimie et Biologie moléculaire, Centre Hospitalier Universitaire d’Angers, France

<sup>2</sup>Unité Mixte de Recherche (UMR) MITOVASC, Centre National de la Recherche Scientifique (CNRS) 6015, Institut National de la Santé et de la recherche Médicale (INSERM) U1083, Université d’Angers, France

<sup>3</sup>Laboratoire de Biochimie, Centre Hospitalier du Mans, France

<sup>4</sup>Service de Médecine Interne et Immunologie Clinique, Centre Hospitalier Universitaire d’Angers, France

## Content

### /SPECTR
The app.R script contains the SPECTR application, including the expert system and the user interface.

The ai.py file is called during its execution, for performing the deep learning inference.

Please note that pre-trained deep learning models are necessary to execute this application (see /models_training).

### /models_training
Python scripts used for training the various deep learning models, as well as a small example dataset with various examples of SPE patterns.
