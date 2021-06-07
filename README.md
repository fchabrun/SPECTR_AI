# SPECTR: Serum Protein Electrophoresis Computer-Assisted Recognition Artificial Intelligence

Achieving Expert-Level Interpretation of Serum Protein Electrophoresis through Deep Learning Driven by Human Reasoning

<strong>Authors</strong>

Floris Chabrun*†,1,2, Xavier Dieu†,1,2, Marc Ferré2, Olivier Gaillard3, Anthony Mery1, Juan Manuel Chao de la Barca1,2, Audrey Taisne1, Geoffrey Urbanski2,4, Pascal Reynier§,1,2, Delphine Mirebeau-Prunier§,1,2


†and § contributed equally to this manuscript


<strong>Affiliations</strong>


1Laboratoire de Biochimie et Biologie moléculaire, Centre Hospitalier Universitaire d’Angers, France

2Unité Mixte de Recherche (UMR) MITOVASC, Centre National de la Recherche Scientifique (CNRS) 6015, Institut National de la Santé et de la recherche Médicale (INSERM) U1083, Université d’Angers, France

3Laboratoire de Biochimie, Centre Hospitalier du Mans, France

4Service de Médecine Interne et Immunologie Clinique, Centre Hospitalier Universitaire d’Angers, France

*Corresponding author: Chabrun Floris, Laboratoire de Biochimie et Biologie moléculaire, Centre Hospitalier Universitaire d’Angers, 4 rue Larrey, 49933 Angers Cedex 9, France floris.chabrun@chu-angers.fr

## Content

### /SPECTR
The app.R script contains the SPECTR application, including the expert system and the user interface.

The ai.py file is called during its execution, for performing the deep learning inference.

Please note that pre-trained deep learning models are necessary to execute this application (see /models_training).

### /models_training
Python scripts used for training the various deep learning models, as well as a small example dataset with various examples of SPE patterns.
