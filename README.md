# SPECTR: Serum Protein Electrophoresis Computer-Assisted Recognition Artificial Intelligence

Find the online SPECTR application at [https://spectr.shinyapps.io/SPECTR](https://spectr.shinyapps.io/SPECTR)

Find the full original article at [https://doi.org/10.1093/clinchem/hvab133](https://academic.oup.com/clinchem/advance-article/doi/10.1093/clinchem/hvab133/6365844)

## Achieving Expert-Level Interpretation of Serum Protein Electrophoresis through Deep Learning Driven by Human Reasoning

<strong>Authors</strong>

Floris Chabrun<sup>*†,1,2</sup>, Xavier Dieu<sup>†,1,2</sup>, Marc Ferré<sup>2</sup>, Olivier Gaillard<sup>3</sup>, Anthony Mery<sup>1</sup>, Juan Manuel Chao de la Barca<sup>1,2</sup>, Audrey Taisne<sup>1</sup>, Geoffrey Urbanski<sup>2,4</sup>, Pascal Reynier<sup>§,1,2</sup>, Delphine Mirebeau-Prunier<sup>§,1,2</sup>

<sup>†</sup>and <sup>§</sup> contributed equally to this manuscript

<strong>Affiliations</strong>

<sup>1</sup>Laboratoire de Biochimie et Biologie moléculaire, Centre Hospitalier Universitaire d’Angers, France

<sup>2</sup>Unité Mixte de Recherche (UMR) MITOVASC, Centre National de la Recherche Scientifique (CNRS) 6015, Institut National de la Santé et de la recherche Médicale (INSERM) U1083, Université d’Angers, France

<sup>3</sup>Laboratoire de Biochimie, Centre Hospitalier du Mans, France

<sup>4</sup>Service de Médecine Interne et Immunologie Clinique, Centre Hospitalier Universitaire d’Angers, France

## Abstract

### Background

Serum protein electrophoresis (SPE) is a common clinical laboratory test, mainly indicated for the diagnosis and follow-up of monoclonal gammopathies. A time-consuming and potentially subjective human expertise is required for SPE analysis to detect possible pitfalls and to provide a clinically relevant interpretation.

### Methods

An expert-annotated SPE dataset of 159 969 entries was used to develop SPECTR (serum protein electrophoresis computer-assisted recognition), a deep learning-based artificial intelligence, which analyzes and interprets raw SPE curves produced by an analytical system into text comments that can be used by practitioners. It was designed following academic recommendations for SPE interpretation, using a transparent architecture avoiding the “black box” effect. SPECTR was validated on an external, independent cohort of 70 362 SPEs and challenged by a panel of 9 independent experts from other hospital centers.

### Results

SPECTR was able to identify accurately both quantitative abnormalities (r ≥ 0.98 for fractions quantification) and qualitative abnormalities [receiver operating characteristic–area under curve (ROC–AUC) ≥ 0.90 for M-spikes, restricted heterogeneity of immunoglobulins, and beta-gamma bridging]. Furthermore, it showed highly accurate at both detecting (ROC–AUC ≥ 0.99) and quantifying (r = 0.99) M-spikes. It proved highly reproducible and resilient to minor variations and its agreement with human experts was higher (κ = 0.632) than experts between each other (κ = 0.624).

### Conclusions

SPECTR is an algorithm based on artificial intelligence suitable to high-throughput SPEs analyses and interpretation. It aims at improving SPE reproducibility and reliability. It is freely available in open access through an online tool providing fully editable validation assistance for SPE.

Find the full original article at [https://doi.org/10.1093/clinchem/hvab133](https://academic.oup.com/clinchem/advance-article/doi/10.1093/clinchem/hvab133/6365844)

## Content

### /SPECTR
The app.R script contains the SPECTR application, including the expert system and the user interface.

The ai.py file is called during its execution, for performing the deep learning inference.

Please note that pre-trained deep learning models are necessary to execute this application (see /models_training).

### /models_training
Python scripts used for training the various deep learning models, as well as a small example dataset with various examples of SPE patterns.
