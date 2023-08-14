# IR CSV Kinetics Extractor

![spectra](exports\laser-loading-time-spectra.png "Lased PDMS: loading vs time spectra")

![scatter](exports\laser-loading-time-scatterfit.jpg "Lased PDMS: loading vs time scatterfit")

## to use:
- format CSVs in input folder as follows: "cure-condition_agent-loading_time-in-s.csv", e.g. "laser-15W/cm2_5e-3-CB_20.csv"

## General Graphing
- Spectra (raw)
- Spectra (corrected)
- Peak integrations vs time bar chart
- Si-H integration vs time kinetics-fit scatterplot

## To-do
 - confirm refactor usability
 - standardize the input csv convention (working version: "condition_time-in-s.csv", e.g. "5e-3_20.csv")
 - procedurally check for conditions
  - cure (ambient, laser, oven) and photothermal agent (none, AuNP, CB)
 - replace color maps with condition conventions listed below
 - separate steps 
    - script 1: consolidate csv folder into a single csv with raw spectra, normalized spectra, and peak integrations and show plots
    - script 2: fit Si-H integrations from above dataframe to a A_t = A_0 e^(-kt) + C kinetic model, show plot, and export to csv

### Graph formatting conditions (WiP):
Cure condition
- ambient
  - gray (#FFA500) hollow cirlce 
- laser
  - red (#FF0000) hollow diamond
- oven
  - blue (#1AA7EC) hollow square

Photothermal agent
- no-agent
  - white (#FFFFFF) filled, outlined circle
- AuNP
  - yellow (#FFFF00) hollow triangle
- CB
  - black (#000000) hollow circle

Higher loading and temperatures --> darker (value? sat?), lightness floor at 0.25 (?)

  ### JS implementation considerations
- GUI (input file directory, graphs, wavenumber bounds, checkbox for included plot types and export)
    - directory fields (1: CSV folder, 2: output folder)
    - number field: controls to average over
    - checkboxes: Graphs (1: line, 2: bar), Bands (if bar selection, select all wanted to create)
    - buttons: 1: Run, 2: Export results CSV

### Considerations:
 - Resource: J Hofman's 2016 "IR Spectroscopic Method for Determiation of Silicone Cross-Linking"
 - Modified normalization for CB because of the Si-O-Si dependence on CB content

### Libraries:
os
glob
numpy
pandas
matplotlib.pyplot
simps