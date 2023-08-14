# This python script reads all IR Data CSVs in a folder from a given path, normalizes/baseline-corrects them, and plots them as superposed scatterplots and bar graphs for a peak area within each spectrum

For now, run from the spectraHandler script

## Resources:
 - J Hofman's 2016 "IR Spectroscopic Method for Determiation of Silicone Cross-Linking"

## To-do
 - create a universal convention for the input
 - refactor for readability (and separate Si-H output to CSV from its workup?)

### General Graphing
- Spectra (raw)
- Spectra (corrected)
- Peak integrations vs time bar chart
- Si-H integration vs time kinetics-fit scatterplot

### General Readability/Usability/Efficiency
- GUI (input file directory, graphs, wavenumber bounds, checkbox for included plot types and export) and web implementation'
    - directory fields (1: CSV folder, 2: output folder)
    - number field: controls to average over
    - checkboxes: Graphs (1: line, 2: bar), Bands (if bar selection, select all wanted to create)
    - buttons: 1: Run, 2: Export results CSV

## Interpretation
- what do first two areas represent, how should they change with curing, and why do they go opposite sometimes with laser cure?
- thinking: homogeneity of samples in ATR?
- path length - cb change dielectric meaningfully? (what is dielectic constant for each)
- creating some surface effects that could lead to scatter in measurement?

Ben conventions
- gray cirlce = no laser
- red diamond = laser

dark_blue = '#1AA7EC'
grey = '#FFA500'