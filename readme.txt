This python script reads all IR Data CSVs in a folder from a given path, normalizes/baseline-corrects them, and plots them as superposed scatterplots and bar graphs for a peak area within each spectrum

For now, run from the spectraHandler script

Resources:
 - https://www.gelest.com/wp-content/uploads/5000A_Section1_InfraredAnalysis.pdf
 - J Hofman's 2016 "IR Spectroscopic Method for Determiation of Silicone Cross-Linking"
 - Joseph Fortenbaugh's 2019 PSU Doctoral Thesis

To-do

vert bars at band cutoffs

[General Graphing]
    - Bar Graph
        - use averages and graph bars with standard error/dev first (how to procedurally take arbitrary number of values to average)
        - in integration bar graph: procedurally color bars for each set to separate without label names
        - have bar labels end on tick instead of middle on tick

[General Readability/Usability/Efficiency]
    - GUI (input file directory, graphs, wavenumber bounds, checkbox for included plot types and export) and web implementation'
        - directory fields (1: CSV folder, 2: output folder)
        - number field: controls to average over
        - checkboxes: Graphs (1: line, 2: bar), Bands (if bar selection, select all wanted to create)
        - buttons: 1: Run, 2: Export results CSV

[Interpretation]
    - what do first two areas represent, how should they change with curing, and why do they go opposite sometimes with laser cure?
    - thinking: homogeneity of samples in ATR?
    - path length - cb change dielectric meaningfully? (what is dielectic constant for each)
    - creating some surface effects that could lead to scatter in measurement?