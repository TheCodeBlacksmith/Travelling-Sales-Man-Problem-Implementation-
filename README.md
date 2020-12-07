# Travelling-Sales-Man-Problem-Implementation

# Description
This was done as part of group where we designed and implemented the Held-Karp algorithm for the Traveling Sales Man Problem utilizing Python and JSON within the Visual Studio Code editor.

For more details please see the `algo-desc.pdf file`

# Compilation & Running Instructions:

The files can be executed from the terminal as described in the project PDF from the containing folder:
CMD line: tsp-3510.py <input-coordinates.txt> <output-tour.txt> <time>  

positional arguments:
    input_coordinates_file  File to get node coordinates from.

    output_tour_file        Output file for results of TSP algorithim.

    time                    Time for the TSP to execute.

optional arguments:
  -h, --help            show this help message and exit

Ex: tsp_3510.py node_coordinates_simple.txt output-tour.txt 300
Ex: tsp_3510.py node_coordinates.txt output-tour.txt 300

This program was written in Python 3.8.1
# Bugs &/or Limitations:

- NOTE: for the 29 file example provided the best current optimol cost acquired was 28044
which can be reached with timeout of around 6 seconds
