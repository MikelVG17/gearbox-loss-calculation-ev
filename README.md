# gearbox-loss-calculation-ev
Python tool for calculating loss maps of traction gearboxes in electric commercial vehicles, developed as part of a Master’s thesis at RWTH Aachen University.  Topics:  python  gearbox  electric-vehicles  ev  loss-calculation  engineering  automotive  master-thesis


Gearbox Sizing and Losses Estimation Tool
=============================

This repository contains the final version of the Python code developed for a Master’s Thesis in Industrial Engineering. 
The project focuses on the development of a Python-based tool designed to calculate loss maps and other characteristic parameters for traction gearboxes used in electric commercial vehicles. By enabling scalable and precise simulations of gearbox characteristics, this tool will facilitate the efficient design of various drive concepts. Streamlining the simulation and sizing processes will empower manufacturers and engineers to explore and evaluate different gearbox configurations, supporting the development of optimized electric drivetrains that meet the unique requirements of commercial vehicle applications. Ultimately, this work will contribute to accelerating the transition to sustainable transportation within the commercial vehicle sector.


Files:
------
- Gearbox_sizing_losses_Final_Version.py : Python script performing the gearbox sizing and losses calculations.
- Gearbox_Configuration_file.json        : Configuration file with input parameters for the gearbox model.

Overview:
---------
The program calculates gear ratios, gear load stresses, and mechanical losses (including sliding, friction, bearing, and gear mesh resistances). 
It estimates the overall efficiency of the gearbox using simplified but realistic models suitable for preliminary design and comparison.

Context:
--------
This work was completed and successfully graded as a Master’s Thesis at RWTH Aachen University, Chair of Production Engineering of E-Mobility Components (PEM). 
The code was initially developed without version control or collaboration tools. This repository aims to make the project publicly available for further use, review, or improvement.

Requirements:
-------------
- Python 3.8 or higher
- Only standard Python libraries (json, math, etc.) are used.

Usage:
------
1. Adjust input parameters in the Gearbox_Configuration_file.json file as needed.
2. Run the Python script:

   python Gearbox_sizing_losses_Final_Version.py

3. Results will be printed on the console. These outputs include gear sizes, stress levels, and calculated torque and power losses. The tool provides these outputs as both numerical data and visual representations, enabling engineers to assess the feasibility and efficiency of different gearbox configurations.

License:
--------
This project is licensed under the GNU General Public License v3.0 (GPLv3).
You can freely use, modify, and distribute this code, but any derivative works must be licensed under the same terms.

See the LICENSE file for more details.

Acknowledgments and Legal Notice:
---------------------------------
This work was presented at the Chair of Production Engineering of E-Mobility Components (PEM) at RWTH Aachen University.

Master Thesis:
- Name: Mikel Vega Godoy
- Topic: Development of a Python Tool for Calculating Loss Maps of Traction Gearboxes for Electric Commercial Vehicles
- Date: Aachen, September 19, 2024

Author:
-------
Developed by Mikel Vega Godoy as part of the Master’s Thesis in Industrial Engineering (2024).
