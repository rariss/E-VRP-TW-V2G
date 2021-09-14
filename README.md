[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


# Electric Vehicle Routing Problem with Time Windows and Vehicle-to-Grid (E-VRP-TW-V2G)

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

We present a novel extension of the Electric Vehicle Routing Problem (E-VRP) that co-optimizes fleet
sizing, vehicle routing, and vehicle-to-grid (V2G) charging and discharging decisions for 
time-varying energy arbitrage and peak shaving to maximize net amortized profit.

[![Product Name Screen Shot][product-screenshot]](https://github.com/rariss/E-VRP-TW-V2G)

<!-- GETTING STARTED -->
## Getting Started

This package includes various formulations of the E-VRP-TW-V2G, with a base model utilizing Big-M
formulations, and three additional model variations using Pyomo's Generalized Disjunctive Programming.

### Prerequisites

This Mixed Integer Linear Program (MILP) was developed using the Pyomo mathematical modeling package
and the commercial-grade solver Gurobi.

* [Install and setup Gurobi](https://www.gurobi.com/documentation/9.1/quickstart_linux/cs_python_installation_opt.html): 
```sh
python -m pip install gurobipy
```
* [Generate a Google Maps API Key](https://developers.google.com/maps/documentation/javascript/get-api-key)
  * Update `GOOGLE_API_KEY` in `LOCAL_CONFIG.py` once generated. 

### Installation

1. Clone the repo:
   ```sh
   git clone https://github.com/rariss/E-VRP-TW-V2G.git
   ```
2. Install requirements:
   ```sh
   pip install -r requirements.txt
   ```
3. Install the evrptwv2g package using `setup.py`: 
   ```sh
   pip install .
   ```

<!-- USAGE EXAMPLES -->
## Usage

### Running the E-VRP-TW-V2G model
Running `main.py` will run the base E-VRP-TW-V2G model, outputting the optimal solution and generating a figure of the results.

1. Update the instance input directory `DIR_INSTANCES` and the output directory `DIR_OUTPUT` in `LOCAL_CONFIG.py`
2. Run `main.py` from terminal using `python3 main.py '<instance>' '<objective and constraint options>' '<distance matrix type>'` where:
   * Argument 1: define the instance in `DIR_INSTANCES` to run
   * Argument 2: define the model problem type and constraints
     * Objective options include: `Schneider` OR any combination of `OpEx`, `CapEx`, `Cycle`, `EA`, `DCM`, `Delivery`
     * Constraint options include combinations of the following: `Start=End`, `FullStart=End`, `NoXkappaBounds`, `NoMinVehicles`, 
     `MaxVehicles`, `NoSymmetry`, `NoXd`, `SplitXp`, `StationaryEVs`, `NoExport`
   * Argument 3 (optional): define the distance matrix type from `scipy`, `<distance matrix CSV filepath>`, `googlemaps`
```sh
python3 main.py 'v2g_m06_s7_ca_tesla_dis_ea_t1_eff' 'splitXp start=end ea capex OpEx dcm NoExport stationaryevs 100xloadprofile'
```
3. Results will be printed to the terminal console and a plot will be generated in `DIR_OUTPUT`

### Creating new test instance CSVs
1. Pass a Schneider text instance name to `convert_txt_instances_to_csv`
2. The Schneider text instance will be converted to a comparable CSV for the E-VRP-TW-V2G model,
saved in the same `DIR_INSTANCES` folder

### Generating a distance matrix CSV using the Google Maps API
1. Instantiate a new EVRPTWV2G object with `dist_type=googlemaps` and `save_dist_matrix=True`. For example:
```sh
m = EVRPTWV2G(problem_type='<any valid problem_type>', dist_type='googlemaps', save_dist_matrix=True)
```

2. Import the instance by passing in an instance filepath, which will generate the distance matrix CSV:
```sh
m.import_instance('<instance_filepath>')
```

3. The Google Maps generated distance matrix will be saved as a timestamped CSV in `DIR_OUTPUT`

To run the model off the generated CSV, pass the CSV filepath as the third argument when running the model using the terminal.

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.


<!-- CONTACT -->
## Contact

Rami Ariss - [LinkedIn](https://www.linkedin.com/in/ramiariss/) - [CMU Directory](https://www.cmu.edu/cee/people/cee-phd-students.html)

[E-VRP-TW-V2G Repository](https://github.com/rariss/E-VRP-TW-V2G)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* **Rami Ariss**, Civil and Environmental Engineering PhD, Carnegie Mellon University
* **Shang Zhu**, Mechanical Engineering PhD, Carnegie Mellon University
* **Leandre Berwe**, Electrical and Computer Engineering MS, Carnegie Mellon University
* **Professor Jeremy Michalek**, Mechanical Engineering, Engineering and Public Policy, Carnegie Mellon University 
* **Professor Ignacio Grossmann**, Chemical Engineering, Carnegie Mellon University for his support on the GDP formulations and Pyomo implementations

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[forks-shield]: https://img.shields.io/github/forks/rariss/E-VRP-TW-V2G
[forks-url]: https://github.com/rariss/E-VRP-TW-V2G/network/members
[stars-shield]: https://img.shields.io/github/stars/rariss/E-VRP-TW-V2G
[stars-url]: https://github.com/rariss/E-VRP-TW-V2G/stargazers
[issues-shield]: https://img.shields.io/github/issues/rariss/E-VRP-TW-V2G
[issues-url]: https://github.com/rariss/E-VRP-TW-V2G/issues
[license-shield]: https://img.shields.io/github/license/rariss/E-VRP-TW-V2G
[license-url]: https://github.com/rariss/E-VRP-TW-V2G/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-0077B5?style=plastic&logo=linkedin&logoColor=white
[linkedin-url]: https://www.linkedin.com/in/ramiariss/
[product-screenshot]: images/E-VRP%20-%20Graph%20Routes%20v1.png