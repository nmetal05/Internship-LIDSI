# Traffic Simulation and Digital Twin Framework

## Overview

This project presents a comprehensive traffic simulation and digital twin framework developed as part of an internship at the **Laboratoire d'Ingénierie des Données et Systèmes Intelligents (LIDSI)**, Faculté des Sciences Ain-Chock – Université Hassan II de Casablanca. The framework integrates SUMO traffic simulation, Simulation-Based Inference (SBI) for parameter calibration, real-time data processing, and visualization through a FIWARE-based dashboard.

## Authors

This project was collaboratively developed by:
- **Noureddine Dahou** - [Dahousan](https://github.com/Dahousan)
- **Lazrak Chaymae** - [Lazrak-chaymae](https://github.com/Lazrak-chaymae)
- **Nizar Sahl** - [nmetal05](https://github.com/nmetal05)

## Project Structure

The repository is organized into several key components:

### Core Components

#### 1. Simulations
- **Episode1-Episode6**: Complete SUMO simulation scenarios with varying traffic conditions
- **Features**: OSM-based network generation, activity-based demand modeling, multi-modal transport
- **Outputs**: Vehicle trajectories, traffic statistics, performance metrics

#### 2. Simulation-Based Inference (SBI)
- **Location**: `Sbi/app.py`
- **Purpose**: Parameter calibration and uncertainty quantification
- **Features**: 
  - Streamlit-based interactive interface
  - Parallel simulation execution
  - Simulation-Based Inference using PyTorch
  - Parameter space exploration for traffic light timing, demand scaling, and vehicle behavior

#### 3. Dashboard Integration
- **Location**: `Dashboard/`
- **Components**:
  - FIWARE Context Broker integration (`docker-compose.yml`)
  - Real-time SUMO data streaming (`sumo_to_fiware.py`)
  - Pre-configured Grafana dashboard (`Grafana_Dashboard_Model.json`)
  - Orion Context Broker, QuantumLeap, CrateDB, and Grafana stack

#### 4. Data Processing Pipeline
- **Real Data Calibration**: `PrepareRealdataForcalibration/`
  - TomTom traffic data processing
  - Speed and waiting time analysis
  - Ground truth preparation for model validation

- **Simulation Data Generation**: `PrepareSimulationDatasetsFormodel/`
  - Comprehensive feature extraction from SUMO outputs
  - Network topology analysis
  - Multi-dimensional dataset preparation

#### 5. Inference Analysis
- **Location**: `Inference/`
- **Features**:
  - SBI parameter estimation
  - Posterior distribution analysis
  - Model validation and comparison plots
  - Real vs simulated data comparison

## Key Features

### Traffic Simulation
- **Multi-Episode Scenarios**: Six different traffic episodes representing various urban conditions
- **OSM Integration**: Real-world road network from OpenStreetMap
- **Activity-Based Modeling**: Realistic trip generation based on urban activity patterns
- **Multi-Modal Support**: Cars, buses, pedestrians, and taxi services

### Parameter Calibration
- **SBI Approach**: Uses Simulation-Based Inference (SBI) for parameter estimation
- **Key Parameters**:
  - Demand scaling factors
  - Traffic light duration scaling
  - Vehicle speed factors
- **Parallel Processing**: Multi-core simulation execution for efficiency

### Real-Time Integration
- **FIWARE Stack**: Complete IoT platform for data management
- **Live Data Streaming**: Real-time vehicle positions, speeds, and emissions
- **Dashboard Visualization**: Comprehensive Grafana dashboard with multiple analytics panels
  - Traffic congestion analysis (top congested streets)
  - Real-time vehicle count and evolution
  - Waiting time analysis
  - Geographic visualization with pollution hotspots
  - Speed distribution and analysis
  - CO2 emissions monitoring
  - Interactive maps of Hay Hassani area

### Data Analysis
- **Comparative Analysis**: Real traffic data vs simulation outputs
- **Statistical Validation**: Posterior distribution analysis
- **Performance Metrics**: Speed, waiting times, emissions, and throughput

## Installation and Setup

### Prerequisites
- Python 3.8+
- SUMO Traffic Simulator
- Docker and Docker Compose
- Required Python packages (see requirements in respective directories)

### SUMO Installation
1. Download and install SUMO from [https://sumo.dlr.de/](https://sumo.dlr.de/)
2. Set the `SUMO_HOME` environment variable
3. Add SUMO binary directory to your PATH

### Python Dependencies
```bash
# Install core dependencies
pip install numpy pandas torch streamlit
pip install sbi traci sumolib
pip install requests xml matplotlib
```

### Dashboard Setup
```bash
# Navigate to dashboard directory
cd Dashboard/

# Start FIWARE services
docker-compose up -d

# Verify services are running
docker-compose ps
```

## Usage

### Running Traffic Simulations

#### Individual Episode Simulation
```bash
cd Simulations/Episode1/
sumo-gui -c osm.sumocfg
```

#### Batch Processing
```bash
cd PrepareSimulationDatasetsFormodel/
python ModelDataGeneration.py
```

### Simulation-Based Inference Training
```bash
cd Sbi/
streamlit run app.py
```

1. Upload a SUMO scenario ZIP file
2. Configure parameter ranges
3. Run parallel simulations
4. Train the SBI model
5. Analyze posterior distributions

### Real Data Processing
```bash
cd PrepareRealdataForcalibration/
python RealDataForCalibration.py
```

### Dashboard Integration
```bash
cd Dashboard/
# Start FIWARE services
docker-compose up -d

# Run SUMO data streaming
python sumo_to_fiware.py
```

#### Import Pre-configured Dashboard
1. Access Grafana at http://localhost:3000 (admin/admin)
2. Go to Dashboard > Import
3. Upload `Grafana_Dashboard_Model.json`
4. Configure CrateDB data source (PostgreSQL protocol, localhost:5432)

Access services at:
- Grafana Dashboard: http://localhost:3000
- CrateDB Admin: http://localhost:4200
- Orion Context Broker: http://localhost:1026

## Technical Details

### Simulation Framework
- **SUMO Version**: Compatible with latest SUMO releases
- **Network Generation**: OSM-based with custom processing
- **Demand Modeling**: Activity-based trip generation
- **Calibration**: TomTom real traffic data integration

### Simulation-Based Inference
- **Method**: Neural Posterior Estimation (NPE)
- **Framework**: PyTorch-based SBI library
- **Parameters**: 3D parameter space (demand, traffic lights, speed)
- **Validation**: Cross-validation with held-out real data

### Data Pipeline
- **Input Processing**: XML parsing for SUMO configurations
- **Feature Extraction**: Network topology, demand patterns, control parameters
- **Output Analysis**: Trip statistics, performance metrics, emissions
- **Normalization**: Min-max scaling for neural network compatibility

### FIWARE Integration
- **Context Broker**: Orion for entity management
- **Time Series**: QuantumLeap for historical data
- **Storage**: CrateDB for scalable analytics
- **Visualization**: Grafana for real-time monitoring
  - Pre-configured dashboard model with SQL queries
  - Multiple visualizations (bar charts, maps, histograms, time series)
  - Georeferenced traffic monitoring
  - Pollution and congestion analytics

## Results and Validation

### Model Performance
- **Parameter Estimation**: SBI posterior distributions for traffic parameters
- **Validation Metrics**: Comparison against real TomTom traffic data
- **Uncertainty Quantification**: Confidence intervals for predictions

### Simulation Outputs
- **Traffic Metrics**: Speed, waiting times, throughput
- **Environmental Impact**: CO2 emissions, fuel consumption
- **Network Performance**: Congestion patterns, bottleneck identification

## File Structure Details

```
Stage-LIDSI/
├── Dashboard/
│   ├── docker-compose.yml          # FIWARE services configuration
│   ├── Grafana_Dashboard_Model.json # Pre-configured dashboard with analytics panels
│   └── sumo_to_fiware.py          # Real-time data streaming
├── Simulations/
│   ├── Episode1/                   # Traffic scenario 1
│   ├── Episode2/                   # Traffic scenario 2
│   ├── ...                        # Additional episodes
│   └── Episode6 (not finished)/   # Work-in-progress scenario
├── Sbi/
│   └── app.py                     # Simulation-Based Inference application
├── Inference/
│   ├── Inference_Testing.ipynb    # Analysis notebook
│   ├── comparison_plot_*.png       # Validation plots
│   └── posterior*.pt              # Trained models
├── PrepareRealdataForcalibration/
│   ├── RealDataForCalibration.py  # TomTom data processing
│   └── real_data.csv              # Processed real traffic data
└── PrepareSimulationDatasetsFormodel/
    ├── ModelDataGeneration.py     # Feature extraction
    └── simulation_ep*_dataset.npz # Generated datasets
```

## Contributing

This project was developed as part of academic research. For contributions or questions, please contact the authors through their respective GitHub profiles.

## License

This project is developed for academic and research purposes at Université Hassan II de Casablanca. Please refer to the institution's policies regarding code usage and distribution.

## Acknowledgments

We would like to express our sincere gratitude to our supervising professors at the Faculté des Sciences Ain-Chock – Université Hassan II de Casablanca:
- **Prof. Jai Said Andaloussi**
- **Prof. El Ouassit Youssef**
- **Prof. El kasmi alaoui seddiq**

Their guidance, expertise, and support were invaluable throughout this internship project.

We also acknowledge:
- **Laboratoire d'Ingénierie des Données et Systèmes Intelligents (LIDSI)**
- **Faculté des Sciences Ain-Chock – Université Hassan II de Casablanca**
- **SUMO Development Team** for the traffic simulation platform
- **FIWARE Foundation** for the IoT platform components
- **PyTorch SBI Community** for the Simulation-Based Inference framework

## Contact

For technical questions or collaboration opportunities, please reach out through the GitHub repository or contact LIDSI at Université Hassan II de Casablanca.