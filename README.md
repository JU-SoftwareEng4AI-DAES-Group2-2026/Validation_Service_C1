# System Architecture

# Requirements

## C1: Sensor Quality and Data Validation Service

### Business Objective Operational

Decisions should not be based on
unreliable sensor measurements. The purpose of this service is to identify and
manage sensor quality issues before data is used by downstream services.

#### Required Machine Learning Functionality 
This service must
include a machine learning component that evaluates incoming sensor data and
identifies potential quality problems.
The required capabilities are:
- Validate incoming sensor data
- Detect and impute missing values
- Detect corrupted measurements
- Detect unusual operating conditions and sensor drift
- Produce quality reports
The exact modeling approach is left to the team.

#### Required API Capabilities
The service should provide functionality for:

Inputs: 
- POST chunks of data from different machines�
Outputs: 
- POST to C2 to add cleaned data from one machine for fault prediction - GET quality report - GET model information
