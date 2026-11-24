# Requirements

## Functional requirements: API and data

| ID | Requirement | Source |
|----|-------------|--------|
| F1 | Accept chunks of raw sensor data from different machines via POST | 1.7.1.3 |
| F2 | Reject or flag invalid input (wrong schema, unknown machine, invalid values) with a clear error | 1.7.1.2 |
| F3 | Forward cleaned data for one machine to C2 via POST, using a format agreed with C2 | 1.7.1.3 |
| F4 | Provide a quality report via GET, per machine and time period | 1.7.1.3 |
| F5 | Provide model information via GET, including the deployed model version | 1.7.1.3, 1.8.2 |
| F6 | Use the dataset's chronological split: first two years for training and validation, last year for testing only | 1.4 |

## Functional requirements: ML

The ML component must cover five capabilities. The modelling approach is ours to choose.

| ID | Requirement | Source |
|----|-------------|--------|
| M1 | Validate incoming sensor data | 1.7.1.2 |
| M2 | Detect and impute missing values | 1.7.1.2 |
| M3 | Detect corrupted measurements (labelled types: sensor spike, sensor frozen, missing value) | 1.7.1.2, 1.3 |
| M4 | Detect unusual operating conditions and sensor drift | 1.7.1.2 |
| M5 | Produce quality reports | 1.7.1.2 |
| M6 | Never use C2 or C3 target columns (failure, RUL, maintenance fields) as model features | 1.3 |

## Non-functional requirements

| ID | Requirement | Source |
|----|-------------|--------|
| N1 | Independently deployable service, Docker recommended | 1.6, 1.8.1 |
| N2 | Automated build, tests and deployment pipeline | 1.8.1 |
| N3 | Explicit model versions, a changelog between versions, and the deployed version visible | 1.8.2 |
| N4 | Fallback behaviour when the model fails (e.g. rule-based checks, warnings, degraded mode) | 1.8.3 |
| N5 | Handle C2 being unavailable without losing data | 1.8.3 |
| N6 | Unit, integration, API, data validation and accuracy tests, run in CI | 1.8.4 |
| N7 | Monitoring of service health, model behaviour, prediction quality and data quality | 1.8.5 |
| N8 | A documented strategy for detecting and handling concept drift | 1.8.5 |
| N9 | An architecture diagram that supports key architectural and technology choices in the project | Sprint document |

## Business objectives