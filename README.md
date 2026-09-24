# InnoTrans 2026 Hackathon

This repository contains the public materials, datasets, reference code, and participant documentation for the Alstom InnoTrans 2026 Hackathon challange - **Alstom Inteligence : Talk To My Train**.

The challenge is to build a conversational AI agent that could help an urban operator, such as Berlin U-Bahn, understand, anticipate, and respond to passenger-flow changes caused by events, disruptions, and weather. A strong solution should produce answers that are relevant, explainable, grounded in the available data, and useful in an operational setting.

## Challenge

The agent should help answer questions such as:

- Which stations are likely to become overcrowded after a line suspension?
- How might a major concert or public event change passenger flows?
- Which stations are most at risk during an event combined with adverse weather?
- What actions or rerouting options should an operator consider?

Solutions are evaluated on relevance, reliability, stress testing, innovation, and real-world impact. See the [problem statement](docs/innotrans2026_hackathon_problem_statement.md) for the complete challenge description and evaluation criteria.

## Repository Contents

| Directory or file | Description |
| --- | --- |
| `src/` | Will be used by participants to upload their agents, MCP servers, data-building scripts, and notebooks. |
| `data/` | Training and testing data available for solution development. |
| `docs/` | Public problem statement, schema, and training/testing questions. |

## Dataset

The training dataset describes a partial Berlin U-Bahn network and covers different urban data from **June 10, 2026 through September 21, 2026**. A separate testing dataset covers the evaluation period from **September 22 through October 1, 2026** and must not be used to develop or tune the solution.

### Main data sources

- **Network topology:** stations, served lines, station coordinates, and bidirectional connections.
- **Passenger flows:** simulated station-level counts at 15-minute intervals.
- **Events:** Berlin concerts, matches, conferences, public gatherings, and other events with time, location, category, and estimated attendance.
- **Closures:** simulated disruption scenarios, including affected times and descriptions.
- **Weather:** temperature, humidity, precipitation, wind, pressure, cloud cover, and weather-condition codes.
- **Energy:** simulated daily energy consumption by U-Bahn line.

Read the complete [dataset schema](docs/dataset_schema.md) before using the data. Important relationships include:

- `stations_with_ubahn.station_id` is the station key.
- Flow station columns match `stations_with_ubahn.station_name`.
- `flows.timestamp` joins to the weather timestamp at the 15-minute grain.
- Events do not contain station IDs and may need to be associated with stations through their venue or address.

The [testing questions](docs/hackathon_questions_testing.md) describe the final evaluation scenarios. Testing data is in `data/testing dataset/`; it contains rest-period flows, closures, weather, energy, and events. Unlike the training event file, `berlin_events_summer_2026_rest.csv` does not include a header row.

## Getting Started

### 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Create a Python environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas networkx numpy openai fastmcp mcp openpyxl
```


## Building a Solution

Recommended capabilities include:

1. Load and validate the network, flow, event, closure, weather, and energy data.
2. Normalize timestamps and document all assumptions about local time and time windows.
3. Map events and disruptions to affected stations and lines.
4. Combine network topology with flow history to identify likely rerouting and pressure points.
5. Ground every answer in calculations or retrieved records and communicate uncertainty clearly.
6. Keep the interface usable for an operator working under time pressure.

The [training questions](docs/hackathon_questions_training.md) provide representative workflows for developing and testing a solution.

## Data and Evaluation Rules

- Each team will be evaluated using the same set of questions, using the [evaluation criteria](docs/ALSTOM_Challenge_Evaluation_Onepager.pdf)
- Treat the data as simulated or partial where the schema says so; do not infer unsupported passenger-level origin-destination journeys.

## Documentation

- [Problem statement](docs/innotrans2026_hackathon_problem_statement.md)
- [Dataset schema](docs/dataset_schema.md)
- [Training questions](docs/hackathon_questions_training.md)
- [Testing questions](docs/hackathon_questions_testing.md)

## License and Usage

This repository is provided for the InnoTrans 2026 Hackathon. Follow the event organizers' instructions for permitted use, redistribution, data handling, and publication of solutions.
