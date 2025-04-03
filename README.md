# Project_Break_II_ML
Final individual project of the Data Science &amp;amp; AI Bootcamp offered by TheBridge

## Table of Contents
- [Introduction](#introduction)
- [Objectives](#objectives)
- [Data](#data)
- [Data Description](#data-description)
- [Usage](#usage)
- [Contribution](#contribution)
- [License](#license)
- [Project Structure](#project-structure)


## Introduction

Changes in schedules and delays in commercial flights are problems that affect everyone involved in aviation. From the passengers' perspective, it results in dissatisfied customers due to potential disruptions to their plans and the stress caused. The crew and other airport employees are also impacted, diverting manpower to resolve issues that could be minimized with better planning. The airlines, in the end, are among the most affected, facing dissatisfaction from both customers and employees, as well as significant financial losses due to airport fees, fines, passenger/crew rebooking, etc.

Thus, it is clear that the sooner a possible delay is predicted, the lower the damage will be for all involved. However, this is not easy, as flight delays are multi-variable and depend on unpredictable, stochastic, and uncontrollable factors, such as extreme weather events, technical problems with aircraft, security-related issues, etc. Nevertheless, although it is not possible to predict all the sources of delays, it is possible to identify some patterns that fall under the logistical control of airports and airlines. Therefore, the following work aims to use logistical data, geolocation, and weather forecasting data to identify patterns and create a classifier that can predict whether a flight will be delayed or not (binary classifier), using only information that can be obtained well in advance, enabling decision-making by airport and airline organizations.

## Objectives
**General:**
- Prediction of whether a flight will be delayed at takeoff;
- 
**Specific:**
- A binary classifier to classify delayed flights (class 1) and on-time flights (class 0) within a 15-minute tolerance beyond the scheduled time. For this objective, only logistical data available since the flight booking and weather forecasts will be used, without real-time updates immediately before takeoff. This makes the classifier very challenging but provides a highly important predictive potential, as it helps identify delay patterns that are independent of unpredictable, stochastic, and uncontrollable events;

# Data
The dataset used was obtained from the following Kaggle page: https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft/data


## Data Description

| Field               | Description                                               | Example                               |
|---------------------|-----------------------------------------------------------|---------------------------------------|
| **FlightDate**      | The date of the flight.                                    | 2024-09-09                           |
| **Day_Of_Week**     | The day of the week when the flight occurred.              | Monday                               |
| **Airline**         | The name or code of the airline operating the flight.      | AA for American Airlines             |
| **Tail_Number**     | The unique identifier (registration number) of the aircraft. | N123AA                           |
| **Dep_Airport**     | The IATA code of the departure airport.                    | JFK                                  |
| **Dep_CityName**    | The name of the city where the departure airport is located. | New York                             |
| **DepTime_label**   | The categorized or labeled departure time.                 | Morning                              |
| **Dep_Delay**       | The number of minutes the flight was delayed at departure.  | 1                                   |
| **Dep_Delay_Tag**   | Categorical representation of the departure delay.          | On-time                              |
| **Dep_Delay_Type**  | The type or reason for the departure delay.                 | Operational                          |
| **Arr_Airport**     | The IATA code of the arrival airport.                       | LAX                                  |
| **Arr_CityName**    | The name of the city where the arrival airport is located.  | Los Angeles                          |
| **Arr_Delay**       | The number of minutes the flight was delayed at arrival.     | 10                                   |
| **Arr_Delay_Type**  | The type or reason for the arrival delay.                    | Weather                              |
| **Flight_Duration** | The duration of the flight.                                  | 300 minutes                          |
| **Distance_type**   | The classification of flight distance.                       | Long-haul                            |
| **Delay_Carrier**   | Delay in minutes attributed to the airline/carrier.          | 5                          |
| **Delay_Weather**   | Delay in minutes attributed to weather conditions.           | 10                         |
| **Delay_NAS**       | Delay in minutes attributed to National Airspace System issues. | 0 |
| **Delay_Security**  | Delay in minutes attributed to security-related issues.       | 0 |
| **Delay_LastAircraft** | Delay in minutes due to the previous flight.               | 20 |
| **Manufacturer**    | The aircraft manufacturer.                                   | Boeing                               |
| **Model**          | The aircraft model.                                          | Boeing 737                           |
| **Aircraft_age**   | The age of the aircraft in years.                            | 12 years                             |
| **time**           | The specific time of the flight record.                      | 12:30 PM                         |
| **tavg**           | The average temperature on the day of the flight.            | 22°C                                 |
| **tmin**           | The minimum temperature on the day of the flight.            | 18°C                                 |
| **tmax**           | The maximum temperature on the day of the flight.            | 28°C                                 |
| **prcp**           | Precipitation amount on the day of the flight.               | 2 mm                                 |
| **snow**           | Snowfall amount on the day of the flight.                    | 0 mm                                 |
| **wdir**           | Wind direction.                                              | 270°                                 |
| **wspd**           | Wind speed.                                                  | 15 km/h                              |
| **pres**           | Atmospheric pressure.                                        | 1015 hPa                             |
| **airport_id**     | A unique identifier for the airport.                         | 12345                                |
| **IATA_CODE**      | The International Air Transport Association code.            | JFK                            |
| **AIRPORT**        | The name of the airport.                                     | John F. Kennedy International Airport |
| **CITY**           | The city where the airport is located.                       | New York                             |
| **STATE**          | The state or region where the airport is located.            | NY                                   |
| **COUNTRY**        | The country where the airport is located.                    | USA                                  |
| **LATITUDE**       | The latitude coordinate of the airport.                      | 40.6413° N                           |
| **LONGITUDE**      | The longitude coordinate of the airport.                     | 73.7781° W                           |

## Usage
This project is intended for educational purposes. You are free to use, modify, and distribute the content as needed for learning and academic activities. Please feel free to explore, experiment, and adapt the code to suit your needs. Please note that this project is a work-in-progress for educational purposes. It may not be perfect or fully optimized, as it was developed to demonstrate specific concepts and practices learned during the course.

## Contribution

Contributions are welcome! If you would like to contribute to this project, feel free to submit a **pull request**, open an **issue**, or engage in discussions. All contributions, whether through code improvements, bug reports, or constructive feedback, are appreciated.

#### How to Contribute

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them.
4. Push your changes to your fork.
5. Open a pull request to the main repository.

Please ensure your pull request is well-documented and clearly explains the changes you made. We encourage **constructive criticism** and **suggestions** to help improve the project.

Thank you for contributing!

## License
License
The dataset used in this project is sourced from Kaggle and is licensed under **CC0: Public Domain**, meaning it is free to use for any purpose, including commercial use, without any copyright restrictions. Please refer to the [dataset's Kaggle page](https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft/data) for further details and attribution requirements.

## Project Structure

src/
│── .gitignore.txt
│── requirements.txt
│
├── data_sample/
│ └── data_sample_flights.csv
│
├── models/
│ ├── best_model_lgbm.joblib
│ ├── best_model_lgbm.pkl
│ ├── best_model_lgbm.txt
│ ├── best_model_xgb.joblib
│ └── best_model_xgb.pkl
│
├── notebooks/
│ ├── main - con dep delay last aircraft.ipynb
│ ├── main - Final - RegressionArrival.ipynb
│ ├── main - Final LastAircraft.ipynb
│ ├── main - Nature - transformaciones.ipynb
│ ├── main - Nature.ipynb
│ ├── main - Regression_Arrival.ipynb
│ ├── main.ipynb
│ └── Notebook_without_undersampling_2022.ipynb
│
├── results_notebook/
│ ├── main.ipynb
│ ├── requirements.txt
│ └── img/
│ ├── openart-image__3.jpg
│ ├── openart_image_1_1.jpg
│ └── openart_image_2.jpg
│
└── utils/
└── User_Functions.py
