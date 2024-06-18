`Independent variable = an input value`
`Dependent variable = an output value (model prediction)`

Regression models are used to predict a single numerical value. For example, imagine we had this dataset:

| Weather condition | Distance (km) | Amount of traffic | Time taken (minutes) |
| ----------------- | ------------- | ----------------- | -------------------- |
| Clear             | 4.8           | Minimal           | 4.47                 |
| Storm             | 11.7          | Heavy             | 21.82                |
| Light rain        | 2.1           | Moderate          | 2.62                 |
| ...               | ...           | ...               | ...                  |
[First 3 columns are input parameters, the fourth column is the single output parameter]

The above table shows the time taken to get between two points of different distances, with data on the weather condition, the distance, and amount of traffic. In a real world scenario, the dataset would need to be much larger if we wanted to train an accurate model. The reason this falls under 'supervised learning' is because the developer provides a source of truth (the dataset) for the model to guide its training. 

The first three features are the input parameters. The fourth parameter (time taken) is what we want the regression model to predict. 

The process where models take one or more independent variables (input values) and predict one dependent variable (output value) is called multiple regression.
## Predicting multiple output values

To predict more than one output value, we need to use a technique known as multivariate regression. This is different from multiple regression in that it allows us to predict more than one numerical value.

If we use the same example as above

| Weather condition | Distance (km) | Amount of traffic | Time taken (minutes) |     |
| ----------------- | ------------- | ----------------- | -------------------- | --- |
| Clear             | 4.8           |                   |                      |     |
| Storm             | 11.7          |                   |                      |     |
| Light rain        | 2.1           |                   |                      |     |
| ...               | ...           | ...               | ...                  | ... |
<< ^^ UNFINISHED ^^ >>