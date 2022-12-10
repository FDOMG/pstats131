---
  title: "131 Final Project"
author: "Ruoyu Liang"
date: '2022-12-08'
output: html_document
---
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, message = FALSE,
                      warning = FALSE)
```

# Introduction

The purpose of this project is to generate models which analyze the sale value of different car brands.

In this project, we used four factors to see if they are effective to the sale price at the end.

Horsepower (hp) is a unit of measurement of power, or the rate at which work is done, usually in reference to the output of engines or motors. There are many different standards and types of horsepower. Two common definitions used today are the mechanical horsepower (or imperial horsepower), which is about 745.7 watts, and the metric horsepower, which is approximately 735.5 watts.(From Wikipedia)

#Loading data and packages needed
Key variables:
  
  Brands: car brands, character variable, nothing to do with numerical 
variables.

sales_value: sale values for each car brands and models

model:car models for each brands(I decide to exclude this since different models have different factors, which are the main things we are going to analyze)

vehicle_type: includes two different types: passengers and car.

Fuel_efficiency: the efficiency of fuel for the cars.

Horsepower:horsepower for each type or brand of cars.

length:length of the car,which represents how comfort you may sit as a passenger.

width:width of the car,which represents how comfort you may sit as a passenger.

```{r}
library(ranger)
library(tidyverse)
library(tidymodels)
library(janitor)
library(ggplot2)
library(dbplyr)
library(corrr)
library(magrittr)
library(ISLR)
library(ISLR2)
library(klaR)
library(rpart.plot)
library(glmnet)
library(randomForest)
library(xgboost)
library(class)
library(kernlab)
library(stringr)
library(purrr)
library(corrplot)
tidymodels_prefer()
```
```{r}
cars<-read_csv("Car_sales.csv")%>% 
  mutate(brands = factor(Manufacturer))%>% 
  mutate(Model = factor(Model))%>% 
  mutate(length = factor(Length))%>% 
  mutate(vehicle_type = factor(Vehicle_type)) %>% 
  mutate(sales_value = factor(Price_in_thousands)) %>%
  mutate(Fuel_efficiency = factor(Fuel_efficiency)) %>% 
  mutate(sales_n = factor(Sales_in_thousands))

```

#Data Clean step
remove unneeded variables that we are not going to concern in this project.
```{r}
cars<-cars %>% 
  select(-year_resale_value,-Engine_size,
         -Wheelbase,-Width,-Curb_weight,-Fuel_capacity,-Latest_Launch,
         -Power_perf_factor)
```
Reomve the listing that contains missing values
```{r}
count(cars)
sum(is.na(cars))
row_status<-complete.cases(cars)
cars<-cars[row_status,]
sum(is.na(cars))
```
# Data split
```{r}
cars_split<-cars %>% 
  initial_split(prop=0.8,strata=brands)
cars_train<-training(cars_split)
cars_test<-testing(cars_split)
```
traning data set has 123 obs and testing data set has 31 obs.


# Exploratory Data Analysis
I am gonna use trainning data set for analysis,which has 123 observations.

```{r}
ggplot(data=cars_train,aes(x=Vehicle_type))+geom_bar(fill="red")+
  labs(title = "Composition of Car Types",
       x="car Type",
       y="cars for sales")+
  theme_minimal()+
  coord_flip()
```
type:car means huge cars such as cadillac Escalade in original data sets. 
Type:passengers represents cars like SUV and normal cars lke audi A4.

We can see most cars for sales are made for passenagers type, which is not suprised since most people will not buy a huge car. People normally buy cars that designed for taking passengers(families, friends).

```{r}
ggplot(cars_train, aes(x = brands, y = Vehicle_type)) +
  labs(title = "cars types produced by differnt brands",
       x="brands",
       y="car Type")+
  geom_point()
```
We can see the result is similiar that different car brands mainly sells passengers type of cars. dots in graph are mostly on passengers line.  There are much less points on car type.



```{r}
ggplot(cars_train, aes(reorder(brands, sales_value), sales_value)) +
  geom_boxplot(varwidth = TRUE)+
  scale_y_discrete(0,200)+
  coord_flip()+
  labs(
    title = "Price by car brands",
    x = "car  Type",
    y="Price")
```
We can see there is no boxplot showing. I think this is probably becuase each models for each brands has such a different prices range. It kind of tells us that brands does not affect the price that much. I should consider other variables to predict price range.But we do see some highest price belongs to Mercedes and Audi since they are counted as luxury cars.

```{r}
ggplot(cars_train, aes(x = sales_n, y = brands)) +
  labs(title = "Number of cars sold by brands",
       x="Number of cars sold",
       y="brand")+
  geom_point()
```
Honda is the most popular car sold and volvo and acura are least popular ones.

```{r}
cars_train %>% 
  select(is.numeric,-brands,-vehicle_type,-sales_n,-Model) %>% 
  cor() %>% 
  corrplot(type = "lower")
```
For his data set, we actually see a connection between horsepower and price. Surprisingly, length does not have a strong affection on price since i thought the longer the car is, the more space it gets,which will affect the price. anyway, length andn horsepower has a light relationship. We will analyze the model between horsepower and price.Price does not affect sale number as well.


# make a linear regression model

```{r}
set.seed(1202)

lm_spec <- linear_reg() %>% 
  set_mode("regression") %>% 
  set_engine("lm")

lm_fit<-lm_spec %>% 
  fit(Price_in_thousands~Horsepower,
      data=cars_train)

augment(lm_fit,new_data=cars_train) %>% 
  rmse(truth=Price_in_thousands,estimate=.pred)

augment(lm_fit,new_data=cars_test) %>% 
  rmse(truth=Price_in_thousands,estimate=.pred)

```
```{r}
cars_train_res<-predict(lm_fit,new_data = cars_train %>% 
                          select(-Price_in_thousands))

cars_train_res<-bind_cols(cars_train_res,cars_train %>% 
                            select(Price_in_thousands))

cars_train_res %>% 
  ggplot(aes(x= .pred,y=Price_in_thousands))+
  geom_point(alpha=0.2)+
  geom_abline(lty=2)+
  theme_bw()+
  coord_obs_pred()
```
The model fits pretty good and we can use this model to predirct price with horsepower. and it is showing a postive relationship which means if the horsepower becomes stronger, then the price is higher.

#polynomial regression model
```{r}
cars_recipe <-recipe(Price_in_thousands~Horsepower
                     ,
                     data = cars_train) %>% 
  step_poly(Horsepower
            ,degree=2)

poly_wf<-workflow() %>% 
  add_recipe(cars_recipe) %>% 
  add_model(lm_spec)

poly_fit<-fit(poly_wf,data=cars_train)

augment(poly_fit,new_data = cars_train) %>% 
  rmse(truth=Price_in_thousands,estimate=.pred)

augment(poly_fit,new_data = cars_test) %>% 
  rmse(truth=Price_in_thousands,estimate=.pred)
```

```{r}
cars_train_res<-predict(poly_fit,new_data = cars_train %>% 
                          select(-Price_in_thousands))

cars_train_res<-bind_cols(cars_train_res,cars_train %>% 
                            select(Price_in_thousands))

cars_train_res %>% 
  ggplot(aes(x= .pred,y=Price_in_thousands))+
  geom_point(alpha=0.2)+
  geom_abline(lty=2)+
  theme_bw()+
  coord_obs_pred()
```
similar outcome and shows a strong positive relationship between price and horsepower. A great model.

```{r}
poly_cars_rec<-recipe(Price_in_thousands ~ Horsepower, data = cars_train) %>%
  step_poly(Horsepower, degree = tune())

poly_cars_wf <- workflow() %>%
  add_recipe(poly_cars_rec) %>%
  add_model(lm_spec)


cars_folds <- vfold_cv(cars_train,v=10)

degree_grid <- grid_regular(degree(range = c(1, 10)), levels = 10)

cars_res <- tune_grid(
  object = poly_cars_wf, 
  resamples = cars_folds, 
  grid = degree_grid
)

autoplot(cars_res)
```
Let s see whtich is the best model out of these
```{r}
show_best(cars_res,metric="rmse")
```
We are gonna pick the first one.
```{r}
select_by_one_std_err(cars_res, degree, metric = "rmse")
best_degree <- select_by_one_std_err(cars_res, degree, metric = "rmse")
final_wf <- finalize_workflow(poly_wf, best_degree)

```
We make sure our training data and test data always fit the best data and then we test its accuracy.
```{r}
carsfinal_fit <- fit(final_wf, cars_train)
augment(carsfinal_fit, new_data = cars_train) %>%
  rmse(truth =Price_in_thousands, estimate = .pred)
augment(carsfinal_fit, new_data = cars_test) %>%
  rmse(truth = Price_in_thousands, estimate = .pred)
```
# use ridge regression model to visualize
```{r}
ridge_spec <- linear_reg(mixture = 0, penalty = 0) %>%
  set_mode("regression") %>%
  set_engine("glmnet")

ridge_fit <- fit(ridge_spec, Price_in_thousands ~ .,data = cars_train)

tidy(ridge_fit)

ridge_fit %>%
  extract_fit_engine() %>%
  plot(xvar = "lambda")
```

Create a recipe

```{r}
ridge_recipe <- 
  recipe(formula =Price_in_thousands ~ ., data = cars_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

ridge_spec <- 
  linear_reg(penalty = tune(), mixture = 0) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet")

ridge_workflow <- workflow() %>% 
  add_recipe(ridge_recipe) %>% 
  add_model(ridge_spec)

penalty_grid <- grid_regular(penalty(range = c(-5, 5)), levels = 50)

tune_res <- tune_grid(
  ridge_workflow,
  resamples = cars_folds, 
  grid = penalty_grid
)
autoplot(tune_res)
```
We see amount of Regularization has the same effect when it first starts, however, as it becomes bigger, it has different effects .

#Now we find the best model and fit it on the twe split data sets.
```{r}
collect_metrics(tune_res)
best_penalty <- select_best(tune_res, metric = "rsq")
ridge_final <- finalize_workflow(ridge_workflow, best_penalty)
ridge_final_fit <- fit(ridge_final, data = cars_train)
augment(ridge_final_fit, new_data = cars_train) %>%
  rsq(truth = Price_in_thousands, estimate = .pred)
augment(ridge_final_fit, new_data = cars_test) %>%
  rsq(truth = Price_in_thousands, estimate = .pred)
```

#Now we fit the lasso model
```{r}
lasso_recipe <- 
  recipe(formula = Price_in_thousands ~ .,data = cars_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors())

lasso_spec <- 
  linear_reg(penalty = tune(), mixture = 1) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet") 

lasso_workflow <- workflow() %>% 
  add_recipe(lasso_recipe) %>% 
  add_model(lasso_spec)

penalty_grid <- grid_regular(penalty(range = c(-2, 2)), levels = 50)

tune_res <- tune_grid(
  lasso_workflow,
  resamples = cars_folds, 
  grid = penalty_grid
)

autoplot(tune_res)

```

#Now we find the best model and do the prediction

```{r}
best_penalty <- select_best(tune_res, metric = "rsq")
lasso_final <- finalize_workflow(lasso_workflow, best_penalty)
lasso_final_fit <- fit(lasso_final, data = cars_train)
augment(lasso_final_fit, new_data = cars_train) %>%
  rsq(truth = Price_in_thousands, estimate = .pred)
augment(lasso_final_fit, new_data = cars_test) %>%
  rsq(truth = Price_in_thousands, estimate = .pred)
```

#keep going to do other models 

#regression tree model
```{r}
tree_spec <- decision_tree() %>%
  set_engine("rpart")
reg_tree_spec <- tree_spec %>%
  set_mode("regression")
reg_tree_fit <- fit(reg_tree_spec,Price_in_thousands ~ .,
                    data=cars_train)
augment(reg_tree_fit, new_data = cars_train) %>%
  rmse(truth = Price_in_thousands, estimate = .pred)

```



```{r}
reg_tree_fit %>%
  extract_fit_engine() %>%
  rpart.plot()
```

#find the best one 

```{r}
reg_tree_wf <- workflow() %>%
  add_model(reg_tree_spec %>% set_args(cost_complexity = tune())) %>%
  add_formula(Price_in_thousands ~ .)

set.seed(1202)
ames_fold <- vfold_cv(cars_train)

param_grid <- grid_regular(cost_complexity(range = c(-4, -1)), levels = 10)

tune_res <- tune_grid(
  reg_tree_wf, 
  resamples = ames_fold, 
  grid = param_grid
)
```

#let s see how the best result looks

```{r}
autoplot(tune_res)
```

#fit the model with the best one out of these

```{r}
best_complexity <- select_best(tune_res, metric = "rmse")
reg_tree_final <- finalize_workflow(reg_tree_wf, best_complexity)
reg_tree_final_fit <- fit(reg_tree_final, data = cars_train)
reg_tree_final_fit %>%
  extract_fit_engine() %>%
  rpart.plot()
```



```{r}
augment(reg_tree_final_fit, new_data = cars_train) %>%
  rmse(truth = Price_in_thousands, estimate = .pred)

```
The results looks great in my opinion.

#random forest model
```{r}
rf_model <- rand_forest(
  mtry = tune(),
  mode = "regression") %>%
  set_engine("ranger")

rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(cars_recipe)

set.seed(1202)

cars_rf <- tibble(mtry=seq(from=1, to=10, by=2))

cars_results_rf <- rf_workflow %>%
  tune_grid(resamples=cars_folds, grid=cars_rf)
```
```{r}
autoplot(cars_results_rf, metric = "rmse")
```

```{r}
show_best(cars_results_rf, metric = "rmse")
```

lowest rmse occurs when the mtry is 5

```{r}
cars_spec_rf <- rand_forest(mtry=5) %>%
  set_engine("ranger")%>%
  set_mode("regression")
cars_fit_rf <- workflow() %>%
  add_recipe(cars_recipe) %>%
  add_model(cars_spec_rf)%>%
  fit(data=cars_train)
cars_summary_rf <- cars_fit_rf %>%
  predict(cars_test) %>%
  bind_cols(cars_test) %>%
  metrics(truth=Price_in_thousands, estimate=.pred) %>%
  filter(.metric=="rmse")
cars_summary_rf
```
Visualize the performance on both training and testing data set

```{r}
augment(cars_fit_rf, new_data = cars_train) %>%
  ggplot(aes(Price_in_thousands, .pred)) +
  geom_abline() +
  geom_point(alpha = 0.5)

augment(cars_fit_rf, new_data = cars_test) %>%
  ggplot(aes(Price_in_thousands, .pred)) +
  geom_abline() +
  geom_point(alpha = 0.5)
```
#Conclusion
Honestly, I still think the cars brands affects the price somehow, but we are not going to discuss more in this project since this database shows that car brands does not have much affection. I thought cars length will affect the price range somehow and the fact is I am wrong. I guess a bigger car means more space, but it does not mean it needs to be expensive if it does not have a good horsepower or engine. From the project, it is clear that the most effective factor for price is horsepower. I further used different models to see how they perform.

The models i have used in this project are: random forest model, regression tree model, ridge regression model  and lasso regression  model and polynomial regression model. I chose to use regression model since I am using numerical variables to predict continuous value. The best model is going to be the polynomial regression model since it has the smallest sq error or rmse. These two values all proved polynomial regression is a great model.I am not surprised at all since if a car has a higher horsepower, it must mean the engine and other important parts of the car is class, which will cost more to buy. I do not see any other models performed really bad, which is kind of surprising. All the points are close to the line we plot, which is great.

After this class, it is my first time to learn machine  learning in my life. I learned how amazing it is to fit models and figure out different things, which will provide help to real life. I decided to further study in this area for my graduate schools. I wish I can even find an job or intern in the years after and use my skills to apply on my interests. Maybe I will be able to analyze much more complicated things in the future. At the end, thank you professor for this great quarter!