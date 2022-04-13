# Pstat 131 Hw 3

library(ggplot2)
library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggthemes)
tidymodels_prefer()
library(readr)

# Read in csv data file and set seed
titanic <- read_csv("titanic.csv")
set.seed(777)
   # remember says i need to factor survived and pclass

# Question 1
titanic_split <- initial_split(titanic, prop = 0.80,
                               strata = survived)  
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)

head(titanic_train)
# There are some missing values for age which could pose an issue
# there are also missing values for cabin however this would mean they did not have a cabin
# could change cabin to true or false

# It is a good idea to use stratified sampling for this data because survived has 2 different levels and we want a proportionate amount of yes and no's for the data otherwise it could be skewed towards did not survive or not have any survived observations( yes value for survived) 

# Question 2


# Question 3


# Question 4


# Question 5


# Question 6


# Question 7


# Question 8


# Question 9


# Question 10









