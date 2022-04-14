# Pstat 131 Hw 3
install.packages('MASS')
library(ggplot2)
library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggthemes)
tidymodels_prefer()
library(readr)
library(corrr)
library(klaR)
library(MASS)

# Read in csv data file and set seed
titanic <- read_csv("titanic.csv")
set.seed(777)

# Changing survived and pclass to factors
titanic$survived =  factor(titanic$survived, levels = c("Yes", "No")) # Note can use parse_factor() in order to give a warning when there is a value not in the set
titanic$pclass =  factor(titanic$pclass)

class(titanic$survived)
class(titanic$pclass)

# Question 1
titanic_split <- initial_split(titanic, prop = 0.80,
                               strata = survived)  
titanic_train <- training(titanic_split)
titanic_test <- testing(titanic_split)

head(titanic_train)
  ## There are some missing values for age which could pose an issue
  ## there are also missing values for cabin however this would mean they did not have a cabin
  ## could change cabin to true or false

  ## It is a good idea to use stratified sampling for this data because survived has 2 different levels and we want a proportionate amount of yes and no's for the data otherwise it could be skewed towards did not survive or not have any survived observations( yes value for survived) 

# Question 2
titanic_train %>% 
  ggplot(aes(x = survived)) +
  geom_bar()

  ## There are almost double the amount of people that did not survive then did survive 
  ## ie the amount of No value is almost twice as big as yes

# Question 3
 # Correlation matrix
titanic_train %>% 
  select(where(is.numeric)) %>%
  cor() %>% 
  corrplot(type = 'lower',
           method = 'number')

 # visualization of correlation matrix
cor_titanic <- titanic_train %>%
  select(where(is.numeric)) %>%
  correlate()
rplot(cor_titanic)

  ## age and sib_sp are negatively correlated and sib_sp and parch are positively correlated

# Question 4
titanic_recipe <- recipe(survived ~ pclass + sex + age + sib_sp + parch + fare, data = titanic_train) %>%
  step_impute_linear(age, impute_with = imp_vars(all_predictors())) %>%  # imputes age to fix missing values
  step_dummy(all_nominal_predictors()) %>%   # creates dummy variables
  step_interact(terms = sex_male ~ fare) %>%    # Note: had to use sex_male since sex was made into a dummy variable
  step_interact(terms = age ~ fare)  # Interactions created

View(titanic_recipe)

# Question 5
 # creating engine
log_reg <- logistic_reg() %>% 
  set_engine("glm") %>% 
  set_mode("classification")

 # creating workflow
log_wkflow <- workflow() %>% 
  add_model(log_reg) %>% 
  add_recipe(titanic_recipe)

 # using fit to apply workflow to training data
log_fit <- fit(log_wkflow, titanic_train)

# Question 6
 # creating engine
lda_mod <- discrim_linear() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

 # creating workflow
lda_wkflow <- workflow() %>% 
  add_model(lda_mod) %>% 
  add_recipe(titanic_recipe)

 # using fit to apply workflow to training data
lda_fit <- fit(lda_wkflow, titanic_train)


# Question 7
 # creating engine
qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>% 
  set_engine("MASS")

 # creating workflow
qda_wkflow <- workflow() %>% 
  add_model(qda_mod) %>% 
  add_recipe(titanic_recipe)

 # using fit to apply workflow to training data
qda_fit <- fit(qda_wkflow, titanic_train)


# Question 8
 # creating engine
nb_mod <- naive_Bayes() %>% 
  set_mode("classification") %>% 
  set_engine("klaR") %>% 
  set_args(usekernel = FALSE) 

 # creating workflow
nb_wkflow <- workflow() %>% 
  add_model(nb_mod) %>% 
  add_recipe(titanic_recipe)

 # using fit to apply workflow to training data
nb_fit <- fit(nb_wkflow, titanic_train)

# Question 9


# Question 10

# to calculate auc simply do auc(roc())








