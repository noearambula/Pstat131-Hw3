# Pstat 131 Hw 3

library(discrim)
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
  step_impute_linear(age) %>%  # imputes age to fix missing values
  step_dummy(all_nominal_predictors()) %>%   # creates dummy variables
  step_interact(terms = sex_male ~ fare) %>%    # Note: had to use sex_male since sex was made into a dummy variable
  step_interact(terms = age ~ fare)  # Interactions created

titanic_recipe

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
  # Logistic Regression Model Predictions + Accuracy
titanic_log_pred <- predict(log_fit, new_data = titanic_train %>% select(-survived))

titanic_log_pred = bind_cols(titanic_log_pred, titanic_train %>% select(survived))

titanic_log_pred %>%
  head

log_acc <- augment(log_fit, new_data = titanic_train) %>%
  accuracy(truth = survived, estimate = .pred_class)
log_acc

  # Linear Discriminant Analysis Model Predictions + Accuracy
titanic_lda_pred <- predict(lda_fit, new_data = titanic_train %>% select(-survived))

titanic_lda_pred = bind_cols(titanic_lda_pred, titanic_train %>% select(survived))

titanic_lda_pred %>%
  head()

lda_acc <- augment(lda_fit, new_data = titanic_train) %>%
  accuracy(truth = survived, estimate = .pred_class)
lda_acc

  # Quadratic Discriminant Analysis Model Predictions + Accuracy
titanic_qda_pred <- predict(qda_fit, new_data = titanic_train %>% select(-survived))

titanic_qda_pred = bind_cols(titanic_qda_pred, titanic_train %>% select(survived))

titanic_qda_pred %>%
  head()

qda_acc <- augment(qda_fit, new_data = titanic_train) %>%
  accuracy(truth = survived, estimate = .pred_class)
qda_acc

  #  Naive Bayes Model Predictions + Accuracy
titanic_nb_pred <- predict(nb_fit, new_data = titanic_train %>% select(-survived))

titanic_nb_pred = bind_cols(titanic_nb_pred, titanic_train %>% select(survived))

titanic_nb_pred %>%
  head()

nb_acc <- augment(nb_fit, new_data = titanic_train) %>%
  accuracy(truth = survived, estimate = .pred_class)
nb_acc

  # The model that achieved the highest accuracy on the training data was the logistic Regression model with an accuracy of 0.81 or 81%

# Question 10

  # Fitting logistic model (model w/ highest accuracy) to testing data
log_testing_fit <- fit(log_wkflow, titanic_test)

  # Accuracy on testing data
log_testing_acc <- augment(log_testing_fit, new_data = titanic_test) %>%
  accuracy(truth = survived, estimate = .pred_class)
log_testing_acc

  # Create confusion matrix and Visualize it
augment(log_testing_fit, new_data = titanic_test) %>%
  conf_mat(truth = survived, estimate = .pred_class) %>%
  autoplot(type = "heatmap")

  # Plot ROC curve
augment(log_testing_fit, new_data = titanic_test) %>%
  roc_curve(survived, .pred_Yes) %>%
  autoplot()

  # Calculate AUC
augment(log_testing_fit, new_data = titanic_test) %>%
  roc_auc(survived, .pred_Yes) 

# The model performed fairly well at predicting those who did survive as the ROC curve is closer to the top left of the graph indicating a good performance if it was close to the diagnol dotted line the predictions would be no better than random guessing
# In addition, we have an AUC of 87.3% which is pretty good as well as the higher it is the better, AUC tells how accurate the models predictions are
# When we take a look at the confusion matrix we can see that we have an overwhelming amount of true positives/negatives compared to a few false positives/negatives which is very good
log_testing_acc
log_acc

# The testing accuracy was a little higher at 0.827 or 82.7% compared to the training accuracy of 81%
# The reason this might have been higher is that the training data might have just had points that were a little better suited for our model since we stratified the testing set it makes sense that the accuracy would be similar to both sets some variation
