# Pstat 131 Hw 3

library(ggplot2)
library(tidyverse)
library(tidymodels)
library(corrplot)
library(ggthemes)
tidymodels_prefer()
library(readr)

titanic <- read_csv("titanic.csv")

View(titanic)
