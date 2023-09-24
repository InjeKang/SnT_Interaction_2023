# Variables
packages = c("here", "stm", "stminsights", "tidyverse", "knitr", "topicmodels",  "tidytext",
"data.table", "lubridate", "reticulate", "dplyr", "readxl", "kableExtra", "openxlsx",
"gridExtra", "ggrepel", "ggplot2", "plyr", "reticulate", "igraph", "reshape2", "tm")

install_package <- function(packages) {
  for (package in packages) {
    if (!require(package, character.only = TRUE)) {
      tryCatch(
        install.packages(package),
        error = function(e) {
          print(paste("Failed to install package:", package))
        }
      )
    }
  }
}

load_package <- function(packages) {
  lapply(packages, require, character.only = TRUE)
}

# Call the functions
install_package(packages)
load_package(packages)


