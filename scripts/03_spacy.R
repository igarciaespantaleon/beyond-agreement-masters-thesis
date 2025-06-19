library(reticulate)
library(readr)
library(here)

py_config()

# Install spaCy only once if not yet installed
py_install("spacy", method = "pip", pip = TRUE)

# Install and load spacyr if needed
if (!require("spacyr")) install.packages("spacyr")
library(spacyr)

# Initialize spacyr with specific Python path
spacy_initialize(python_executable =
                "C:/Users/Irene.DESKTOP-Q4C20CF/OneDrive/Documentos/.virtualenvs/r-spacyr/Scripts/python.exe")

# Import HS data
data <- read.csv(here("data", "data.csv")) 

# Parse texts with dependency info
texts <- data$text
names(texts) <- data$X
parsed <- spacy_parse(texts, dependency = TRUE)

# Extract subordinate clauses
subordinate_clauses <- parsed %>%
  filter(dep_rel %in% c("advcl", "ccomp", "acl")) %>%
  distinct(doc_id)

# Count subordinate clauses per document
subordinate_counts <- parsed %>%
  filter(dep_rel %in% c("advcl", "ccomp", "acl")) %>%
  count(doc_id, name = "subordinate_clause_count")

# Save result (already included in the repo)
write.csv(subordinate_counts, here("data", "subordinate_counts.csv"), 
          row.names = FALSE)
