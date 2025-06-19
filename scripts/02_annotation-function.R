     


      # Requirements Notice:
      # 
      # This script relies on Python interoperability via the reticulate package in R. 
      # To run it successfully, you must have Python installed on your system and 
      # properly configured for use with reticulate. Specifically:
      #   
      # 1) A working Python installation (version 3.8 or later recommended)
      # 
      # 2) The following Python packages installed in the active environment:
      #   
      #  - transformers
      # 
      #  - torch
      # 
      #  - protobuf
      # 
      # 3) A valid Hugging Face access token set as an environment 
      # variable (HF_TOKEN)
      # 
      # You can create and manage a virtual environment using reticulate::virtualenv_create() 
      # or reticulate::conda_create(), and install the required Python packages using 
      # reticulate::py_install(). Make sure the environment is activated or properly 
      # referenced using reticulate::use_virtualenv() or use_condaenv() before running
      # the script.




# 1. LOAD LIBRARIES AND ENVIRONMENT ----
library(reticulate)
library(stringr)
library(readr)
library(dplyr)
library(tidyr)
library(purrr)
library(here)


torch <- reticulate::import("torch")
transformers <- reticulate::import("transformers")
protobuf <- reticulate::import("google.protobuf") 

Sys.setenv(HF_TOKEN = "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")
token <- Sys.getenv("HF_TOKEN")

model_ids <- c(
  "mistralai/Mistral-7B-Instruct-v0.3",
  "deepseek-ai/deepseek-llm-7b-chat",
  "Qwen/Qwen3-8B",
  "01-ai/Yi-1.5-9B-Chat",
  "microsoft/Phi-3-mini-4k-instruct"
)

# 2. LOAD DATA AND ASSIGN ALL MODELS TO EACH TEXT ----
data <- read_csv(here("data", "data.csv")) 

set.seed(123)

texts_to_annotate <- data %>%
  slice_sample(prop = 1)  # shuffles all rows

# Assign models: all 5 models to every row
texts_to_annotate <- texts_to_annotate %>%
  rowwise() %>%
  mutate(models = list(sample(model_ids, 5))) %>%
  unnest(models) %>%
  rename(model = models) %>%
  mutate(index = row_number())

# 3. LOAD OR INIT NEW PROGRESS FILE ---- COMMENTED OUT
# progress_file <- here("data", "new_annotations.rds") 

# if (file.exists(progress_file)) {
#   results_all <- readRDS(progress_file)
#   message("Progress loaded: ", nrow(results_all), " rows")
# } else {
#   results_all <- data.frame()
#   message("No previous progress found. Starting fresh.")
# }

# 4. LOAD MODELS ONCE ----
load_model_and_tokenizer <- function(model_id) {
  tokenizer <- transformers$AutoTokenizer$from_pretrained(model_id, token = Sys.getenv("HF_TOKEN"))
  if (is.null(tokenizer$pad_token)) tokenizer$pad_token <- tokenizer$eos_token
  model <- transformers$AutoModelForCausalLM$from_pretrained(model_id, token = Sys.getenv("HF_TOKEN"))
  list(tokenizer = tokenizer, model = model)
}

model_objects <- setNames(lapply(model_ids, load_model_and_tokenizer), model_ids)

# 5. FILTER OUT ALREADY DONE COMBINATIONS ---- COMMENTED OUT
# if (nrow(results_all) > 0) {
#   texts_to_annotate <- anti_join(texts_to_annotate, results_all, by = c("X", "model"))
# }

# 6. ANNOTATION FUNCTION ----
annotate_texts_batch <- function(df, model_objects) {
  result_list <- lapply(seq_len(nrow(df)), function(i) {
    tryCatch({
      txt <- df$text[i]
      model_id <- df$model[i]
      components <- model_objects[[model_id]]
      row_id <- df$index[i]
      row_X <- df$X[i]  # <- original index column
      
      prompt <- paste0(
        "Classify the following text as EITHER hate speech OR not hate speech.\n\n",
        "Text: ", txt, "\n\n",
        "Complete with ONLY a label: 'hate speech' OR 'not hate speech'. DO NOT provide any explanation.\n\n",
        "Answer: This text is classified as"
      )

      inputs <- components$tokenizer(
        prompt,
        return_tensors = "pt",
        padding = TRUE,
        truncation = TRUE,
        max_length = as.integer(512)
      )

      with(torch$no_grad(), {
        output <- components$model$generate(
          input_ids = inputs$input_ids,
          attention_mask = inputs$attention_mask,
          max_new_tokens = as.integer(10),
          pad_token_id = components$tokenizer$eos_token_id
        )
      })

      decoded <- components$tokenizer$decode(output[0], skip_special_tokens = TRUE)

      label <- stringr::str_match(
        tolower(decoded),
        "this text is classified as\\s+['\"]?(not hate speech|hate speech)['\"]?"
      )[, 2]

      data.frame(
        X = row_X,  # text identifier
        index = row_id, # row identifier
        text = txt,
        prediction = label,
        model = model_id,
        stringsAsFactors = FALSE
      )
    }, error = function(e) {
      message(sprintf("Error on row %d (%s): %s", i, df$model[i], e$message))
      data.frame(
        X = df$X[i],
        index = df$index[i],
        text = df$text[i],
        prediction = NA_character_,
        model = df$model[i],
        stringsAsFactors = FALSE
      )
    })
  })

  bind_rows(result_list)
}
# 7. BATCH LOOP ----
batch_size <- 50
num_batches <- ceiling(nrow(texts_to_annotate) / batch_size)

for (i in seq_len(num_batches)) {
  batch <- texts_to_annotate[((i - 1) * batch_size + 1):(i * batch_size), , drop = FALSE]
  batch <- na.omit(batch)
  if (nrow(batch) == 0) next
  
  cat(sprintf("Processing batch %d of %d\n", i, num_batches))
  results_all <- annotate_texts_batch(texts_to_annotate, model_objects)
  
  saveRDS(results_all, here("data", "llm_annotations.rds")) # already in the repo
}


