trainingNN <- function(preparedData, batchSize, learningRate, epochs, validationSplit = 0.2){
  
  globalTensorX <- preparedData$globalTensorX
  globalTensorID <- preparedData$globalTensorID
  globalNAnimals <- preparedData$globalNAnimals
  featureCandidates <- preparedData$featureCandidates
  
  message("Global Model: Training with Early Stopping & Best Model Saving...")
  
  # 1. Dataset
  ds <- dataset("ds", 
                initialize = function(x, id) {self$x <- x; self$id <- id},
                .getitem = function(i) { 
                  list(
                    list(x=self$x[i,,], id=self$id[i]), 
                    torch_tensor(1, dtype=torch_long()) 
                  ) 
                },
                .length = function() { self$x$size(1) }
  )
  
  # 2. Split Data (Train vs Validation)
  n_samples <- globalTensorX$size(1) 
  all_indices <- 1:n_samples
  
  val_indices <- sample(all_indices, size = floor(n_samples * validationSplit))
  train_indices <- setdiff(all_indices, val_indices)
  
  train_x <- globalTensorX[train_indices, , ]
  train_id <- globalTensorID[train_indices]
  
  val_x <- globalTensorX[val_indices, , ]
  val_id <- globalTensorID[val_indices]
  
  train_dl <- dataloader(ds(train_x, train_id), batch_size = batchSize, shuffle = TRUE)
  val_dl   <- dataloader(ds(val_x, val_id),   batch_size = batchSize, shuffle = FALSE)
  
  # 3. Model
  Net <- nn_module(
    "Net",
    initialize = function(nIn, nAnimals) {
      self$idEmb <- nn_embedding(nAnimals + 1, 8)
      self$fc1 <- nn_linear(nIn + 8, 128)
      self$fc2 <- nn_linear(128, 64)
      self$out <- nn_linear(64, 1)
      self$act <- nn_selu()
    },
    forward = function(input) {
      x <- input$x; id <- input$id
      emb <- self$idEmb(id)$unsqueeze(2)$expand(c(-1,x$shape[2],-1))
      torch_cat(list(x,emb),3) %>% 
        self$fc1() %>% self$act() %>% 
        self$fc2() %>% self$act() %>% 
        self$out() %>% torch_squeeze(3)
    }
  )
  
  # 4. Loss Wrapper
  loss_wrapper <- function(input, target) {
    nnf_cross_entropy(input, target$squeeze())
  }
  
  # 5. Fit
  tmp_path <- file.path(tempdir(), "global_best_model.pt")
  
  fitted <- Net %>%
    setup(
      loss = loss_wrapper, 
      optimizer = optim_adam
    ) %>%
    set_hparams(
      nIn = length(featureCandidates), 
      nAnimals = globalNAnimals
    ) %>%
    set_opt_hparams(
      lr = learningRate
    ) %>%
    fit(
      train_dl, 
      epochs = epochs, 
      valid_data = val_dl,
      callbacks = list(
        # Save best model to disk
        luz_callback_model_checkpoint(path = tmp_path, save_best_only = TRUE),
        # Stop if no improvement
        luz_callback_early_stopping(patience = 5),
        luz_callback_lr_scheduler(
          lr_reduce_on_plateau, 
          factor = 0.5,
          patience = 2,
          verbose = TRUE
        )
      ),
      verbose = TRUE
    )
  
  message("\nGlobal Model Fitting Complete.")
  
  # 6. RELOAD BEST WEIGHTS (FIXED)
  if (file.exists(tmp_path)) {
    message("Reloading weights from best epoch...")
    checkpoint <- torch_load(tmp_path)
    
    tryCatch({
      checkpoint <- torch_load(tmp_path)
      weights_to_load <- NULL
      
      # CHECK 1: Standard Luz Key
      if (is.list(checkpoint) && "model" %in% names(checkpoint)) {
        message(" -> Detected Luz Bundle (Key: 'model'). Extracting weights.")
        weights_to_load <- checkpoint$model
        
        # CHECK 2: Alternative Luz Key
      } else if (is.list(checkpoint) && "model_state_dict" %in% names(checkpoint)) {
        message(" -> Detected Luz Bundle (Key: 'model_state_dict'). Extracting weights.")
        weights_to_load <- checkpoint$model_state_dict
        
        # CHECK 3: Raw Weights
      } else {
        message(" -> Assuming Raw State Dict.")
        weights_to_load <- checkpoint
      }
      
      fitted$model$load_state_dict(weights_to_load)
      message(" -> Success! Best model loaded.")
      
    }, error = function(e) {
      message("!!! WARNING: Could not reload best weights. Using Last Epoch. !!!")
      message("Error: ", e$message)
    })
    
    unlink(tmp_path)
  } else {
    warning("Best model checkpoint not found. Using last epoch weights.")
  }
  
  return(fitted)
}