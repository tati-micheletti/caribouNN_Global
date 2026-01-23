featureImportance <- function(globalModel, preparedData, batchSize){
  
  message("Global Model: Calculating Feature Importance...")
  
  # Unpack
  x <- preparedData$globalTensorX
  id <- preparedData$globalTensorID
  candidates <- preparedData$featureCandidates
  
  # Extract the raw torch model from the Luz object
  model <- globalModel$model 
  model$eval()
  
  # 1. DEFINE DATASET LOCALLY
  # We need this definition to create loaders for both Baseline and Permutations
  ds <- dataset("ds", 
                initialize = function(x, id) {self$x<-x; self$id<-id},
                .getitem = function(i) { 
                  list(list(x=self$x[i,,], id=self$id[i]), torch_tensor(1, dtype=torch_long())) 
                },
                .length = function() { self$x$size(1) }
  )
  
  # 2. CALCULATE BASELINE
  # Create a fresh loader for the original data
  dl <- dataloader(ds(x, id), batch_size = batchSize, shuffle = FALSE)
  
  base_loss <- 0
  count <- 0
  
  with_no_grad({
    coro::loop(for (b in dl) {
      scores <- model(b[[1]]) # Model output is [Batch, Steps]
      # Squeeze target only
      base_loss <- base_loss + nnf_cross_entropy(scores, b[[2]]$squeeze())$item()
      count <- count + 1
    })
  })
  base_loss <- base_loss / count
  message("Baseline Loss: ", round(base_loss, 4))
  
  # 3. PERMUTATION LOOP
  imp_df <- data.frame(Feature = candidates, Importance = 0)
  original_x <- x$clone()
  
  for(i in seq_along(candidates)) {
    # Shuffle column i
    perm_x <- original_x$clone()
    idx <- torch::torch_randperm(perm_x$size(1)) + 1L
    perm_x[,,i] <- perm_x[idx,,i]
    
    # Create loader for shuffled data using the 'ds' defined above
    perm_dl <- dataloader(ds(perm_x, id), batch_size = batchSize, shuffle = FALSE)
    
    new_loss <- 0
    count <- 0
    
    with_no_grad({
      coro::loop(for (b in perm_dl) {
        scores <- model(b[[1]])
        new_loss <- new_loss + nnf_cross_entropy(scores, b[[2]]$squeeze())$item()
        count <- count + 1
      })
    })
    
    imp_val <- (new_loss / count) - base_loss
    imp_df$Importance[i] <- imp_val
    cat(".")
  }
  
  # 4. Return Sorted List
  imp_df <- imp_df[order(-imp_df$Importance),]
  print(head(imp_df, 10))
  
  message("\nGlobal Feature Ranking Complete.")
  return(imp_df)
}