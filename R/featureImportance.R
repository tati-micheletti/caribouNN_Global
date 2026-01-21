featureImportance <- function(globalModel, preparedData, batchSize){
  
  globalTensorX <- preparedData$globalTensorX
  globalTensorID <- preparedData$globalTensorID
  featureCandidates <- preparedData$featureCandidates
  
  # 4. Permutation Importance
  message("Global Model: Calculating Feature Importance...")
  model <- globalModel$model
  model$eval()
  
  # Get Baseline
  base_loss <- 0
  count <- 0
  # Use same DL for convenience (technically should be validation, but for RANKING it's okay)
  with_no_grad({
    coro::loop(for (b in dl) {
      scores <- model(b[[1]])$squeeze(3)
      base_loss <- base_loss + nnf_cross_entropy(scores, b[[2]])$item()
      count <- count + 1
    })
  })
  base_loss <- base_loss / count
  
  imp_df <- data.frame(Feature = featureCandidates, Importance = 0)
  
  # Permute each feature
  # Note: We perform permutation on the CPU tensor to save GPU memory if needed
  # Or directly on the tensor if it fits.
  
  original_x <- globalTensorX$clone()
  
  for(i in seq_along(featureCandidates)) {
    # Shuffle column i
    perm_x <- original_x$clone()
    idx <- torch::torch_randperm(perm_x$size(1)) + 1L
    perm_x[,,i] <- perm_x[idx,,i]
    
    # Eval
    new_loss <- 0
    count <- 0
    perm_ds <- ds(perm_x, globalTensorID)
    perm_dl <- dataloader(perm_ds, batch_size = batchSize)
    
    with_no_grad({
      coro::loop(for (b in perm_dl) {
        scores <- model(b[[1]])$squeeze(3)
        new_loss <- new_loss + nnf_cross_entropy(scores, b[[2]])$item()
        count <- count + 1
      })
    })
    
    imp_val <- (new_loss / count) - base_loss
    imp_df$Importance[i] <- imp_val
    cat(".")
  }
  
  # 5. Set Output
  imp_df <- imp_df[order(-imp_df$Importance),]
  print(head(imp_df, 10))
  
  featurePriority <- imp_df$Feature
  message("\nGlobal Feature Ranking Complete.")
  return(featurePriority)
}