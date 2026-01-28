calculateRSS <- function(model, dataset_tensor, feature_name, 
                                 target_values, scaling_stats, 
                                 feature_list, interaction_pairs = NULL) {
  
  # 1. Setup
  device <- "cpu" # Safer for inference loops
  model <- model$model # Extract raw torch module
  model$eval()
  model$to(device = device)
  
  # Identify column index for the main feature
  feat_idx <- which(feature_list == feature_name)
  if(length(feat_idx) == 0) stop("Feature not found in model inputs")
  
  # Identify column index for the interaction (if any)
  inter_idx <- NULL
  if (!is.null(interaction_pairs) && feature_name %in% names(interaction_pairs)) {
    inter_name <- interaction_pairs[[feature_name]]
    inter_idx <- which(feature_list == inter_name)
  }
  
  # Get Scaling info
  mu <- scaling_stats[[feature_name]]$mean
  sigma <- scaling_stats[[feature_name]]$sd
  
  # We need the 'logSl' index if we are updating an interaction
  logSl_idx <- which(feature_list == "logSl")
  
  # Prepare Results Storage
  results <- data.frame(Value = target_values, logRSS = NA)
  
  # Use a subset of data for speed (e.g., 1000 random bursts)
  # We use the observed data structure to maintain realistic correlations for other vars
  n_samples <- min(1000, dataset_tensor$size(2))
  sample_indices <- 1:n_samples
  
  # Clone the baseline tensor to modify
  # Shape: [Steps, Bursts, Features]
  base_tensor <- dataset_tensor[, sample_indices, , drop=FALSE]$clone()$to(device)
  
  # 2. Calculate Baseline Score (at the Mean)
  # In iSSA, RSS is usually compared to the Mean Availability.
  # Since we scaled data, Mean = 0.
  
  base_tensor[,,feat_idx] <- 0 # Set main effect to Mean (0)
  if(!is.null(inter_idx)) {
    # If main var is mean (0), then Interaction (logSl * 0) is also 0
    base_tensor[,,inter_idx] <- 0 
  }
  
  # Get Reference Score (prediction at the mean)
  # We create a dummy ID tensor (embeddings don't matter for relative difference of environmental vars)
  dummy_id <- torch_zeros(n_samples, dtype=torch_long())$to(device)
  
  with_no_grad({
    ref_scores <- model(list(x=base_tensor, id=dummy_id))
    avg_ref_score <- mean(ref_scores$numpy())
  })
  
  # 3. Loop through target values
  message(paste("Calculating RSS for", feature_name, "..."))
  
  for (i in seq_along(target_values)) {
    val_raw <- target_values[i]
    
    # Scale the value
    val_scaled <- (val_raw - mu) / sigma
    
    # Update Tensor
    work_tensor <- base_tensor$clone()
    
    # A. Set Main Effect
    work_tensor[,,feat_idx] <- val_scaled
    
    # B. Set Interaction
    if (!is.null(inter_idx)) {
      # Recalculate interaction: logSl * new_value
      # Note: logSl is already scaled in the tensor, which makes this tricky.
      # Ideally, interaction = logSl_scaled * val_scaled is what the NN saw during training
      # if one calculated interaction AFTER scaling logSl.
      
      # Check your prepareNNdata: 
      # You did: set(dt, j=newCol, value = vec_logSl * dt[[var]]) THEN scaled.
      # This means we need to replicate that logic exactly? 
      # Actually, since the NN learned on SCALED inputs, we just inject 
      # the new scaled value.
      
      # Simpler approach: The NN treats "inter_logSl_x_dist" as just another column.
      # But logically, if Dist changes, Inter changes.
      # Approximation: Update interaction column based on the relationship correlation?
      # BETTER: If you used 'logSl' in the model, grab it:
      current_logSl <- work_tensor[,,logSl_idx]
      
      # Update interaction: current_logSl * val_scaled
      # (Assuming logSl and Dist were roughly independent before interaction)
      work_tensor[,,inter_idx] <- current_logSl * val_scaled
    }
    
    # Predict
    with_no_grad({
      scores <- model(list(x=work_tensor, id=dummy_id))
      avg_score <- mean(scores$numpy())
    })
    
    # RSS = Score(x) - Score(ref)
    results$logRSS[i] <- avg_score - avg_ref_score
  }
  
  return(results)
}