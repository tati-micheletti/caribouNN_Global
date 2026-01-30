calculateRSS <- function(fittedModel, datasetTensor, idTensor, featureName, 
                         targetValues, scalingStats, 
                         featureList, interactionMap = NULL) {
  
  # 1. Setup Device
  device <- torch_device("cpu")
  
  # 2. Extract Model Module Safely
  if (inherits(fittedModel, "luz_module_fitted")) {
    ModelObj <- fittedModel$model
  } else {
    ModelObj <- fittedModel
  }
  ModelObj$eval()
  ModelObj$to(device = device)
  
  # 3. Identify Indices
  featIdx  <- which(featureList == featureName)
  logSlIdx <- which(featureList == "logSl")
  
  interIdx <- NULL
  if (!is.null(interactionMap) && featureName %in% names(interactionMap)) {
    interName <- interactionMap[[featureName]]
    interIdx  <- which(featureList == interName)
  }
  
  # 4. Get Scaling Params
  mu    <- scalingStats[[featureName]]$mean
  sigma <- scalingStats[[featureName]]$sd
  
  # 5. Subset Data
  nSamples      <- min(500, datasetTensor$size(1))
  baseTensor    <- datasetTensor$narrow(1, 1, nSamples)$clone()$to(device = device)
  realIds       <- idTensor$narrow(1, 1, nSamples)$to(device = device)
  
  # 6. Baseline (Set variable to Mean = 0)
  # Create index tensor safely to avoid dtype conversion errors
  idxT <- torch_tensor(as.integer(featIdx))$to(dtype = torch_long(), device = device)
  baseTensor$index_fill_(3, idxT, 0)
  
  if(!is.null(interIdx)) {
    idxInterT <- torch_tensor(as.integer(interIdx))$to(dtype = torch_long(), device = device)
    baseTensor$index_fill_(3, idxInterT, 0)
  }
  
  with_no_grad({
    refScores    <- ModelObj(list(x = baseTensor, id = realIds))
    avgRefScore  <- as.numeric(refScores$mean()$cpu())
  })
  
  # 7. Prediction Loop
  results <- data.frame(Value = targetValues, logRSS = NA)
  
  for (i in seq_along(targetValues)) {
    valScaled <- (targetValues[i] - mu) / sigma
    
    workTensor <- baseTensor$clone()
    workTensor$index_fill_(3, idxT, valScaled)
    
    if (!is.null(interIdx)) {
      currentLogSl <- workTensor$select(3, logSlIdx)
      newInterVals <- currentLogSl * valScaled
      workTensor$narrow(3, interIdx, 1)$copy_(newInterVals$unsqueeze(3))
    }
    
    with_no_grad({
      scores   <- ModelObj(list(x = workTensor, id = realIds))
      avgScore <- as.numeric(scores$mean()$cpu())
    })
    
    results$logRSS[i] <- avgScore - avgRefScore
  }
  
  return(results)
}
