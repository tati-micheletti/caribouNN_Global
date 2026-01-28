dataStats <- function(preparedDataFinal, featurePriorityObject) {
  
  message("Calculate Mean and SD...")
  
  # 1. FIX THE INPUT: Extract just the column of names if it's a table
  if (is.data.frame(featurePriorityObject) || is.data.table(featurePriorityObject)) {
    feature_list <- featurePriorityObject$Feature
  } else {
    feature_list <- featurePriorityObject
  }
  
  # 1. CALCULATE STATS
  stats <- list()
  for (f in feature_list) {
    if (f %in% names(preparedDataFinal)) {
      stats[[f]] <- list(
        mean = mean(preparedDataFinal[[f]], na.rm = TRUE),
        sd = sd(preparedDataFinal[[f]], na.rm = TRUE)
      )
    } else {
      warning(paste("Could not calculate stats for:", f, "(Column missing)"))
    }
  }
  return(stats)
}
