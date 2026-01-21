prepareNNdata <- function(extractedVariables){
  message("Preparing Data and Creating Interactions...")
  dt <- copy(extractedVariables)
  
  # 1. Cleaning
  dt <- na.omit(dt, cols = c("sl_", "ta_", "indiv_step_id", "id"))
  dt[, indiv_step_id := as.factor(indiv_step_id)]
  dt[, id := as.factor(id)]
  setorder(dt, indiv_step_id, -case_) # Critical sort
  
  # 2. Basic Transforms
  dt[, logSl := log(sl_ + 1)]
  dt[, cosTa := cos(ta_)]
  dt[, sinTa := sin(ta_)]
  
  varsToLog <- c("timeSinceFire", "timeSinceHarvest", "distpaved", "distunpaved", "distpolys")
  for(v in varsToLog) {
    if(paste0(v, "_start") %in% names(dt)) dt[, (paste0(v, "_startLog")) := log(get(paste0(v, "_start")) + 1)]
    if(paste0(v, "_end") %in% names(dt))   dt[, (paste0(v, "_endLog")) := log(get(paste0(v, "_end")) + 1)]
  }
  
  # 3. Create Interactions (The Full List)
  interVars <- c("prop_needleleaf_start", "prop_mixedforest_start", "prop_veg_start", "prop_wets_start",
                 "timeSinceFire_startLog", "timeSinceHarvest_startLog",
                 "distpaved_startLog", "distunpaved_startLog", "distpolys_startLog")
  
  interactionCols <- c()
  for (var in interVars) {
    if(var %in% names(dt)) {
      newCol <- paste0("inter_logSl_x_", var)
      dt[, (newCol) := logSl * get(var)]
      interactionCols <- c(interactionCols, newCol)
    }
  }
  
  # 4. Define All Candidates
  candidates <- c(
    "logSl", "cosTa", "sinTa",
    "prop_needleleaf_end", "prop_mixedforest_end", "prop_veg_end", "prop_wets_end",
    "timeSinceFire_endLog", "timeSinceHarvest_endLog", 
    "distpaved_endLog", "distunpaved_endLog", "distpolys_endLog",
    "prop_needleleaf_start", "prop_mixedforest_start", "prop_veg_start", "prop_wets_start",
    "timeSinceFire_startLog", "timeSinceHarvest_startLog",
    "distpaved_startLog", "distunpaved_startLog", "distpolys_startLog",
    interactionCols
  )
  # Filter to what actually exists in data
  dtfeatureCandidates <- intersect(candidates, names(dt))
  
  # Scale
  dt[, (dtfeatureCandidates) := lapply(.SD, function(x) as.numeric(scale(x))), .SDcols = dtfeatureCandidates]
  dt[, idIndex := as.numeric(as.factor(id))]
  
  # Filter Step Count (Mode)
  counts <- dt[, .N, by = indiv_step_id]
  steps <- as.numeric(names(sort(table(counts$N), decreasing=TRUE)[1]))
  dt <- dt[indiv_step_id %in% counts[N == steps, indiv_step_id]]
  
  # Shape Tensor
  dtglobalSteps <- steps
  dtglobalNAnimals <- max(dt$idIndex)
  
  mat <- as.matrix(dt[, ..dtfeatureCandidates])
  arr <- array(mat, dim = c(steps, uniqueN(dt$indiv_step_id), length(dtfeatureCandidates)))
  
  # Store in sim
  globalTensorX <- torch::torch_tensor(aperm(arr, c(2, 1, 3)), dtype = torch::torch_float())
  globalTensorID <- torch::torch_tensor(dt[case_==TRUE, idIndex], dtype = torch::torch_long())
  
  return(list(globalTensorX = globalTensorX,
              globalTensorID = globalTensorID,
              globalNAnimals = dtglobalNAnimals,
              featureCandidates = dtfeatureCandidates))
}