generateRSS <- function(preparedDataFinal, globalModelPath, preparedData, 
                        featurePriority, outDir) {
  
  # 1. LOAD ONCE (Critical for speed)
  message("Loading model and tensors into memory...")

  loadedModel  <- luz::luz_load(globalModelPath)
  loadedTensor <- torch::torch_load(preparedData$globalTensorX)
  loadedIds    <- torch::torch_load(preparedData$globalTensorID)
  
  # 2. Scaling Info
  scalingInfo <- dataStats(preparedDataFinal, featurePriority)
  
  # 3. Interaction Map
  interVars <- c("prop_needleleaf_start", "prop_mixedforest_start", "prop_veg_start",
                 "prop_wets_start", "timeSinceFire_startLog", "timeSinceHarvest_startLog",
                 "distpaved_startLog", "distunpaved_startLog", "distpolys_startLog")
  interMap <- setNames(as.list(paste0("inter_logSl_x_", interVars)), interVars)
  
  # 4. Filter Variables
  allFeatures <- preparedData$featureCandidates
  varsToPlot  <- grep("^inter_", allFeatures, invert = TRUE, value = TRUE)
  varsToPlot  <- unique(c(varsToPlot, "logSl", "cosTa"))
  varsToPlot  <- setdiff(varsToPlot, "sinTa")
  
  plotList <- list()
  rssResultsList <- list()
  
  # Define Sequences
  seqDist  <- seq(0, 5000, length.out = 100)
  seqTime  <- seq(0, 60, length.out = 100)
  seqProp  <- seq(0, 1, length.out = 100)
  seqAngle <- seq(-pi, pi, length.out = 100) 
  seqSl    <- seq(0, 5000, length.out = 100)
  
  # 5. LOOP
  for (feat in varsToPlot) {
    
    # Initialize iteration-specific variables explicitly
    targetValues <- numeric(0)
    xAxisValues  <- numeric(0)
    xLabel       <- ""
    
    if (grepl("dist", feat) && grepl("Log", feat)) {
      xAxisValues <- seqDist
      targetValues <- log(xAxisValues + 1)
      xLabel <- paste0("Distance (m) ", ifelse(grepl("_start", feat), "[Start]", "[End]"))
    } else if (grepl("time", feat) && grepl("Log", feat)) {
      xAxisValues <- seqTime
      targetValues <- log(xAxisValues + 1)
      xLabel <- paste0("Years ", ifelse(grepl("_start", feat), "[Start]", "[End]"))
    } else if (grepl("prop_", feat)) {
      xAxisValues <- seqProp
      targetValues <- xAxisValues
      xLabel <- paste0("Proportion ", ifelse(grepl("_start", feat), "[Start]", "[End]"))
    } else if (feat == "cosTa") {
      xAxisValues <- seqAngle
      targetValues <- cos(xAxisValues)
      xLabel <- "Turn Angle (Radians)"
    } else if (feat == "logSl") {
      xAxisValues <- seqSl
      targetValues <- log(xAxisValues + 1)
      xLabel <- "Step Length (m)"
    } else { next }
    
    if (!feat %in% names(scalingInfo)) {
      message("Skipping ", feat, ": No scaling stats.")
      next
    }
    
    message(" -> Plotting: ", feat)
    
    tryCatch({
      # 6. CALL
      rssData <- calculateRSS(
        fittedModel = loadedModel, 
        datasetTensor = loadedTensor, 
        idTensor = loadedIds,
        featureName = feat, 
        targetValues = targetValues, # Explicitly passing the local variable
        scalingStats = scalingInfo,
        featureList = allFeatures,
        interactionMap = interMap
      )
      
      rssData$xAxis <- xAxisValues
      rssResultsList[[feat]] <- rssData
      
      p <- ggplot(rssData, aes(x = xAxis, y = logRSS)) +
        geom_line(color = "#377eb8", linewidth = 1.2) +
        geom_ribbon(aes(ymin = 0, ymax = logRSS), alpha = 0.1, fill = "#377eb8") +
        geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
        theme_minimal() +
        labs(title = paste("RSS:", feat), y = "log-RSS", x = xLabel)
      
      plotList[[feat]] <- p
      
      # Save individual plots
      plotPath <- file.path(outDir, paste0("RSS_",feat,".png"))
      png(plotPath, width = 12, height = 10)
      p
      dev.off()
      
      
    }, error = function(e) {
      message("Error plotting ", feat, ": ", e$message)
    })
  }
  
  # 7. SAVE
  if (length(plotList) > 0) {
    pdfPath <- file.path(outDir, "RSS_Plots_Global.pdf")
    message("Saving all plots to PDF: ", pdfPath)
    pdf(pdfPath, width = 12, height = 10)
    for (i in seq(1, length(plotList), by = 9)) {
      indices <- i:min(i + 8, length(plotList))
      gridExtra::grid.arrange(grobs = plotList[indices], ncol = 3)
    }
    dev.off()
  }
  
  return(list(RSSplots = plotList, RSSdata = rssResultsList))
}
