generateRSS <- function(preparedDataFinal, 
                        globalModel,
                        preparedData){
  
  # 1. Get Scaling Stats (From your RAW data, before prepareNNdata)
  # You need the original unscaled data.table
  scalingInfo <- dataStats(preparedDataFinal, featurePriority)
  
  # 2. Define Interactions Mapping
  # Map Main Variable -> Interaction Column Name
  # 1. Define the variables that have interactions
  interVars <- c("prop_needleleaf_start", "prop_mixedforest_start", "prop_veg_start",
                 "prop_wets_start", "timeSinceFire_startLog", "timeSinceHarvest_startLog",
                 "distpaved_startLog", "distunpaved_startLog", "distpolys_startLog")
  
  # 2. Create the map automatically
  # This creates a list where: Key = Main Variable, Value = Interaction Variable
  inter_map <- setNames(
    as.list(paste0("inter_logSl_x_", interVars)), 
    interVars
  )
  
  # 3. Choose Variable to Plot
  # Example: Distance to Paved Road (Log Transformed)
  # We want to plot range 0 to 5000 meters.
  # Note: The model used Log(x+1). So we must feed Log(x+1) values.
  meters <- seq(0, 5000, by = 50)
  log_meters <- log(meters + 1)
  
  print("Decide which variables to plot. Make a loop for them.")
  browser()
  
  rss_data <- calculateRSS(
    model = globalModel, 
    dataset_tensor = preparedData$globalTensorX, 
    feature_name = "distpaved_endLog", # Variable name in the model
    target_values = log_meters,        # Range of values (Log scale)
    scaling_stats = scalingInfo,
    feature_list = preparedData$featureCandidates,
    interaction_pairs = inter_map      # Pass list if interaction exists
  )
  
  # 4. Plot (Convert X axis back to Meters for readability)
  rss_data$Meters <- meters
  
  p1 <- ggplot(rss_data, aes(x = Meters, y = logRSS)) +
    geom_line(color = "blue", size = 1.5) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    theme_minimal() +
    labs(
      title = "Neural Network RSS: Distance to Paved Road",
      subtitle = "Compared to mean availability",
      y = "log-RSS (Selection Strength)",
      x = "Distance (m)"
    )
  return(list(RSSplot = p1,
              RSSdata = rss_data))
}