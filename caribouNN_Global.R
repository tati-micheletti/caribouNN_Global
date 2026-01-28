
defineModule(sim, list(
  name = "caribouNN_Global",
  description = "Trains a global NN on a subset to rank feature importance via Permutation",
  keywords = c("Neural Network", "Feature Importance", "Luz"),
  authors = structure(list(list(given = "Tati", family = "Micheletti", role = c("aut", "cre"), email = "tati.micheletti@gmail.com", comment = NULL)), class = "person"),
  childModules = character(0),
  version = list(caribouNN_Global = "0.0.0.1"),
  timeframe = as.POSIXlt(c(NA, NA)),
  timeunit = "year",
  citation = list("citation.bib"),
  documentation = list("NEWS.md", "README.md", "caribouNN_Global.Rmd"),
  reqdPkgs = list("SpaDES.core (>= 3.0.4)", "ggplot2", "data.table", "torch", "luz"),
  parameters = bindrows(
    #defineParameter("paramName", "paramClass", value, min, max, "parameter description"),
    defineParameter(".plots", "character", "screen", NA, NA,
                    "Used by Plots function, which can be optionally used here"),
    defineParameter(".plotInitialTime", "numeric", start(sim), NA, NA,
                    "Describes the simulation time at which the first plot event should occur."),
    defineParameter(".plotInterval", "numeric", NA, NA, NA,
                    "Describes the simulation time interval between plot events."),
    defineParameter(".saveInitialTime", "numeric", NA, NA, NA,
                    "Describes the simulation time at which the first save event should occur."),
    defineParameter(".saveInterval", "numeric", NA, NA, NA,
                    "This describes the simulation time interval between save events."),
    defineParameter(".studyAreaName", "character", NA, NA, NA,
                    "Human-readable name for the study area used - e.g., a hash of the study",
                          "area obtained using `reproducible::studyAreaName()`"),
    ## .seed is optional: `list('init' = 123)` will `set.seed(123)` for the `init` event only.
    defineParameter(".seed", "list", list(), NA, NA,
                    "Named list of seeds to use for each event (names)."),
    defineParameter(".useCache", "logical", FALSE, NA, NA,
                    "Should caching of events or module be used?"),
    defineParameter("sampleSize", "numeric", 0.20, 0.01, 1.0, "Fraction of data to use for ranking (0.2 = 20%)"),
    defineParameter("epochs", "numeric", 100, 10, 200, 
                    "Epochs for the ranking model (keep low for speed)"),
    defineParameter("batchSize", "numeric", 512, 32, 4096, 
                    "Batch size"),
    defineParameter("learningRate", "numeric", 0.01, 0.001, 0.1, 
                    paste0("Learning rate. The smaller it is, the longer it takes, but the more",
                           " precise to find the best parameters.")),
    defineParameter("device", "character", "cpu", NA, NA, 
                    "Device (cpu/cuda). With high RAM, cpu is better.")
  ),
  inputObjects = bindrows(
    expectsInput("extractedVariables", "data.table", "Raw feature table")
  ),
  outputObjects = bindrows(
    #createsOutput("objectName", "objectClass", "output object description", ...),
    createsOutput("featurePriority", "character", 
                  "Ordered list of variable names based on importance"),
    createsOutput("preparedData", "list", 
                  "List containing information for running the model"),
    createsOutput("globalModel", "luz_module_fitted", 
                  "The trained global model"),
    createsOutput("preparedDataFinal", "data.table", 
                  "Dataset of raw features after preparation (interactions added, etc.)")
  )
))

doEvent.caribouNN_Global = function(sim, eventTime, eventType) {
  switch(
    eventType,
    init = {
      ### check for more detailed object dependencies:
      ### (use `checkObject` or similar)

      # do stuff for this event
      sim <- scheduleEvent(sim, time(sim), "caribouNN_Global", "prepareData")
      sim <- scheduleEvent(sim, time(sim), "caribouNN_Global", "trainTheModel")
      sim <- scheduleEvent(sim, time(sim), "caribouNN_Global", "rankFeatures")
      sim <- scheduleEvent(sim, time(sim), "caribouNN_Global", "calculateRSS")
    },
    prepareData = {
      
      # TODO If the data is prepared and nobody wants to redo it, just use it!
       sim$preparedData <- prepareNNdata(extractedVariables = sim$extractedVariables)
       sim$preparedDataFinal <- sim$preparedData$preparedDataFinal
       fwrite(sim$preparedDataFinal, file = file.path(outputPath(sim), 
                                                      "preparedModelInputs.csv"))
    },
    trainTheModel = {
      # TODO If the model has been fit and is saved and there is no reason to redo it,
      # just load it.
      sim$globalModel <- trainingNN(preparedData = sim$preparedData$modelPrep, 
                                    batchSize = P(sim)$batchSize,
                                    epoch = P(sim)$epoch,
                                    learningRate =  P(sim)$learningRate,
                                    outputDir = outputPath(sim))
    },
      rankFeatures = {
        
        sim$featurePriority <- featureImportance(globalModel = sim$globalModel, 
                                                 preparedData = sim$preparedData, 
                                                 batchSize = P(sim)$batchSize)
    },
    calculateRSS = {
      
      sim$RSS <- generateRSS(preparedDataFinal = sim$preparedDataFinal, 
                                               globalModel = sim$globalModel,
                                               preparedData = sim$preparedData$modelPrep)
    },
    warning(noEventWarning(sim))
  )
  return(invisible(sim))
}

.inputObjects <- function(sim) {
  #cacheTags <- c(currentModule(sim), "function:.inputObjects") ## uncomment this if Cache is being used
  dPath <- asPath(getOption("reproducible.destinationPath", dataPath(sim)), 1)
  message(currentModule(sim), ": using dataPath '", dPath, "'.")
  if (!suppliedElsewhere("extractedVariables", sim = sim)){
    stop("No defaults have been implemented yet...")
    
    # Require::Require("osfr")
    # osf_auth("")
    # my_file <- osf_retrieve_file("6970e858f56e9335cc0729b2")
    # pathToStore <- file.path("C:/Users/Tati/GitHub/TestNN", my_file$name)
    # if (!file.exists(pathToStore)){
    #   osf_download(x = my_file, path = pathToStore)
    # }
    # sim$extractedVariables <- data.table::fread(pathToStore)
    
  }
  return(invisible(sim))
}
