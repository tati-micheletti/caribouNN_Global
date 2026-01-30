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
  reqdPkgs = list("SpaDES.core (>= 3.0.4)", "ggplot2", "data.table", "torch", "luz", "gridExtra"),
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
                    "Device (cpu/cuda). With high RAM, cpu is better."),
    defineParameter("rerunPrepData", "logical", FALSE, NA, NA, 
                    "Should the dataPrep be re-run?"),
    defineParameter("rerunTraining", "logical", FALSE, NA, NA, 
                    "Should the training be re-run?"),
    defineParameter("rerunRanking", "logical", FALSE, NA, NA, 
                    "Should the feature ranking be re-run?"),
    defineParameter("rerunRSS", "logical", FALSE, NA, NA, 
                    "Should the RSS be re-run?")
  ),
  inputObjects = bindrows(
    expectsInput("extractedVariables", "data.table", "Raw feature table")
  ),
  outputObjects = bindrows(
    #createsOutput("objectName", "objectClass", "output object description", ...),
    createsOutput("featurePriority", "character", 
                  "Ordered list of variable names based on importance"),
    createsOutput("preparedData", "list", 
                  paste0("List of one list () containing information for running the model, ",
                         "and a data table (preparedDataFinal) containing Dataset of raw ",
                         "features after preparation (interactions added, etc.)")),
    createsOutput("globalModel", "character", 
                  "Path to the the trained global model")
  )
))

doEvent.caribouNN_Global = function(sim, eventTime, eventType) {
  switch(
    eventType,
    init = {
      sim <- scheduleEvent(sim, time(sim), "caribouNN_Global", "prepareData")
      sim <- scheduleEvent(sim, time(sim), "caribouNN_Global", "trainTheModel")
      sim <- scheduleEvent(sim, time(sim), "caribouNN_Global", "rankFeatures")
      sim <- scheduleEvent(sim, time(sim), "caribouNN_Global", "calculateRSS")
    },
    prepareData = {
      pathX <- file.path(outputPath(sim), "tensor_x.pt")
      pathID <- file.path(outputPath(sim), "tensor_id.pt")
      globalNAnimalsPath <- file.path(outputPath(sim), "globalNAnimals.rds")
      featureCandidatesPath <- file.path(outputPath(sim), "featureCandidates.rds")
      prepDataPath <- file.path(outputPath(sim), "preparedModelInputs.csv")
       if (all(file.exists(pathX),
               file.exists(pathID),
               file.exists(globalNAnimalsPath),
               file.exists(featureCandidatesPath),
               file.exists(prepDataPath),
               !P(sim)$rerunPrepData)){
         message("All inputs for prepareData found. Loading...")
         sim$preparedData <- list(modelPrep = list(globalTensorX = pathX,
                                                   globalTensorID = pathID,
                                                   globalNAnimals = readRDS(globalNAnimalsPath),
                                                   featureCandidates = readRDS(featureCandidatesPath)),
                                  preparedDataFinal = fread(prepDataPath))
       } else {
         sim$preparedData <- prepareNNdata(extractedVariables = sim$extractedVariables,
                                           pathX = pathX,
                                           pathID = pathID)
         saveRDS(sim$preparedData$modelPrep$globalNAnimals, file = globalNAnimalsPath)
         saveRDS(sim$preparedData$modelPrep$featureCandidates, file = featureCandidatesPath)
         fwrite(sim$preparedData$preparedDataFinal, file = prepDataPath)
       }
    },
    trainTheModel = {
      globalModelPath <- file.path(outputPath(sim), "global_best_model.pt")
      if (all(file.exists(globalModelPath),
              !P(sim)$rerunTraining)){
        message("All inputs for trainTheModel found. Loading...")
        sim$globalModel <- globalModelPath
      } else {
        sim$globalModel <- trainingNN(preparedData = sim$preparedData$modelPrep, 
                                      batchSize = P(sim)$batchSize,
                                      epoch = P(sim)$epoch,
                                      learningRate =  P(sim)$learningRate,
                                      outputDir = outputPath(sim),
                                      globalModelPath = globalModelPath)
      }
    },
      rankFeatures = {
        prepRankPath <- file.path(outputPath(sim), "featureTable.csv")
        if (all(file.exists(prepRankPath),
                !P(sim)$rerunRanking)){
          message("All inputs for rankFeatures found. Loading...")
          sim$featurePriority <- fread(prepRankPath)
        } else {
          sim$featurePriority <- featureImportance(globalModelPath = sim$globalModel, 
                                                   preparedData = sim$preparedData, 
                                                   batchSize = P(sim)$batchSize)
          fwrite(as.data.table(sim$featurePriority), file = prepRankPath)
          }
    },
    calculateRSS = {

      sim$RSS <- generateRSS(preparedDataFinal = sim$preparedData$preparedDataFinal,
                             globalModel = sim$globalModel,
                             preparedData = sim$preparedData$modelPrep,
                             featurePriority = sim$featurePriority,
                             outDir = outputPath(sim))
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
