#' Train the global ranking network (all years >= startYear) on a SEEDED random 80/20 split of
#' strata. It is used only to rank covariates; its quality is irrelevant to the experiment.
#' The held-out 20% is saved and used for the importance calculation.
#' @return path to the best-epoch weights (a config file with nIn / nAnimals is written next to it)
trainingNN <- function(preparedData, batchSize, learningRate, epochs, validationSplit = 0.2,
                       outputDir, globalModelPath, seed = 20261007L, device = "cpu",
                       earlyStopPatience = 5L, registryPath = NULL) {
  message("Global Model: loading tensors from disk...")
  x <- torch_load(preparedData$globalTensorX)
  id <- torch_load(preparedData$globalTensorID)
  nAnimals <- preparedData$globalNAnimals
  n <- x$size(1)
  registerSeeds(registryPath, "caribouNN_Global::trainingNN", "split+init", seed,
                "global ranking model: 80/20 stratum split (R) and torch init/batch order")
  valIdx <- sort(withSeed(seed, sample.int(n, floor(n * validationSplit))))
  trIdx <- setdiff(seq_len(n), valIdx)
  vi <- torch_tensor(valIdx, dtype = torch_long()); ti <- torch_tensor(trIdx, dtype = torch_long())
  fit <- fitStratumNet(x$index_select(1, ti)$to(device = device), id$index_select(1, ti)$to(device = device),
                       x$index_select(1, vi)$to(device = device), id$index_select(1, vi)$to(device = device),
                       nAnimals = nAnimals, lr = learningRate, epochs = epochs, batchSize = batchSize,
                       seed = seed, device = device, earlyStopPatience = earlyStopPatience)
  checkWeightsRoundTrip(fit$bestState, globalModelPath, nIn = x$size(3), nAnimals = nAnimals,
                        xVal = x$index_select(1, vi)$to(device = device), idVal = id$index_select(1, vi)$to(device = device),
                        trainIdsSeen = fit$trainIdsSeen, expectedValLoss = fit$bestValLoss, device = device)
  saveRDS(list(nIn = x$size(3), nAnimals = nAnimals, valIdx = valIdx, seed = seed, history = fit$history,
               bestEpoch = fit$bestEpoch, bestValLoss = fit$bestValLoss, trainIdsSeen = fit$trainIdsSeen),
          file.path(outputDir, "global_model_info.rds"))
  message(sprintf("Global model done: best epoch %d, validation loss %.4f (random guessing: %.4f)",
                  fit$bestEpoch, fit$bestValLoss, log(x$size(2))))
  globalModelPath
}
