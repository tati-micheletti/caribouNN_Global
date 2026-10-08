#' Permutation importance of each candidate covariate on the HELD-OUT strata of the global model.
#'
#' The model scores the 11 candidate steps of a stratum against each other, so a covariate can
#' only help through differences BETWEEN candidates of the same stratum. The ranking therefore
#' uses WITHIN-STRATUM permutation: the covariate's 11 values are shuffled among the 11
#' candidates of each stratum (averaged over `nRep` repeats). The previous method (moving whole
#' blocks of 11 values between strata) is also returned (`impBlocksAcrossStrata`) for reference.
#' Features are ordered by the within-stratum importance.
featureImportance <- function(globalModelPath, preparedData, batchSize, nRep = 5L,
                              seed = 20261008L, registryPath = NULL, chunk = 20000L) {
  message("Global Model: calculating within-stratum permutation importance...")
  registerSeeds(registryPath, "caribouNN_Global::featureImportance", "permutations", seed,
                "torch seed for within-stratum and block permutations")
  torch::torch_manual_seed(seed)
  prep <- preparedData$modelPrep
  x <- torch_load(prep$globalTensorX); id <- torch_load(prep$globalTensorID)
  info <- readRDS(file.path(dirname(globalModelPath), "global_model_info.rds"))
  cand <- prep$featureCandidates
  net <- makeStratumNet()(nIn = info$nIn, nAnimals = info$nAnimals)
  net$load_state_dict(torch_load(globalModelPath))
  net$eval()
  vi <- torch_tensor(info$valIdx, dtype = torch_long())
  xv <- x$index_select(1, vi); idv <- id$index_select(1, vi)
  n <- xv$size(1); S <- xv$size(2)
  seen <- info$trainIdsSeen

  lossOf <- function(xx) mean(scoreStrata(net, xx, idv, seen, info$nAnimals, chunk = chunk)$loss)
  base <- lossOf(xv)
  message("Baseline held-out loss: ", round(base, 4), " (random guessing ", round(log(S), 4), ")")

  withinShuffle <- function(col) {              # col: [n, S]
    ord <- torch::torch_argsort(torch::torch_rand(n, S), dim = 2)
    col$gather(2, ord)
  }
  res <- vector("list", length(cand))
  for (k in seq_along(cand)) {
    impW <- numeric(nRep); impB <- numeric(nRep)
    for (r in seq_len(nRep)) {
      xw <- xv$clone(); xw[, , k] <- withinShuffle(xv[, , k]); impW[r] <- lossOf(xw) - base
      xb <- xv$clone(); xb[, , k] <- xv[, , k]$index_select(1, torch::torch_randperm(n) + 1L); impB[r] <- lossOf(xb) - base
    }
    res[[k]] <- data.frame(Feature = cand[k], Importance = mean(impW), ImportanceSD = stats::sd(impW),
                           impBlocksAcrossStrata = mean(impB))
    cat(".")
  }
  imp <- do.call(rbind, res)
  imp$baselineLoss <- base
  imp <- imp[order(-imp$Importance), ]
  rownames(imp) <- NULL
  imp$rankWithin <- seq_len(nrow(imp))
  imp$rankBlocks <- rank(-imp$impBlocksAcrossStrata)
  print(utils::head(imp, 12))
  message("\nGlobal feature ranking complete (Spearman within vs blocks: ",
          round(stats::cor(imp$rankWithin, imp$rankBlocks, method = "spearman"), 3), ").")
  imp
}
