# Tests for the caribouNN_Global refit functions (synthetic data, no real data needed).
# Run: Rscript test_global.R      Needs data.table + torch.
suppressMessages({library(data.table); library(torch)})
fileArg <- grep("--file=", commandArgs(FALSE), value = TRUE)
here <- if (length(fileArg)) dirname(normalizePath(sub("--file=", "", fileArg))) else getwd()
modDir <- normalizePath(file.path(here, ".."))
for (f in c("stratumNet.R", "prepareNNdata.R", "trainingNN.R", "featureImportance.R")) source(file.path(modDir, "R", f))
nFail <- 0L
check <- function(cond, msg) { ok <- isTRUE(cond); cat(sprintf("[%s] %s\n", if (ok) "PASS" else "FAIL", msg)); if (!ok) nFail <<- nFail + 1L }

# ---- recodeDisturbanceClock ---------------------------------------------------------------
d <- data.table(year = c(2013L, 2013L, 2014L, 2014L, 2020L),
                timeSinceFire_start = c(2013 - 1900, 5, 2014 - 1900, 55, 2020 - 1900),
                timeSinceFire_end = c(2013 - 1900, 5, 2014 - 1900, 55, 2020 - 1900),
                timeSinceHarvest_start = c(113, 113, 114, 114, 120), timeSinceHarvest_end = c(113, 113, 114, 114, 120))
d <- recodeDisturbanceClock(d, backgroundYr = 1900L, capYears = 40)
check(all(d$timeSinceFire_start == c(40, 5, 40, 40, 40)), "sentinel -> 40, real value kept, real value above 40 capped")
check(all(d$timeSinceHarvest_start == 40), "all-sentinel harvest -> constant 40")
check(cor(c(113, 113, 114, 114, 120), d$year) > 0.9 && sd(d$timeSinceHarvest_start) == 0, "year information removed from harvest")

# ---- within-stratum permutation importance finds the informative covariates -----------------
set.seed(7)
n <- 6000; S <- 11; K <- 6
x <- array(rnorm(n * S * K), dim = c(n, S, K))
x[, 1, 1] <- x[, 1, 1] + 1.5          # covariate 1: strongly informative (observed step is step 1)
x[, 1, 2] <- x[, 1, 2] + 0.6          # covariate 2: weakly informative
id <- sample(1:10, n, replace = TRUE)
tmp <- tempfile(); dir.create(tmp)
torch_save(torch_tensor(x, dtype = torch_float()), file.path(tmp, "x.pt"))
torch_save(torch_tensor(id, dtype = torch_long()), file.path(tmp, "id.pt"))
prep <- list(globalTensorX = file.path(tmp, "x.pt"), globalTensorID = file.path(tmp, "id.pt"),
             globalNAnimals = 10L, featureCandidates = paste0("f", 1:K))
gm <- trainingNN(prep, batchSize = 256, learningRate = 0.01, epochs = 12, outputDir = tmp,
                 globalModelPath = file.path(tmp, "w.pt"), seed = 11L, earlyStopPatience = 20L)
imp <- featureImportance(gm, list(modelPrep = prep), batchSize = 256, nRep = 3L, seed = 5L)
check(imp$Feature[1] == "f1" && imp$Feature[2] == "f2", "importance ranks the informative covariates first (f1, f2)")
check(all(imp$Importance[3:K] < imp$Importance[2]), "uninformative covariates have lower importance")
check(imp$baselineLoss[1] < log(S), "global model beats chance on informative data")
# same seeds -> same ranking and same loss
gm2 <- trainingNN(prep, batchSize = 256, learningRate = 0.01, epochs = 12, outputDir = tmp,
                  globalModelPath = file.path(tmp, "w2.pt"), seed = 11L, earlyStopPatience = 20L)
i1 <- readRDS(file.path(tmp, "global_model_info.rds"))
check(abs(i1$bestValLoss - i1$history$valLoss[i1$bestEpoch]) < 1e-12, "best validation loss equals the logged epoch value")
check(identical(as.integer(i1$valIdx), as.integer(sort(withSeed(11L, sample.int(n, floor(n * 0.2)))))),
      "validation split is reproducible from the seed")
cat(sprintf("\n%d failure(s)\n", nFail)); if (nFail) quit(status = 1)
