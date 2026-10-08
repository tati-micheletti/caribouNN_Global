#' Recode "no recorded disturbance" (value = year - backgroundYr, a calendar-year counter that
#' leaks the year into the covariate) to a fixed cap, as in the published iSSA analysis
#' (no recorded fire/harvest = 40 years; real values above 40 are also set to 40).
#' Applied to the start and end covariates; identical for every regime and model.
#' @return the recoded table (by reference) with attribute "recodeReport"
recodeDisturbanceClock <- function(dt, backgroundYr = 1900L, capYears = 40) {
  rep <- list()
  for (v in c("timeSinceFire_start", "timeSinceFire_end", "timeSinceHarvest_start", "timeSinceHarvest_end")) {
    if (!v %in% names(dt)) next
    x <- dt[[v]]
    sentinel <- !is.na(x) & x == (dt$year - backgroundYr)
    real <- !is.na(x) & !sentinel
    capped <- real & x > capYears
    newx <- ifelse(sentinel, capYears, pmin(x, capYears))
    set(dt, j = v, value = newx)
    rep[[v]] <- data.table(variable = v, nStrataRows = length(x), sentinelShare = mean(sentinel),
                           realShare = mean(real), cappedShare = mean(capped), nNA = sum(is.na(x)))
  }
  setattr(dt, "recodeReport", rbindlist(rep))
  dt
}

#' @param extractedVariables table with 11 rows per stratum
#' @param startYear first year used (earlier years are discarded before anything else)
#' @param zClip standardised covariates are clipped to +/- zClip (same rule as in the experiment)
prepareNNdata <- function(extractedVariables, pathX, pathID, startYear = 2013L,
                          backgroundYr = 1900L, capYears = 40, zClip = 10) {
  message("Preparing Data...")
  dt <- copy(extractedVariables)
  nStart <- uniqueN(dt$indiv_step_id)
  dt <- dt[year >= startYear]
  message(sprintf("Years >= %d: %d of %d strata kept.", startYear, uniqueN(dt$indiv_step_id), nStart))

  # Basic cleaning: strata with a missing step length / turn angle / id are removed WHOLE below
  dt <- na.omit(dt, cols = c("sl_", "ta_", "indiv_step_id", "id"))

  # FILTER VALID STRATA: correct number of rows AND exactly one observed step
  originalTotalStrata <- uniqueN(dt$indiv_step_id)
  counts <- dt[, .N, by = indiv_step_id]
  steps <- as.numeric(names(sort(table(counts$N), decreasing = TRUE)[1]))
  valid_stats <- dt[, .(n_rows = .N, n_case = sum(case_)), by = indiv_step_id]
  valid_ids <- valid_stats[n_rows == steps & n_case == 1, indiv_step_id]
  dt <- dt[indiv_step_id %in% valid_ids]
  setorder(dt, indiv_step_id, -case_)   # observed step first
  message("Kept ", length(valid_ids), " valid strata (", round(100 * length(valid_ids) / originalTotalStrata, 1),
          "% of ", originalTotalStrata, "). The rest lose >= 1 row (e.g. first step of a burst has no turn angle).")

  # Calendar-year counters -> fixed cap (see recodeDisturbanceClock)
  dt <- recodeDisturbanceClock(dt, backgroundYr = backgroundYr, capYears = capYears)
  recodeReport <- attr(dt, "recodeReport")
  print(recodeReport)

  # Burst key (animal + burst) for later clustering / blocking
  if ("burst_" %in% names(dt)) dt[, burstKey := paste(id, burst_, sep = "_")]

  # FEATURE ENGINEERING
  dt[, prop_veg_start := rowSums(.SD, na.rm = TRUE), .SDcols = intersect(names(dt), c("prop_herbs_start", "prop_shrub_start", "prop_bryoids_start"))]
  dt[, prop_veg_end := rowSums(.SD, na.rm = TRUE), .SDcols = intersect(names(dt), c("prop_herbs_end", "prop_shrub_end", "prop_bryoids_end"))]
  dt[, prop_wets_start := prop_wetland_start]
  dt[, prop_wets_end   := prop_wetland_end]
  dt[, prop_needleleaf_start := rowSums(.SD, na.rm = TRUE), .SDcols = intersect(names(dt), c("prop_needleleaf_start", "prop_wet_treed_start"))]
  dt[, prop_needleleaf_end := rowSums(.SD, na.rm = TRUE), .SDcols = intersect(names(dt), c("prop_needleleaf_end", "prop_wet_treed_end"))]
  dt[, prop_mixedforest_start := rowSums(.SD, na.rm = TRUE), .SDcols = intersect(names(dt), c("prop_mixed_start", "prop_deciduous_start"))]
  dt[, prop_mixedforest_end := rowSums(.SD, na.rm = TRUE), .SDcols = intersect(names(dt), c("prop_mixed_end", "prop_deciduous_end"))]

  dt[, logSl := log(sl_ + 1)]
  dt[, cosTa := cos(ta_)]
  dt[, sinTa := sin(ta_)]

  varsToLog <- c("timeSinceFire", "timeSinceHarvest", "distpaved", "distunpaved", "distpolys")
  for (v in varsToLog) {
    if (paste0(v, "_start") %in% names(dt)) dt[, (paste0(v, "_startLog")) := log(get(paste0(v, "_start")) + 1)]
    if (paste0(v, "_end") %in% names(dt))   dt[, (paste0(v, "_endLog")) := log(get(paste0(v, "_end")) + 1)]
  }

  interVars <- c("prop_needleleaf_start", "prop_mixedforest_start", "prop_veg_start", "prop_wets_start",
                 "timeSinceFire_startLog", "timeSinceHarvest_startLog",
                 "distpaved_startLog", "distunpaved_startLog", "distpolys_startLog")
  vec_logSl <- dt$logSl
  for (var in interVars) if (var %in% names(dt)) set(dt, j = paste0("inter_logSl_x_", var), value = vec_logSl * dt[[var]])

  candidates <- c(
    "logSl", "cosTa", "sinTa",
    "prop_needleleaf_end", "prop_mixedforest_end", "prop_veg_end", "prop_wets_end",
    "timeSinceFire_endLog", "timeSinceHarvest_endLog",
    "distpaved_endLog", "distunpaved_endLog", "distpolys_endLog",
    "prop_needleleaf_start", "prop_mixedforest_start", "prop_veg_start", "prop_wets_start",
    "timeSinceFire_startLog", "timeSinceHarvest_startLog",
    "distpaved_startLog", "distunpaved_startLog", "distpolys_startLog",
    grep("^inter_", names(dt), value = TRUE)
  )
  featureCandidates <- intersect(candidates, names(dt))

  # NA policy: NEVER silent. Report, then fail if any candidate contains NA.
  nNA <- vapply(featureCandidates, function(f) sum(is.na(dt[[f]])), numeric(1))
  if (any(nNA > 0)) {
    print(nNA[nNA > 0])
    stop("NA in candidate covariates (see above). Decide explicitly how to treat them; no silent imputation.")
  }

  # Scaling is used ONLY for the ranking model's tensors; preparedDataFinal keeps the UNSCALED
  # engineered covariates, and every experiment model standardises with its own training statistics.
  scaleInfo <- lapply(featureCandidates, function(f) list(mean = mean(dt[[f]]), sd = sd(dt[[f]])))
  names(scaleInfo) <- featureCandidates
  constant <- names(scaleInfo)[vapply(scaleInfo, function(s) !is.finite(s$sd) || s$sd == 0, logical(1))]
  if (length(constant)) message("Constant covariates (no information, set to 0 for ranking): ", paste(constant, collapse = ", "))
  scaledCol <- function(f) {
    s <- scaleInfo[[f]]
    if (!is.finite(s$sd) || s$sd == 0) return(rep(0, nrow(dt)))
    pmax(pmin((dt[[f]] - s$mean) / s$sd, zClip), -zClip)
  }

  dt[, idIndex := as.numeric(as.factor(id))]
  dtglobalNAnimals <- max(dt$idIndex)

  message("Shaping and saving tensors to disk...")
  n <- nrow(dt) / steps
  X <- torch_empty(n, steps, length(featureCandidates), dtype = torch_float())
  for (k in seq_along(featureCandidates))
    X[, , k] <- torch_tensor(t(matrix(scaledCol(featureCandidates[k]), nrow = steps)), dtype = torch_float())
  torch_save(X, pathX)
  torch_save(torch_tensor(dt[case_ == TRUE, idIndex], dtype = torch_long()), pathID)

  list(
    modelPrep = list(globalTensorX = pathX, globalTensorID = pathID, globalNAnimals = dtglobalNAnimals,
                     featureCandidates = featureCandidates, steps = steps),
    preparedDataFinal = dt,
    recodeReport = recodeReport
  )
}
