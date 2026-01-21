trainingNN <- function(preparedData, batchSize, learningRate, epochs){
  
  globalTensorX <- preparedData$globalTensorX
  globalTensorID <- preparedData$globalTensorID
  globalNAnimals <- preparedData$globalNAnimals
  featureCandidates <- preparedData$featureCandidates
  
  message("Global Model: Training...")
  
  # 1. Dataset and Dataloader
  ds <- dataset(
    "ds", initialize = function(x, id) {
      self$x <- x
      self$id <- id
      },
    .getitem = function(i) {
      list(list(x=self$x[i,,], id=self$id[i]), torch_tensor(1, dtype=torch_long())) 
      },
    .length = function() {
      self$x$size(1)
      }
  )
  dl <- dataloader(ds(globalTensorX, globalTensorID), 
                   batch_size = batchSize, 
                   shuffle = TRUE)
  
  # 2. Model
  Net <- nn_module(
    "Net",
    initialize = function(nIn, nAnimals) {
      self$idEmb <- nn_embedding(nAnimals + 1, 8)
      self$fc1 <- nn_linear(nIn + 8, 128)
      self$fc2 <- nn_linear(128, 64)
      self$out <- nn_linear(64, 1)
      self$act <- nn_selu()
    },
    forward = function(input) {
      x <- input$x
      id <- input$id
      emb <- self$idEmb(id)$unsqueeze(2)$expand(c(-1,x$shape[2],-1))
      torch_cat(list(x,emb),3) %>% 
        self$fc1() %>% 
        self$act() %>% 
        self$fc2() %>% 
        self$act() %>% 
        self$out() %>% 
        squeeze(3)
    }
  )
  
  # 3. Fit
  fitted <- Net %>%
    setup(loss = nn_cross_entropy_loss(), optimizer = optim_adam) %>%
    set_hparams(nIn = length(featureCandidates), nAnimals = globalNAnimals) %>%
    set_opt_hparams(lr = learningRate) %>%
    fit(dl, epochs = epochs, verbose = TRUE)
  
  message("\nGlobal Model Fitting Complete.")
  return(fitted)
}
