# https://www.opensourcefootball.com/posts/2020-09-28-nflfastr-ep-wp-and-cp-models/
set.seed(2013) # GoHawks
library(tidyverse)
library(xgboost)

# some helper files are in these
source("https://raw.githubusercontent.com/mrcaseb/nflfastR/master/R/helper_add_nflscrapr_mutations.R")
source("https://raw.githubusercontent.com/mrcaseb/nflfastR/master/R/helper_add_ep_wp.R")
source("https://raw.githubusercontent.com/mrcaseb/nflfastR/master/R/helper_add_cp_cpoe.R")

# from remote
pbp_data <- readRDS(url("https://github.com/guga31bb/nflfastR-data/blob/master/models/cal_data.rds?raw=true"))

# from local
# pbp_data <- readRDS('../../nflfastR-data/models/cal_data.rds')

model_data <- pbp_data %>%
  # in 'R/helper_add_nflscrapr_mutations.R'
  make_model_mutations() %>%
  mutate(
    label = case_when(
      Next_Score_Half == "Touchdown" ~ 0,
      Next_Score_Half == "Opp_Touchdown" ~ 1,
      Next_Score_Half == "Field_Goal" ~ 2,
      Next_Score_Half == "Opp_Field_Goal" ~ 3,
      Next_Score_Half == "Safety" ~ 4,
      Next_Score_Half == "Opp_Safety" ~ 5,
      Next_Score_Half == "No_Score" ~ 6
    ),
    label = as.factor(label),
    # use nflscrapR weights
    Drive_Score_Dist = Drive_Score_Half - drive,
    Drive_Score_Dist_W = (max(Drive_Score_Dist) - Drive_Score_Dist) /
      (max(Drive_Score_Dist) - min(Drive_Score_Dist)),
    ScoreDiff_W = (max(abs(score_differential), na.rm = T) - abs(score_differential)) /
      (max(abs(score_differential), na.rm = T) - min(abs(score_differential), na.rm = T)),
    Total_W = Drive_Score_Dist_W + ScoreDiff_W,
    Total_W_Scaled = (Total_W - min(Total_W, na.rm = T)) /
      (max(Total_W, na.rm = T) - min(Total_W, na.rm = T))
  ) %>%
  filter(
    !is.na(defteam_timeouts_remaining), !is.na(posteam_timeouts_remaining),
    !is.na(yardline_100)
  ) %>%
  select(
    label,
    season,
    half_seconds_remaining,
    yardline_100,
    #home,
    #retractable,
    #dome,
    #outdoors,
    ydstogo,
    #era0, era1, era2, era3, era4,
    down1, down2, down3, down4
    #posteam_timeouts_remaining,
    #defteam_timeouts_remaining,
    #Total_W_Scaled
  )

# idk why this is all necessary for xgb but it is
model_data <- model_data %>%
  mutate(
    label = as.numeric(label),
    label = label - 1
  )


seasons <- unique(model_data$season)


mod_epa <- nnet::multinom(label~.,data = model_data)
s3saveRDS(x = mod_epa,
          object = "mod_epa.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354",
          multipart = TRUE)


predict(newdata = data.frame(season = c(2021,),half_seconds_remaining = 1000, 
                             yardline_100 = 20,ydstogo = 10,down1 = 1,down2 = 0,down3 = 0, down4=0),mod,type = "probs")


nrounds <- 525
params <-
  list(
    booster = "gbtree",
    objective = "multi:softprob",
    eval_metric = c("mlogloss"),
    num_class = 7,
    eta = 0.025,
    gamma = 1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    max_depth = 5,
    min_child_weight = 1
  )


best_param = list()
best_seednumber = 1234
best_mlogloss = Inf
best_mlogloss_index = 0
start = Sys.time()
for (iter in 1:5){
  param <- list(objective = "multi:softprob",
                eval_metric = "mlogloss",
                num_class = 7,
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3),
                gamma = runif(1, 0.0, 0.2), 
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround = 1000
  cv.nfold = 5
  seed.number = sample.int(10000, 1)[[1]]
  set.seed(seed.number)
  mdcv <- xgb.cv(data = model.matrix(label ~ .,data = model_data),
                 label = as.vector(model_data$label), 
                 params = param, 
                 nthread=6, 
                 nfold=cv.nfold, 
                 nrounds=cv.nround,
                 verbose = F, 
                 #early.stop.round=10, 
                 maximize=FALSE)
  
  min_mlogloss = min(mdcv$evaluation_log[, "test_mlogloss_mean"])
  min_mlogloss_index = which.min(mdcv$evaluation_log[, test_mlogloss_mean])
  
  if (min_mlogloss < best_mlogloss) {
    best_mlogloss = min_mlogloss 
    best_mlogloss_index = min_mlogloss_index
    best_seednumber = seed.number
    best_param = param
  }
  
  print(iter)
  
}
stop = Sys.time()
