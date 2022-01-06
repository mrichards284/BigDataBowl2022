# Packages
install.packages("nflfastR")
library(nflfastR)
library(aws.s3)
library(tidyverse)

# Function to convert s3 object to csv and read into r
s3_to_csv <- function(s3_path) {
  
  usercsvobj <- aws.s3::get_object(s3_path)
  csvcharobj <- rawToChar(usercsvobj)
  con <- textConnection(csvcharobj) 
  data <- read.csv(con) 
  close(con) 
  return(data)
  
}


# Pull all objects in s3 bucket
get_bucket_df('elasticbeanstalk-us-east-1-320699877354', prefix="nfl-big-data-bowl-2022")

# Example reading in the games data frame
tracking2018 <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/tracking2018.csv')
tracking2019 <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/tracking2019.csv')
tracking2020 <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/tracking2020.csv')
tracking <- rbind(tracking2018,tracking2019,tracking2020)
rm(tracking2018,tracking2019,tracking2020)

plays <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/plays.csv')
games <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/games.csv')
pff_scouting <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/PFFScoutingData.csv')


punt_plays <- plays %>% filter(specialTeamsPlayType == "Punt") %>%  mutate(unique_id = paste0(gameId,"_",playId))


# 
punts_at_ball_lands_df <- tracking %>%
  dplyr::mutate(unique_id = paste0(gameId,"_",playId)) %>%
  dplyr::filter(unique_id %in% punt_plays$unique_id) %>%
  dplyr::filter(event %in% c("punt_received", "punt_land", "fair_catch","punt_muffed", "touchback","out_of_bounds")) %>%
  dplyr::group_by(unique_id) %>%
  dplyr::slice_min(frameId) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(x = case_when(playDirection == "left"~120-x,
                       TRUE~x),
         y = case_when(playDirection == "left"~53-y,
                       TRUE~y)) 

plays <- plays %>%
  mutate(absoluteYardlineNumber = as.double(absoluteYardlineNumber))

punts_at_ball_lands_df_wide <- punts_at_ball_lands_df %>%
  left_join(punts_at_ball_lands_df %>% filter(displayName == "football") %>% dplyr::select(gameId,playId,x,y) %>% dplyr::rename(football_x = x,football_y = y),by = c("gameId","playId")) %>%
  left_join(plays,by = c("gameId","playId")) %>%
  left_join(games,by = c("gameId")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(dist = sqrt((x - football_x)^2 + (y - football_y)^2),
                team_off_def = case_when(homeTeamAbbr==possessionTeam & team == "home"~"off",
                                         homeTeamAbbr==possessionTeam & team == "away"~"def",
                                         visitorTeamAbbr==possessionTeam & team == "away"~"off",
                                         visitorTeamAbbr==possessionTeam & team == "home"~"def",
                                         TRUE~"None"),
                returner = ifelse(nflId == returnerId,1,0),
                absoluteYardlineNumber = case_when(playDirection == "left"~120-absoluteYardlineNumber,
                                                   TRUE~absoluteYardlineNumber)) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(dist) %>%
  dplyr::filter(displayName != "football") %>%
  dplyr::group_by(gameId,playId,team_off_def) %>%
  dplyr::arrange(dist) %>%
  dplyr::mutate(player = paste0("player_by_dist_",team_off_def,"_",ifelse(row_number() < 10,paste0("0",row_number()),row_number()))) %>%
  dplyr::ungroup() %>%
  #filter(player != "player_by_dist_01") %>%
  pivot_wider(c(gameId,playId,event,football_x,football_y,absoluteYardlineNumber,kickLength),values_from = c(x,y,dist),names_from = player) %>%
  left_join(pff_scouting %>% dplyr::select(gameId,playId,hangTime),by = c("gameId","playId")) #%>%
  #left_join(punts_at_ball_lands_df %>%
  #            left_join(punts_at_ball_lands_df %>% dplyr::filter(displayName == "football") %>% dplyr::select(gameId,playId,x,y) %>% dplyr::rename(football_x = x,football_y = y),by = c("gameId","playId")) %>%
  #            left_join(plays,by = c("gameId","playId")) %>%
  #            left_join(games,by = c("gameId")) %>%
  #            dplyr::rowwise() %>%
  #            dplyr::mutate(dist = sqrt((x - football_x)^2 + (y - football_y)^2),
  #                          returner = ifelse(nflId == returnerId,1,0)) %>%
  #            dplyr::filter(returner == 1) %>%
  #            dplyr::rename(returner_x = x,
  #                   returner_y = y,
  #                   returner_dist = dist) %>%
  #            dplyr::select(gameId,playId,returner_x,returner_y,returner_dist),by = c("gameId","playId"))

summary(punts_at_ball_lands_df_wide)

# filter out def 11
punts_at_ball_lands_df_wide <- punts_at_ball_lands_df_wide %>%
  dplyr::select(-dist_player_by_dist_off_11,-y_player_by_dist_off_11,-x_player_by_dist_off_11,
                -dist_player_by_dist_def_11,-x_player_by_dist_def_11,-y_player_by_dist_def_11) 

# display categories of each. Group together punt_land, out_of_bounds and touchback
table(punts_at_ball_lands_df_wide$event)

#punts_at_ball_lands_df_wide %>%
#  mutate(event = case_when(event %in% c("out_of_bounds","punt_land","touchback"))) %>%

punts_at_ball_lands_df_wide <- punts_at_ball_lands_df_wide %>%
  filter(!(event %in% c("out_of_bounds","touchback","punt_muffed"))) %>%
  filter(!is.na(kickLength)) %>%
  filter(!is.na(hangTime))

punts_at_ball_lands_df_wide_sub <- punts_at_ball_lands_df_wide %>%
  dplyr::select(-gameId,-playId)
  

library(lightgbm)
library(Matrix)
set.seed(100)
nfolds = 5
shuff <- sample(1:nrow(punts_at_ball_lands_df_wide),size = nrow(punts_at_ball_lands_df_wide),replace = F)
train <- punts_at_ball_lands_df_wide_sub[shuff,]
#train_punt_outcome <- punts_at_ball_lands_df_wide_sub
#s3saveRDS(x = train_punt_outcome,
#          object = "train_punt_outcome.RDS",
#          bucket = "elasticbeanstalk-us-east-1-320699877354")

train_hold <- punts_at_ball_lands_df_wide[shuff,]
folds <- cut(seq(1,nrow(train)),breaks = nfolds,labels = FALSE)
y_hat_multi <- y_hat_multiv2 <- y_hat_lgbm <- y_hat_base_rate <- matrix(rep(0,nrow(punts_at_ball_lands_df_wide)*3),ncol = 3)
y_hat_mutli_class <- y_hat_mutli_classv2 <- y_hat_lgbm_class <- rep("0",nrow(punts_at_ball_lands_df_wide))
params = list(
  learning_rate = 0.1,
  objective = "multiclass",
  num_classes = 3,
  metric = "multi_logloss"
)
for (cv in 1:nfolds){
  
  indexes <- which(folds == cv,arr.ind = T)
  train_cv <- train[-indexes,]
  test_cv <- train[indexes,]
  #trainm = sparse.model.matrix(event~.,data = train_cv)
  #train_label = train_cv[,"event"] %>% mutate(event = case_when(event == "fair_catch"~0,
  #                                                              event == "punt_land"~1,
  #                                                              event == "punt_received"~2)) %>% t %>% c
  #valm = sparse.model.matrix(event~.,data = test_cv)
  #val_label = test_cv[,"event"] %>% mutate(event = case_when(event == "fair_catch"~0,
  #                                                           event == "punt_land"~1,
  #                                                           event == "punt_received"~2))  %>% t %>% c
  #train_matrix = lgb.Dataset(data = as.matrix(trainm),label = train_label)
  #val_matrix = lgb.Dataset(data = as.matrix(valm),label = val_label)
  
  mod_multi_sub <- nnet::multinom(event~.,data = train_cv,maxit = 1000)
  mod_multi_subv2 <- nnet::multinom(event~.,data = train_cv %>% dplyr::select(event,football_x,football_y,absoluteYardlineNumber,
                                                                     kickLength,hangTime),maxit = 1000) # pre snap only info
  
  
  #mod_lgb_sub <- lightgbm(params = params,train_matrix,nrounds = 100)
  
  y_hat_base_rate[indexes,] <- table(train_cv$event)/nrow(train_cv) %>% t %>% c
  y_hat_multi[indexes,] <- predict(mod_multi_sub,newdata = test_cv,type = "probs")
  y_hat_multiv2[indexes,] <- predict(mod_multi_subv2,newdata = test_cv,type = "probs")
  
  #y_hat_lgbm[indexes,] <- matrix(predict(mod_lgb_sub,valm),ncol = 3,byrow = TRUE)
  
  
  y_hat_mutli_class[indexes] <- as.character(predict(mod_multi_sub,newdata = test_cv))
  y_hat_mutli_classv2[indexes] <- as.character(predict(mod_multi_subv2,newdata = test_cv))
  
  #y_hat_lgbm_class[indexes] <- c("fair_catch","punt_land","punt_received")[max.col(y_hat_lgbm[indexes,],ties.method = "first")]
  
}

library(MLmetrics)
event_num <- train %>%
  mutate(event_num = case_when(event == "fair_catch"~1,
                               event == "punt_land"~2,
                               event == "punt_received"~3,
                               TRUE~999)) %>%
  pull(event_num)

actual_mat <- matrix(ifelse(rep(event_num,each = 3) == rep(1:3,length(rep(event_num))),1,0),ncol = 3,byrow = T)
MultiLogLoss(y_pred = y_hat_base_rate,y_true = actual_mat)
MultiLogLoss(y_pred = y_hat_multiv2,y_true = actual_mat)
MultiLogLoss(y_pred = y_hat_multi,y_true = actual_mat)
#MultiLogLoss(y_pred = y_hat_lgbm,y_true = actual_mat)
Accuracy(y_pred = rep("punt_received",nrow(train_hold)),y_true = train_hold$event)
Accuracy(y_pred = y_hat_mutli_class,y_true = train_hold$event)
#Accuracy(y_pred = y_hat_lgbm_class,y_true = train_hold$event)


train <- train %>% 
  mutate(event = case_when(event == "fair_catch"~0,
                               event == "punt_land"~1,
                               event == "punt_received"~2,
                               TRUE~999))
best_param = list()
best_seednumber = 1234
best_mlogloss = Inf
best_mlogloss_index = 0
start = Sys.time()
for (iter in 6:12){
  param <- list(objective = "multi:softprob",
                eval_metric = "mlogloss",
                num_class = 3,
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
  mdcv <- xgb.cv(data = model.matrix(event ~ .,data = train),
                 label = as.vector(train$event), 
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

best_param

mod_boost_punt_outcome <- xgboost(data = model.matrix(event ~ .,data = train),
       label = as.vector(train$event), 
       params = best_param, 
       nrounds = 1000,
       nthread=6, 
       verbose = F, 
       maximize=FALSE)

s3saveRDS(x = mod_boost_punt_outcome,
          object = "mod_boost_punt_outcome.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354",
          multipart = TRUE,
          prefix="nfl-big-data-bowl-2022")






