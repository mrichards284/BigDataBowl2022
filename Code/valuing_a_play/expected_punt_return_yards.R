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

punts_at_ball_lands_df <- tracking %>%
  mutate(unique_id = paste0(gameId,"_",playId)) %>%
  filter(unique_id %in% punt_plays$unique_id) %>%
  #filter(gameId == 2018090600 & playId == 1989) %>% View()
  filter(event %in% c("punt_received", "punt_land", "fair_catch","punt_muffed", "touchback","out_of_bounds")) %>%
  group_by(unique_id) %>%
  slice_min(frameId) %>%
  ungroup() %>%
  mutate(x = case_when(playDirection == "left"~120-x,
                       TRUE~x)) 


punts_at_ball_received_df_return_wide <- punts_at_ball_lands_df %>%
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
                returner = ifelse(nflId == returnerId,1,0)) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(dist) %>%
  dplyr::filter(displayName != "football" & returner == 0) %>%
  dplyr::group_by(gameId,playId,team_off_def) %>%
  dplyr::arrange(dist) %>%
  dplyr::mutate(player = paste0("player_by_dist_",team_off_def,"_",ifelse(row_number() < 10,paste0("0",row_number()),row_number()))) %>%
  dplyr::ungroup() %>%
  #filter(player != "player_by_dist_01") %>%
  pivot_wider(c(gameId,playId,event,football_x,football_y,yardsToGo,quarter,gameClock,absoluteYardlineNumber,kickLength,kickReturnYardage),values_from = c(x,y,dist),names_from = player) %>%
  left_join(pff_scouting %>% select(gameId,playId,hangTime),by = c("gameId","playId")) %>%
  left_join(punts_at_ball_lands_df %>%
              left_join(punts_at_ball_lands_df %>% dplyr::filter(displayName == "football") %>% dplyr::select(gameId,playId,x,y) %>% dplyr::rename(football_x = x,football_y = y),by = c("gameId","playId")) %>%
              left_join(plays,by = c("gameId","playId")) %>%
              left_join(games,by = c("gameId")) %>%
              dplyr::rowwise() %>%
              dplyr::mutate(dist = sqrt((x - football_x)^2 + (y - football_y)^2),
                            returner = ifelse(nflId == returnerId,1,0)) %>%
              dplyr::filter(returner == 1) %>%
              dplyr::rename(returner_x = x,
                            returner_y = y,
                            returner_dist = dist) %>%
              dplyr::select(gameId,playId,returner_x,returner_y,returner_dist),by = c("gameId","playId"))

# filter out def 11
punts_at_ball_received_df_return_wide <- punts_at_ball_received_df_return_wide %>%
  dplyr::select(-dist_player_by_dist_off_11,-y_player_by_dist_off_11,-x_player_by_dist_off_11,
                -dist_player_by_dist_def_11,-x_player_by_dist_def_11,-y_player_by_dist_def_11) 

punts_at_ball_received_df_return_wide <- punts_at_ball_received_df_return_wide %>%
  filter(!(event %in% c("out_of_bounds","touchback","punt_muffed"))) %>%
  filter(!is.na(kickLength)) %>%
  filter(!is.na(hangTime)) %>%
  filter(!is.na(kickReturnYardage)) %>%
  filter(!is.na(returner_dist))


punts_at_ball_received_df_return_wide %>%
  ggplot(aes(x = kickReturnYardage)) +
  geom_histogram() +
  theme_bw()

punts_at_ball_received_df_return_wide <- punts_at_ball_lands_df_wide %>%
  filter(event == "punt_received") %>%
  left_join(plays %>% dplyr::select(gameId,playId,kickReturnYardage),by = c("gameId","playId")) %>%
  filter(!is.na(kickReturnYardage))

punts_at_ball_received_df_return_wide_sub <- punts_at_ball_received_df_return_wide %>%
  dplyr::select(-gameId,-playId,-event)


library(lightgbm)
library(xgboost)
library(Matrix)
set.seed(500)
nfolds = 5
shuff <- sample(1:nrow(punts_at_ball_received_df_return_wide),size = nrow(punts_at_ball_received_df_return_wide),replace = F)
train <- punts_at_ball_received_df_return_wide_sub[shuff,]
#train_exp_return_yards <- punts_at_ball_received_df_return_wide_sub
#s3saveRDS(x = train_exp_return_yards,
#          object = "train_exp_return_yards.RDS",
#          bucket = "elasticbeanstalk-us-east-1-320699877354")
train_hold <- punts_at_ball_received_df_return_wide[shuff,]
folds <- cut(seq(1,nrow(train)),breaks = nfolds,labels = FALSE)
y_hat <- y_hatv2 <- y_hat_base_rate <- rep(0,nrow(punts_at_ball_received_df_return_wide))
#params = list(
#  learning_rate = 0.1,
#  objective = "multiclass",
#  num_classes = 3,
#  metric = "multi_logloss"
#)
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
  
  mod_lm_sub <- lm(kickReturnYardage~.,data = train_cv)
  mod_lm_subv2 <- lm(kickReturnYardage~.,data = train_cv %>% dplyr::select(kickReturnYardage,football_x,football_y,absoluteYardlineNumber,
                                                                              kickLength,hangTime)) # pre snap only info
  
  
  
  #mod_lgb_sub <- lightgbm(params = params,train_matrix,nrounds = 100)
  
  y_hat_base_rate[indexes] <- mean(train_cv$kickReturnYardage)
  y_hat[indexes] <- predict(mod_lm_sub,newdata = test_cv)
  y_hatv2[indexes] <- predict(mod_lm_subv2,newdata = test_cv)
  #y_hat_lgbm[indexes,] <- matrix(predict(mod_lgb_sub,valm),ncol = 3,byrow = TRUE)
  
  
}

best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0
start = Sys.time()
for (iter in 1:25){
  param <- list(objective = "reg:squarederror",
                #eval.metric = list("mse"),
                metrics = list("mse"),
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
  mdcv <- xgb.cv(data = model.matrix(kickReturnYardage ~ .,data = train),
                 label = as.vector(train$kickReturnYardage), 
                 params = param, 
                 nthread=6, 
                 nfold=cv.nfold, 
                 nrounds=cv.nround,
                 verbose = F, 
                 #early.stop.round=10, 
                 maximize=FALSE)
  
  min_rmse = min(mdcv$evaluation_log[, "test_rmse_mean"])
  min_rmse_index = which.min(mdcv$evaluation_log[, test_rmse_mean])
  
  if (min_rmse < best_rmse) {
    best_rmse = min_rmse
    best_rmse_index = min_rmse_index
    best_seednumber = seed.number
    best_param = param
  }
  
  print(iter)
  
}
stop = Sys.time()

best_rmse

MAE(y_pred = y_hat_base_rate,y_true = train$kickReturnYardage)
MAE(y_pred = y_hat,y_true = train$kickReturnYardage)
RMSE(y_pred = y_hat_base_rate,y_true = train$kickReturnYardage)
RMSE(y_pred = y_hat,y_true = train$kickReturnYardage)
RMSE(y_pred = y_hatv2,y_true = train$kickReturnYardage)
best_rmse # 10.19793


best_param <- list(objective = "reg:squarederror",
              #eval.metric = list("mse"),
              metrics = list("mse"),
              max_depth = 9,
              eta = 0.1379486,
              gamma = 0.1922929, 
              subsample = 0.8081798,
              colsample_bytree = 0.7458915, 
              min_child_weight = 39,
              max_delta_step = 1
)

mod_exp_yards_gained <- xgboost(
  data = model.matrix(kickReturnYardage ~ .,data = train),
  label = as.vector(train$kickReturnYardage), 
  params = best_param, 
  nthread=6, 
  nrounds=1000,
  verbose = F, 
  maximize=FALSE
)

s3saveRDS(x = mod_exp_yards_gained,
          object = "mod_exp_yards_gained.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354",
          multipart = TRUE,
          prefix="nfl-big-data-bowl-2022")
