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

plays <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/plays.csv')
games <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/games.csv')
pff_scouting <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/PFFScoutingData.csv')


punt_plays <- plays %>% filter(specialTeamsPlayType == "Punt") %>%  mutate(unique_id = paste0(gameId,"_",playId))

punts_at_ball_lands_only_df <- tracking %>%
  mutate(unique_id = paste0(gameId,"_",playId)) %>%
  filter(unique_id %in% punt_plays$unique_id) %>%
  #filter(gameId == 2018090600 & playId == 1989) %>% View()
  filter(event %in% c("punt_land") &
           displayName == "football") %>%
  group_by(unique_id) %>%
  slice_min(frameId) %>%
  ungroup() %>%
  mutate(x = case_when(playDirection == "left"~120-x,
                       TRUE~x)) 

punts_at_final_punt_loc_df <- tracking %>%
  mutate(unique_id = paste0(gameId,"_",playId)) %>%
  filter(unique_id %in% punt_plays$unique_id) %>%
  #filter(gameId == 2018090600 & playId == 1989) %>% View()
  filter(event %in% c("punt_downed","touchback","out_of_bounds") &
           displayName == "football") %>%
  group_by(unique_id) %>%
  slice_min(frameId) %>%
  ungroup() %>%
  mutate(x = case_when(playDirection == "left"~120-x,
                       TRUE~x)) 


punt_land_final_df_chng <- punts_at_ball_lands_only_df %>%
  dplyr::select(unique_id,x,y,event) %>%
  left_join(punts_at_final_punt_loc_df %>%
              dplyr::rename(final_x = x,
                            final_y = y,
                            final_event = event) %>%
              dplyr::select(unique_id,final_x,final_y,final_event),by = c("unique_id")) 

punt_land_final_df_chng %>%
  dplyr::mutate(chg_x = final_x - x) %>%
  ggplot(aes(x = chg_x)) +
  geom_histogram(binwidth = 4) +
  theme_bw()

sample_dist_x <- rnorm(1000,mean = 5 , sd = 5)
punt_land_final_df_chng %>%
  dplyr::mutate(chg_x = final_x - x) %>%
  ggplot(aes(x = chg_x)) +
  geom_density() +
  geom_density(data = data.frame(x = sample_dist_x),aes(x = x),color = 'red') +
  theme_bw()


punt_land_final_df_chng %>%
  dplyr::mutate(chg_y = final_y - y) %>%
  ggplot(aes(x = chg_y)) +
  geom_histogram(binwidth = 4) +
  theme_bw()



sample_dist_y <- rnorm(1000,mean = 0 , sd = 5)
punt_land_final_df_chng %>%
  dplyr::mutate(chg_y = final_y - y) %>%
  ggplot(aes(x = chg_y)) +
  geom_density() +
  geom_density(data = data.frame(x = sample_dist_y),aes(x = x),color = 'red') +
  theme_bw()

punt_land_final_df_chng %>%
  dplyr::mutate(chg_x = final_x - x,
                chg_y = final_y - y) %>%
  summary()


punt_land_final_df_chng %>%
  dplyr::mutate(chg_x = final_x - x,
                chg_y = final_y - y) %>%
  ggplot(aes(x = chg_x,y = chg_y)) +
  geom_point() +
  theme_bw()

punt_land_roll_yards <- tracking %>%
  mutate(unique_id = paste0(gameId,"_",playId)) %>%
  filter(unique_id %in% punt_plays$unique_id) %>%
  #filter(gameId == 2018090600 & playId == 1989) %>% View()
  filter(event %in% c("punt_downed","touchback","out_of_bounds") &
           displayName == "football") %>%
  group_by(unique_id) %>%
  slice_min(frameId) %>%
  ungroup() %>%
  dplyr::mutate(x = case_when(playDirection == "left"~120-x,
                              TRUE~x),
                y = case_when(playDirection == "left"~53-y,
                              TRUE~y)) %>%
  left_join(punts_at_ball_lands_df_wide %>%
              filter(event == "punt_land") %>%
              dplyr::select(gameId,playId,football_x,football_y),by = c("gameId","playId")) %>%
  dplyr::mutate(roll_yards = (x - football_x)) %>%
  dplyr::select(gameId,playId,roll_yards)


punts_at_ball_lands_df_roll_yards_wide <- punts_at_ball_lands_df_wide %>%
  filter(event == "punt_land") %>%
  left_join(punt_land_roll_yards,by = c("gameId","playId")) %>%
  filter(!is.na(roll_yards)) %>%
  filter(!is.na(hangTime)) %>%
  filter(roll_yards > -20)

punts_at_ball_received_df_roll_yards_wide_sub <- punts_at_ball_lands_df_roll_yards_wide %>%
  dplyr::select(-gameId,-playId,-event) 


library(lightgbm)
library(xgboost)
library(Matrix)
set.seed(100)
nfolds = 5
shuff <- sample(1:nrow(punts_at_ball_lands_df_roll_yards_wide),size = nrow(punts_at_ball_lands_df_roll_yards_wide),replace = F)
train <- punts_at_ball_received_df_roll_yards_wide_sub[shuff,]
#train_exp_roll_yards <- punts_at_ball_received_df_roll_yards_wide_sub
#s3saveRDS(x = train_exp_roll_yards,
#          object = "train_exp_roll_yards.RDS",
#          bucket = "elasticbeanstalk-us-east-1-320699877354")
train_hold <- punts_at_ball_lands_df_roll_yards_wide[shuff,]
folds <- cut(seq(1,nrow(train)),breaks = nfolds,labels = FALSE)
y_hat <- y_hatv2 <- y_hat_base_rate <- rep(0,nrow(punts_at_ball_received_df_roll_yards_wide_sub))
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
  
  mod_lm_sub <- lm(roll_yards~.,data = train_cv)
  mod_lm_subv2 <- lm(roll_yards~.,data = train_cv %>% dplyr::select(roll_yards,football_x,football_y,absoluteYardlineNumber,
                                                                           kickLength,hangTime)) # pre snap only info
  
  
  
  #mod_lgb_sub <- lightgbm(params = params,train_matrix,nrounds = 100)
  
  y_hat_base_rate[indexes] <- mean(train_cv$roll_yards)
  y_hat[indexes] <- predict(mod_lm_sub,newdata = test_cv)
  y_hatv2[indexes] <- predict(mod_lm_subv2,newdata = test_cv)
  #y_hat_lgbm[indexes,] <- matrix(predict(mod_lgb_sub,valm),ncol = 3,byrow = TRUE)
  
  
}

best_param = list()
best_seednumber = 1234
best_rmse = Inf
best_rmse_index = 0
start = Sys.time()
for (iter in 1:5){
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
  mdcv <- xgb.cv(data = model.matrix(roll_yards ~ .,data = train %>% dplyr::select(roll_yards,football_x,football_y,absoluteYardlineNumber,
                                                                             kickLength,hangTime)),
                 label = as.vector(train$roll_yards), 
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

library(MLmetrics)
MAE(y_pred = y_hat_base_rate,y_true = train$roll_yards)
MAE(y_pred = y_hat,y_true = train$roll_yards)
RMSE(y_pred = y_hat_base_rate,y_true = train$roll_yards) # 6.080987
RMSE(y_pred = y_hat,y_true = train$roll_yards) # 3.152402
RMSE(y_pred = y_hatv2,y_true = train$roll_yards) # 3.122402
best_rmse # 3.964111

mod_lm_roll_yards <- lm(roll_yards~.,data = train)


best_param <- list(objective = "reg:squarederror",
                   #eval.metric = list("mse"),
                   metrics = list("mse"),
                   max_depth = 6,
                   eta = 0.1118561,
                   gamma = 0.1544718, 
                   subsample = 0.6577441,
                   colsample_bytree = 0.7496422, 
                   min_child_weight = 24,
                   max_delta_step = 10
)

mod_roll_yards <- xgboost(
  data = model.matrix(roll_yards ~ .,data = train),
  label = as.vector(train$roll_yards), 
  params = best_param, 
  nthread=6, 
  nrounds=1000,
  verbose = F, 
  maximize=FALSE
)

s3saveRDS(x = mod_roll_yards,
          object = "mod_roll_yards.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354",
          multipart = TRUE,
          prefix="nfl-big-data-bowl-2022")
