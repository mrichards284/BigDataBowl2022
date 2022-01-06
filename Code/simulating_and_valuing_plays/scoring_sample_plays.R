# Packages
install.packages("nflfastR")
library(nflfastR)
library(aws.s3)
library(tidyverse)
library(xgboost)

# Function to convert s3 object to csv and read into r
s3_to_csv <- function(s3_path,header_yes = TRUE) {
  
  usercsvobj <- aws.s3::get_object(s3_path)
  csvcharobj <- rawToChar(usercsvobj)
  con <- textConnection(csvcharobj) 
  data <- read.csv(con,header = header_yes) 
  close(con) 
  return(data)
  
}


s_plays_design_mat <- s3readRDS(object = "s_plays_design_matrix_20220101.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354")

s_indexv2 <- s3readRDS(object = "s_indexv2_20220102.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354",
          prefix="nfl-big-data-bowl-2022")

mod_boost_punt_outcome <- s3readRDS(object = "mod_boost_punt_outcome.RDS",
                                    bucket = "elasticbeanstalk-us-east-1-320699877354")

mod_lm_roll_yards <- s3readRDS(object = "mod_lm_roll_yards.RDS",
                               bucket = "elasticbeanstalk-us-east-1-320699877354")

mod_roll_yards <- s3readRDS(object = "mod_roll_yards.RDS",
                               bucket = "elasticbeanstalk-us-east-1-320699877354")

mod_exp_yards_gained <- s3readRDS(object = "mod_exp_yards_gained.RDS",
                                    bucket = "elasticbeanstalk-us-east-1-320699877354")

mod_epa <- s3readRDS(object = "mod_epa.RDS",
                               bucket = "elasticbeanstalk-us-east-1-320699877354")

train_punt_outcome <- s3readRDS(object = "train_punt_outcome.RDS",
                                    bucket = "elasticbeanstalk-us-east-1-320699877354")

train_exp_return_yards <- s3readRDS(object = "train_exp_return_yards.RDS",
                                  bucket = "elasticbeanstalk-us-east-1-320699877354")

train_exp_roll_yards <- s3readRDS(object = "train_exp_roll_yards.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354")


pred_vals <- predict(mod_boost_punt_outcome,newdata = model.matrix(event~.,s_plays_design_mat[,names(train_punt_outcome)]))
pred_vals <- matrix(pred_vals,ncol = 3,byrow = TRUE)
colnames(pred_vals) <- c("fair_catch","punt_land","punt_received")
pred_vals <- data.frame(pred_vals)
pred_vals$play_index <- s_plays_design_mat$play_index
s_indexv2 <- s_indexv2 %>%
  left_join(pred_vals,by = c("play_index"))

head(pred_vals)

pred_max <- apply(pred_vals,1,which.max)

# fair catch yards is 0. Get ball at spot of catch.
s_indexv2$fair_catch_yards <- 0
# punt yards model
s_plays_design_mat$roll_yards <- 0
pred_roll_yards <- predict(mod_roll_yards,newdata = model.matrix(roll_yards~.,s_plays_design_mat[,names(train_exp_roll_yards)]))
s_indexv2$punt_land_yards <- pred_roll_yards
# Predict punt return yards
s_plays_design_mat$kickReturnYardage <- 0
pred_yards <- predict(mod_exp_yards_gained,newdata = model.matrix(kickReturnYardage~.,s_plays_design_mat[,names(train_exp_return_yards)]))
s_indexv2$punt_received_yards <- pred_yards

extra_s_indexv2 <- data.frame(expand.grid(seq(47,117,by = 2),c(seq(-26,-8,by = 2),seq(60,78,by = 2))))
extra_s_indexv2 <- rbind(extra_s_indexv2,data.frame(expand.grid(seq(119,135,by = 2),c(seq(-26,78,by = 2)))))
names(extra_s_indexv2) <- c("land_x","land_y")
extra_s_indexv2$hang_time <- 2.5
hangtimes <- seq(2.5,5,by = 0.1)
extra_s_indexv2_ <- extra_s_indexv2
for (i in 2:26){
  
  extra_s_indexv2_$hang_time <- hangtimes[i]
  extra_s_indexv2 <- rbind(extra_s_indexv2,extra_s_indexv2_)
  
}

start_variations <- expand.grid(seq(20,70,by = 10),c("L","C","R"))
names(start_variations) <- c("absoluteYardlineNumber","horizontal_loc")
extra_s_indexv2$absoluteYardlineNumber <- start_variations[1,1]
extra_s_indexv2$horizontal_loc <- start_variations[1,2]
extra_s_indexv2_ <- extra_s_indexv2
for (i in 2:18){
  
  extra_s_indexv2_$absoluteYardlineNumber <- start_variations[i,1]
  extra_s_indexv2_$horizontal_loc <- start_variations[i,2]
  extra_s_indexv2 <- rbind(extra_s_indexv2,extra_s_indexv2_)
  
}

extra_s_indexv2 <- extra_s_indexv2 %>%
  mutate(play_num = 0,
         start_y = 999,
         play_index = paste(absoluteYardlineNumber,horizontal_loc,land_x,land_y,hang_time,play_num,sep = "/"),
         kickLength = 999,
         fair_catch = 0,
         punt_land = 0,
         punt_received = 0,
         fair_catch_yards = 0,
         punt_land_yards = 0,
         punt_received_yards = 0) %>%
  dplyr::select(names(s_indexv2))


s_indexv2 <- s_indexv2 %>%
  rbind(extra_s_indexv2) %>%
  dplyr::mutate(final_yardline_fair_catch = 120 - land_x,
                final_yardline_punt_land = case_when(land_x + punt_land_yards < 110~120 - (land_x + punt_land_yards),
                                                     TRUE~30),
                final_yardline_punt_received_yards = 120 - (land_x - punt_received_yards),
                out_bounds_yardline = case_when(land_y < 0~120 - (land_x - .5*abs(land_y)),
                                                land_y >= 53~120 - (land_x - .5*(land_y-53)),
                                                TRUE~999),
                out_bounds_yardline = case_when(out_bounds_yardline <= 10~30,
                                                TRUE~out_bounds_yardline),
                touchback_yardline = case_when(land_x >= 110~30,
                                               TRUE~999))

s_epa_prior_df <- data.frame(
  season = rep(2021,nrow(s_indexv2)*1),
  half_seconds_remaining = rep(900,nrow(s_indexv2)*1),
  ydstogo = rep(10,nrow(s_indexv2)),
  yardline_100 = 100-(c(s_indexv2$absoluteYardlineNumber)-10),
  down1 = rep(0,nrow(s_indexv2)*1),
  down2 = rep(0,nrow(s_indexv2)*1),
  down3 = rep(0,nrow(s_indexv2)*1),
  down4 = rep(1,nrow(s_indexv2)*1)
)

s_epa_df <- data.frame(
  season = rep(2021,nrow(s_indexv2)*5),
  half_seconds_remaining = rep(895,nrow(s_indexv2)*5),
  ydstogo = rep(10,nrow(s_indexv2)*5),
  yardline_100 = 100-(c(s_indexv2$final_yardline_fair_catch,
                   s_indexv2$final_yardline_punt_land,
                   s_indexv2$final_yardline_punt_received_yards,
                   s_indexv2$out_bounds_yardline,
                   s_indexv2$touchback_yardline)-10),
  down1 = rep(1,nrow(s_indexv2)*5),
  down2 = rep(0,nrow(s_indexv2)*5),
  down3 = rep(0,nrow(s_indexv2)*5),
  down4 = rep(0,nrow(s_indexv2)*5)
)


ep_prior_probs <- predict(mod_epa,newdata = s_epa_prior_df,type = "probs")
ep_prior_vals <- apply(ep_prior_probs,1,function(x){return(sum(x * c(7,-7,3,-3,2,-2,0)))})
ep_after_probs <- predict(mod_epa,newdata = s_epa_df,type = "probs")
ep_after_vals <- apply(ep_after_probs,1,function(x){return(sum(x * c(-7,7,-3,3,-2,2,0)))})
ep_after_vals <- matrix(ep_after_vals,ncol = 5,byrow = FALSE)


s_indexv2$epa_fair_catch <- ep_after_vals[,1] - ep_prior_vals %>% t %>% c
s_indexv2$epa_punt_land <- ep_after_vals[,2] - ep_prior_vals %>% t %>% c
s_indexv2$epa_punt_received <- ep_after_vals[,3] - ep_prior_vals %>% t %>% c
s_indexv2$epa_out_of_bounds <- ep_after_vals[,4] - ep_prior_vals %>% t %>% c
s_indexv2$epa_touchback <- ep_after_vals[,5] - ep_prior_vals %>% t %>% c


s_indexv2 <- s_indexv2 %>%
  dplyr::rowwise() %>%
  dplyr::mutate(epa = (fair_catch*epa_fair_catch)+(punt_land*epa_punt_land)+(punt_received*epa_punt_received),
                  epa = case_when(out_bounds_yardline != 999~epa_out_of_bounds,
                                  touchback_yardline != 999~epa_touchback,
                                TRUE~epa)) %>%
  ungroup()

# Look at a few examples
View(filter(s_indexv2, absoluteYardlineNumber == 70 & horizontal_loc == "C" & land_x == 107 & land_y == 10 & hang_time == 4))

View(filter(s_indexv2, absoluteYardlineNumber == 70 & horizontal_loc == "L" & land_x == 87 & land_y == -4))


s3saveRDS(x = s_indexv2,
          object = "s_results_20220102_PM.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354",
          multipart = TRUE)



