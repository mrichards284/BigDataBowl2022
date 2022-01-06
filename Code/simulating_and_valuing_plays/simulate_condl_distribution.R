# Packages
library(tidyverse)
library(plyr)
library(aws.s3)
library(MASS)

# Read in Data
s_results <- s3readRDS(object = "s_results_20220102_PM.RDS",
                     bucket = "elasticbeanstalk-us-east-1-320699877354")

sigma_stars <- s3readRDS(object = "sigma_stars.RDS",
                       bucket = "elasticbeanstalk-us-east-1-320699877354")

# Roll up s results for look up table
s_results_rolledup <- s_results %>%
  group_by(absoluteYardlineNumber,horizontal_loc,start_y,land_x,land_y,hang_time) %>%
  summarize(
    epa = mean(epa,na.rm = T),
    .groups = "drop"
  )


# set starting x and y as well as Sigma.
yardline <- 70
horizontal_loc <- "C"
Sigma <- sigma_stars[6]$c_6
  
sampl_loc_func <- function(intended_x,intended_y,intended_t,yardline,horizontal_loc,Sigma){
  samples <- data.frame(mvrnorm(n = 1000,mu = c(intended_x,intended_y,intended_t),Sigma = Sigma))
  samples <- samples %>%
    mutate(x = ifelse(x > 131,132,ifelse(x < 47, 48,round_any(x,2)))-1,
           y = ifelse(y < -26,-26,ifelse(y > 78,78,round_any(y,2))),
           hangTime = round(hangTime,1),
           absoluteYardlineNumber = yardline,
           horizontal_loc = horizontal_loc,
           intended_x = intended_x,
           intended_y = intended_y,
           intended_t = intended_t) %>%
    as.tibble() %>%
    left_join(s_results_rolledup %>% dplyr::select(absoluteYardlineNumber,horizontal_loc,land_x,land_y,hang_time,epa),by = c("x"="land_x","y"="land_y","hangTime"="hang_time","absoluteYardlineNumber","horizontal_loc")) %>%
    group_by(absoluteYardlineNumber,horizontal_loc,intended_x,intended_y,intended_t) %>%
    summarize(
      epa = mean(epa,na.rm = T),
      .groups = "drop"
    )
  
  return(samples)
}

yardline <- 70
horizontal_loc <- "C"
Sigma_Star <- sigma_stars[1]$c_1
epsilon <- 0.1
Sigma <- epsilon*Sigma_Star

start <- Sys.time()
for (x in seq(85,119,by = 1)){
  for (y in seq(-6,58,by = 1)){
    for (t in seq(3.5,5,by = 0.1)){
      if (x == 85 & y == -6 & t == 3.5){
        sampled_df <- sampl_loc_func(intended_x = x,intended_y = y,intended_t = t,yardline = yardline,horizontal_loc = horizontal_loc,Sigma = Sigma)
      } else {
        sampled_df <- rbind(sampled_df,sampl_loc_func(intended_x = x,intended_y = y,intended_t = t,yardline = yardline,horizontal_loc = horizontal_loc,Sigma = Sigma))
      }
    }
  }
  print(x)
}
stop <- Sys.time()

sampled_df_y_70_loc_c_ep_0.1 <- sampled_df

s3saveRDS(
  x = sampled_df_y_70_loc_c_ep_0.1,
  object = "sampled_df_y_70_loc_c_ep_0.1.RDS",
  bucket = "elasticbeanstalk-us-east-1-320699877354",
  multipart = TRUE
)


yardline <- 30
horizontal_loc <- "C"
Sigma_Star <- sigma_stars[5]$c_5
epsilon <- 1
Sigma <- epsilon*Sigma_Star

start <- Sys.time()
for (x in seq(45,85,by = 1)){
  for (y in seq(-6,58,by = 1)){
    for (t in seq(3.5,5,by = 0.1)){
      if (x == 45 & y == -6 & t == 3.5){
        sampled_df <- sampl_loc_func(intended_x = x,intended_y = y,intended_t = t,yardline = yardline,horizontal_loc = horizontal_loc,Sigma = Sigma)
      } else {
        sampled_df <- rbind(sampled_df,sampl_loc_func(intended_x = x,intended_y = y,intended_t = t,yardline = yardline,horizontal_loc = horizontal_loc,Sigma = Sigma))
      }
    }
  }
  print(x)
}
stop <- Sys.time()

sampled_df_y_30_loc_c_ep_1 <- sampled_df

s3saveRDS(
  x = sampled_df_y_30_loc_c_ep_1,
  object = "sampled_df_y_30_loc_c_ep_1.RDS",
  bucket = "elasticbeanstalk-us-east-1-320699877354",
  multipart = TRUE
)

yardline <- 30
horizontal_loc <- "C"
Sigma_Star <- sigma_stars[5]$c_5
epsilon <- 0.1
Sigma <- epsilon*Sigma_Star

start <- Sys.time()
for (x in seq(45,85,by = 1)){
  for (y in seq(-6,58,by = 1)){
    for (t in seq(3.5,5,by = 0.1)){
      if (x == 45 & y == -6 & t == 3.5){
        sampled_df <- sampl_loc_func(intended_x = x,intended_y = y,intended_t = t,yardline = yardline,horizontal_loc = horizontal_loc,Sigma = Sigma)
      } else {
        sampled_df <- rbind(sampled_df,sampl_loc_func(intended_x = x,intended_y = y,intended_t = t,yardline = yardline,horizontal_loc = horizontal_loc,Sigma = Sigma))
      }
    }
  }
  print(x)
}
stop <- Sys.time()

sampled_df_y_30_loc_c_ep_0.1 <- sampled_df

s3saveRDS(
  x = sampled_df_y_30_loc_c_ep_0.1,
  object = "sampled_df_y_30_loc_c_ep_0.1.RDS",
  bucket = "elasticbeanstalk-us-east-1-320699877354",
  multipart = TRUE
)

sampled_df %>%
  arrange(epa)

