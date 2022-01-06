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


punts_at_snap_df <- tracking %>%
  dplyr::mutate(unique_id = paste0(gameId,"_",playId)) %>%
  dplyr::filter(unique_id %in% punt_plays$unique_id) %>%
  dplyr::filter(event %in% c("ball_snap")) %>%
  dplyr::group_by(unique_id) %>%
  dplyr::slice_min(frameId) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(x = case_when(playDirection == "left"~120-x,
                              TRUE~x),
                y = case_when(playDirection == "left"~53-y,
                              TRUE~y)) %>%
  filter(displayName == "football") %>%
  dplyr::select(gameId,playId,x,y)

plays <- plays %>%
  mutate(absoluteYardlineNumber = as.double(absoluteYardlineNumber)) 

punt_loc_sub <- punts_at_ball_lands_df %>%
  left_join(plays,by = c("gameId","playId")) %>%
  left_join(games,by = c("gameId")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(absoluteYardlineNumber = case_when(playDirection == "left"~120-absoluteYardlineNumber,
                                                   TRUE~absoluteYardlineNumber)) %>%
  left_join(punts_at_snap_df %>% rename(start_x = x,start_y = y),by = c("gameId","playId")) %>%
  left_join(pff_scouting %>% dplyr::select(gameId,playId,hangTime),by = c("gameId","playId")) %>%
  filter(absoluteYardlineNumber >= 60 & start_y < 25 & y < 30) %>%
  filter(displayName == "football") %>%
  dplyr::select(x,y,hangTime) %>%
  filter(!is.na(hangTime))

summary(punt_loc_sub)
c_1 <- cov(punt_loc_sub)  

ggplot(data = punt_loc_sub,aes(x = x,y = y)) +
  geom_point() +
  theme_bw() +
  coord_flip()

library(MASS)
mvrnorm(n= 20,mu = c(40,15,4.2),Sigma = c_1)


punt_loc_sub2 <- punts_at_ball_lands_df %>%
  left_join(plays,by = c("gameId","playId")) %>%
  left_join(games,by = c("gameId")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(absoluteYardlineNumber = case_when(playDirection == "left"~120-absoluteYardlineNumber,
                                                   TRUE~absoluteYardlineNumber)) %>%
  left_join(punts_at_snap_df %>% rename(start_x = x,start_y = y),by = c("gameId","playId")) %>%
  left_join(pff_scouting %>% dplyr::select(gameId,playId,hangTime),by = c("gameId","playId")) %>%
  filter(absoluteYardlineNumber >= 50 & absoluteYardlineNumber <= 60 & start_y < 25 & y < 30) %>%
  filter(displayName == "football") %>%
  dplyr::select(x,y,hangTime) %>%
  filter(!is.na(hangTime))

summary(punt_loc_sub2)
c_2 <- cov(punt_loc_sub2)  

ggplot(data = punt_loc_sub2,aes(x = x,y = y)) +
  geom_point() +
  theme_bw() +
  coord_flip()

mvrnorm(n= 10,mu = c(40,15,4.2),Sigma = c_2)


punt_loc_sub3 <- punts_at_ball_lands_df %>%
  left_join(plays,by = c("gameId","playId")) %>%
  left_join(games,by = c("gameId")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(absoluteYardlineNumber = case_when(playDirection == "left"~120-absoluteYardlineNumber,
                                                   TRUE~absoluteYardlineNumber)) %>%
  left_join(punts_at_snap_df %>% rename(start_x = x,start_y = y),by = c("gameId","playId")) %>%
  left_join(pff_scouting %>% dplyr::select(gameId,playId,hangTime),by = c("gameId","playId")) %>%
  filter(absoluteYardlineNumber >= 40 & absoluteYardlineNumber <= 50 & start_y < 25 & y < 30) %>%
  filter(displayName == "football") %>%
  dplyr::select(x,y,hangTime) %>%
  filter(!is.na(hangTime))

summary(punt_loc_sub3)
c_3 <- cov(punt_loc_sub3)  

ggplot(data = punt_loc_sub3,aes(x = x,y = y)) +
  geom_point() +
  theme_bw() +
  coord_flip()

mvrnorm(n= 10,mu = c(40,15,4.2),Sigma = c_2)


punt_loc_sub4 <- punts_at_ball_lands_df %>%
  left_join(plays,by = c("gameId","playId")) %>%
  left_join(games,by = c("gameId")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(absoluteYardlineNumber = case_when(playDirection == "left"~120-absoluteYardlineNumber,
                                                   TRUE~absoluteYardlineNumber)) %>%
  left_join(punts_at_snap_df %>% rename(start_x = x,start_y = y),by = c("gameId","playId")) %>%
  left_join(pff_scouting %>% dplyr::select(gameId,playId,hangTime),by = c("gameId","playId")) %>%
  filter(absoluteYardlineNumber >= 30 & absoluteYardlineNumber <= 40 & start_y < 25 & y < 30) %>%
  filter(displayName == "football") %>%
  dplyr::select(x,y,hangTime) %>%
  filter(!is.na(hangTime))

summary(punt_loc_sub4)
c_4 <- cov(punt_loc_sub4)  

ggplot(data = punt_loc_sub4,aes(x = x,y = y)) +
  geom_point() +
  theme_bw() +
  coord_flip()

mvrnorm(n= 10,mu = c(40,15,4.2),Sigma = c_2)

punt_loc_sub5 <- punts_at_ball_lands_df %>%
  left_join(plays,by = c("gameId","playId")) %>%
  left_join(games,by = c("gameId")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(absoluteYardlineNumber = case_when(playDirection == "left"~120-absoluteYardlineNumber,
                                                   TRUE~absoluteYardlineNumber)) %>%
  left_join(punts_at_snap_df %>% rename(start_x = x,start_y = y),by = c("gameId","playId")) %>%
  left_join(pff_scouting %>% dplyr::select(gameId,playId,hangTime),by = c("gameId","playId")) %>%
  filter(absoluteYardlineNumber >= 20 & absoluteYardlineNumber <= 30 & start_y < 25 & y < 30) %>%
  filter(displayName == "football") %>%
  dplyr::select(x,y,hangTime) %>%
  filter(!is.na(hangTime))

summary(punt_loc_sub5)
c_5 <- cov(punt_loc_sub5)  

ggplot(data = punt_loc_sub5,aes(x = x,y = y)) +
  geom_point() +
  theme_bw() +
  coord_flip()

mvrnorm(n= 10,mu = c(40,15,4.2),Sigma = c_5)


punt_loc_sub6 <- punts_at_ball_lands_df %>%
  left_join(plays,by = c("gameId","playId")) %>%
  left_join(games,by = c("gameId")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(absoluteYardlineNumber = case_when(playDirection == "left"~120-absoluteYardlineNumber,
                                                   TRUE~absoluteYardlineNumber)) %>%
  left_join(punts_at_snap_df %>% rename(start_x = x,start_y = y),by = c("gameId","playId")) %>%
  left_join(pff_scouting %>% dplyr::select(gameId,playId,hangTime),by = c("gameId","playId")) %>%
  filter(absoluteYardlineNumber >= 10 & absoluteYardlineNumber <= 20 & start_y < 25 & y < 30) %>%
  filter(displayName == "football") %>%
  dplyr::select(x,y,hangTime) %>%
  filter(!is.na(hangTime))

summary(punt_loc_sub6)
c_6 <- cov(punt_loc_sub6)  

ggplot(data = punt_loc_sub6,aes(x = x,y = y)) +
  geom_point() +
  theme_bw() +
  coord_flip()

mvrnorm(n= 10,mu = c(40,10,4.2),Sigma = c_6)

sigma_stars <- list(c_1 = c_1,
                    c_2 = c_2,
                    c_3 = c_3,
                    c_4 = c_4,
                    c_5 = c_5,
                    c_6 = c_6)

s3saveRDS(x = sigma_stars,
                         object = "sigma_stars.RDS",
                         bucket = "elasticbeanstalk-us-east-1-320699877354")
