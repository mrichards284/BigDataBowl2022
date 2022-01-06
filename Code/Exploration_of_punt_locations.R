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

ids <- nflfastR::fast_scraper_schedules(2018:2020) %>%
  #dplyr::filter(game_type == "SB") %>%
  dplyr::pull(game_id)
pbp <- nflfastR::build_nflfastR_pbp(ids)

nfl_pbp_2020 <- read_csv(url("https://raw.githubusercontent.com/ryurko/nflscrapR-data/master/play_by_play_data/regular_season/reg_pbp_2020.csv")) %>%
  dplyr::select(game_id,play_id,desc, play_type, ep, epa, home_wp) 
nfl_pbp_2019 <- read_csv(url("https://raw.githubusercontent.com/ryurko/nflscrapR-data/master/play_by_play_data/regular_season/reg_pbp_2019.csv")) %>%
  dplyr::select(game_id,play_id,desc, play_type, ep, epa, home_wp) 
nfl_pbp_2018 <- read_csv(url("https://raw.githubusercontent.com/ryurko/nflscrapR-data/master/play_by_play_data/regular_season/reg_pbp_2018.csv")) %>%
  dplyr::select(game_id,play_id,desc, play_type, ep, epa, home_wp) 

nfl_pbp <- rbind(nfl_pbp_2018,nfl_pbp_2019)

# Punts
punt_plays <- plays %>% filter(specialTeamsPlayType == "Punt")

# How many punt plays do we have?
nrow(punt_plays)

# How often do each of the Special Teams Results occur?
punt_plays %>%
  group_by(specialTeamsResult) %>%
  summarize(
    cnt = n(),
    .groups = "drop"
  ) %>%
  mutate(perc_of_total = cnt/sum(cnt))

# Intersting that 38% of punt return and 27.4% are fair caught.
# I am sure this changes based on the location of the punter?

# How often do penalties occur on punts?
punt_plays %>%
  mutate(penalty = ifelse(!is.na(penaltyCodes),1,0)) %>%
  group_by(penalty) %>%
  summarize(
    cnt = n(),
    .groups = "drop"
  ) %>%
  mutate(perc_of_total = cnt/sum(cnt))

# About 10% of punts have a penalty. But we should subset on returned punts, 
# when we do that we observe a spike closer to 20% of returns have a penalty on them.
punt_plays %>%
  filter(specialTeamsResult == "Return") %>%
  mutate(penalty = ifelse(!is.na(penaltyCodes),1,0)) %>%
  group_by(penalty) %>%
  summarize(
    cnt = n(),
    .groups = "drop"
  ) %>%
  mutate(perc_of_total = cnt/sum(cnt))

# Lets look at the distribution of punt length
punt_plays %>%
  ggplot(aes(x = kickLength)) +
  geom_histogram(binwidth = 5) +
  theme_bw()

punt_plays %>%
  select(kickLength) %>%
  summary()

# Fairly normally distributed, min 2 yards and max of 79 yards -- 113 NAs....

# Lets look at the distribution of kick Return Yardage, should be skewed?
punt_plays %>%
  ggplot(aes(x = kickReturnYardage)) +
  geom_histogram(binwidth = 5) +
  theme_bw()

punt_plays %>%
  select(kickReturnYardage) %>%
  summary()

## Skewed right, minimum -13 yards, and maximum of 99 yards

# Lets look at one punt 
GAMEID_SAMPLE <-2018123006 #2018123006
PLAYID_SAMPLE <- 2157 #2157
tracking2018_example <- filter(tracking2018, gameId == GAMEID_SAMPLE & playId == PLAYID_SAMPLE)
tracking2018_example %>% select(time,event) %>% distinct()
# how to get where the ball was actually punted from: punt
# how to get where ball landed... events: punt_received, punt_land, fair_catch, touchback (if it lands in the end zone)
# What about punts out of bounds: (look at plays with an event == "out_of_bounds") 
tracking2018 %>% select(event) %>% distinct()
tracking2018 %>% filter(event == "out_of_bounds")


# Lets filter on the x,y of these events for punts that were at the 40-40 yard line
punt_plays_sample <- plays %>% 
  filter(specialTeamsPlayType == "Punt" & yardlineNumber <= 50 & yardlineSide != possessionTeam) %>%
  mutate(unique_id = paste0(gameId,"_",playId)) %>%
  pull(unique_id)

punt_plays_sample_v2 <- plays %>% 
  filter(specialTeamsPlayType == "Punt" & yardlineNumber <= 10 & yardlineSide == possessionTeam) %>%
  mutate(unique_id = paste0(gameId,"_",playId)) %>%
  pull(unique_id)

punt_ball_landing_locs_df <- tracking %>%
  mutate(unique_id = paste0(gameId,"_",playId)) %>%
  filter(event %in% c("punt_received", "punt_land", "fair_catch", "touchback","out_of_bounds") &
           #unique_id %in% punt_plays_sample &
           displayName == "football") %>%
  group_by(unique_id) %>%
  slice_min(frameId) %>%
  ungroup() %>%
  mutate(x = case_when(playDirection == "left"~120-x,
                       TRUE~x)) 



# Lets spot check some of these against watching the play to see if these are right
# join tables to get game and play info -- gives us stuff like jersey number
tracking.example.merged <- tracking2018 %>% inner_join(games,by = c("gameId"="game_id")) %>% inner_join(plays,by = c("gameId","playId")) 
# Select Sample play
example.play <- tracking.example.merged %>% 
  filter(playId == 313 & gameId == 2018090901)
# Animate Sample Play
animate_play_func(example.play)
filter(punt_ball_landing_locs_df,playId == 313 & gameId == 2018090901)




punt_ball_landing_locs_df %>%
  filter(unique_id %in% punt_plays_sample) %>%
  ggplot(aes(x = x,y = y)) +
  gg_field(direction = "vert") +
  geom_point() 

yy<-seq(-15,65,length.out=27+1)
xx<-seq(0,140,length.out=50+1)
breaks_data <-list(xx=xx, yy=yy)

g <- punt_ball_landing_locs_df %>%
  filter(unique_id %in% punt_plays_sample) %>%
  mutate(z = 1) %>%
  dplyr::select(x,y,z) %>%
  rbind(data.frame(x = rep(xx+0.1,each = length(yy)),y = rep(yy+0.1,length(xx)),z = rep(0,length(xx)*length(yy)))) %>%
  ggplot(aes(x = x,y = y,z = z)) +
  #stat_binhex()
  stat_summary_2d(fun=sum,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "white",high = "blue",midpoint = 1) +
  geom_point(data = data.frame(x = 65,y = 26.5,z = 0),size = 7) +
  xlab("") +
  ylab("") +
  ggtitle("Location of Punts Between 40 and 50 Yardline of Opponent Territory")

gg_field_2(g)

g2 <- punt_ball_landing_locs_df %>%
  filter(unique_id %in% punt_plays_sample_v2) %>%
  mutate(z = 1) %>%
  dplyr::select(x,y,z) %>%
  rbind(data.frame(x = rep(xx+0.1,each = length(yy)),y = rep(yy+0.1,length(xx)),z = rep(0,length(xx)*length(yy)))) %>%
  ggplot(aes(x = x,y = y,z = 1)) +
  #stat_binhex()
  stat_summary_2d(fun=sum,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "red",mid = "white",high = "blue",midpoint = 1) +
  geom_point(data = data.frame(x = 10,y = 26.5,z = 0),size = 7) +
  xlab("") +
  ylab("") +
  ggtitle("Location of Punts from the 10 or less yardline")

gg_field_2(g2)


g3 <- punt_ball_landing_locs_df %>%
  filter(unique_id %in% punt_plays_sample) %>%
  left_join(nfl_pbp,by = c("gameId"="game_id","playId"="play_id")) %>%
  mutate(z = 1) %>%
  dplyr::select(x,y,epa) %>%
  rbind(data.frame(x = rep(xx+0.1,each = length(yy)),y = rep(yy+0.1,length(xx)),epa = rep(0,length(xx)*length(yy)))) %>%
  ggplot(aes(x = x,y = y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "red",mid = "white",high = "blue",midpoint = 0) +
  geom_point(data = data.frame(x = 65,y = 26.5,epa = 0),size = 7) +
  xlab("") +
  ylab("") +
  ggtitle("EPA based on location of Punts Between 40 and 50 Yardline of Opponent Territory")

gg_field_2(g3)

g4_epa <- punt_ball_landing_locs_df %>%
  filter(unique_id %in% punt_plays_sample_v2) %>%
  left_join(nfl_pbp,by = c("gameId"="game_id","playId"="play_id")) %>%
  mutate(z = 1) %>%
  dplyr::select(x,y,epa) %>%
  rbind(data.frame(x = rep(xx+0.1,each = length(yy)),y = rep(yy+0.1,length(xx)),epa = rep(0,length(xx)*length(yy)))) %>%
  ggplot(aes(x = x,y = y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "red",mid = "white",high = "blue",midpoint = 0) +
  geom_point(data = data.frame(x = 10,y = 26.5,epa = 0),size = 7) +
  xlab("") +
  ylab("") +
  ggtitle("EPA based on location of Punts at the 10 or less yardline")

gg_field_2(g4_epa)

plot_player_trajectory <- function(example.play){
  
  v <- example.play %>%
  ggplot(aes(x = x,y = y,group = displayName,color = team),size = 3) +
  geom_path() +
  geom_point(data = example.play %>% dplyr::filter(event == "ball_snap"),aes(x = x,y = y,color = team),size = 3)

  return(gg_field_2(v))

}
  
plot_player_trajectory(example.play)

