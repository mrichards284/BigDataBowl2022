# Packages
install.packages("nflfastR")
library(nflfastR)
library(aws.s3)
library(plyr)
library(tidyverse)

# Function to convert s3 object to csv and read into r
s3_to_csv <- function(s3_path,header_yes = TRUE) {
  
  usercsvobj <- aws.s3::get_object(s3_path)
  csvcharobj <- rawToChar(usercsvobj)
  con <- textConnection(csvcharobj) 
  data <- read.csv(con,header = header_yes) 
  close(con) 
  return(data)
  
}
# Function to convert s3 object to csv and read into r
s3_to_txt <- function(s3_path) {
  
  usercsvobj <- aws.s3::get_object(s3_path)
  csvcharobj <- rawToChar(usercsvobj)
  con <- textConnection(csvcharobj) 
  data <- read.delim(con, header = FALSE, sep = "\t")
  close(con) 
  return(data)
  
}


# Pull all objects in s3 bucket
get_bucket_df('elasticbeanstalk-us-east-1-320699877354', prefix="nfl-big-data-bowl-2022")

s_plays <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/python_code/s_plays_01.txt',header_yes = FALSE)
s_columns <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/python_code/s_columns.txt',header_yes = FALSE)
s_dict <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/python_code/s_dict.csv')
s_index <- s3_to_csv('s3://elasticbeanstalk-us-east-1-320699877354/nfl-big-data-bowl-2022/python_code/s_index.txt',header_yes = FALSE)

names(s_plays) <- s_columns %>% t %>% c
names(s_index) <- "play_index"

names(s_plays) <- c(paste0("x_off_initial_",1:11),
                    paste0("x_def_initial_",12:22),
                    "football_x",
                    paste0("y_off_initial_",1:11),
                    paste0("y_def_initial_",12:22),
                    "football_y",
                    "football_x_land",
                    "football_y_land",
                    "hang_time",
                    paste0("x_off_land_",1:11),
                    paste0("x_def_land_",12:22),
                    paste0("y_off_land_",1:11),
                    paste0("y_def_land_",12:22))

# create design matrix
s_plays$play_index <- s_index$play_index
s_plays_ex1 <- s_plays[1:633798,] %>%
  dplyr::select(play_index,contains("x_off_land"),contains("x_def_land")) %>%
  pivot_longer(!(play_index),values_to = "x",names_to = "player_x") %>%
  group_by(play_index) %>%
  mutate(row_number = row_number()) %>%
  ungroup() %>%
  left_join(
    s_plays[1:633798,] %>%
      dplyr::select(play_index,contains("y_off_land"),contains("y_def_land")) %>%
      pivot_longer(!(play_index),values_to = "y",names_to = "player_y") %>%
      group_by(play_index) %>%
      mutate(row_number = row_number()) %>%
      ungroup(),by = c("play_index","row_number")) %>%
  left_join(s_plays %>% dplyr::select(play_index,football_x_land,football_y_land,hang_time),by = c("play_index")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(dist = sqrt((x - football_x_land)^2 + (y - football_y_land)^2),
                team_off_def = ifelse(row_number <= 11,"off","def")) %>%
  dplyr::ungroup() %>%
  dplyr::group_by(play_index,team_off_def) %>%
  dplyr::arrange(dist) %>%
  dplyr::mutate(player = paste0("player_by_dist_",team_off_def,"_",ifelse(row_number() < 10,paste0("0",row_number()),row_number()))) %>%
  dplyr::ungroup() %>%
  pivot_wider(c(play_index,football_x_land,football_y_land),values_from = c(x,y,dist),names_from = player)

s_plays_ex2 <- s_plays[(nrow(s_plays)/5+1):(nrow(s_plays)/5 * 2),] %>%
  dplyr::select(play_index,contains("x_off_land"),contains("x_def_land")) %>%
  pivot_longer(!(play_index),values_to = "x",names_to = "player_x") %>%
  group_by(play_index) %>%
  mutate(row_number = row_number()) %>%
  ungroup() %>%
  left_join(
    s_plays[(nrow(s_plays)/5+1):(nrow(s_plays)/5 * 2),] %>%
      dplyr::select(play_index,contains("y_off_land"),contains("y_def_land")) %>%
      pivot_longer(!(play_index),values_to = "y",names_to = "player_y") %>%
      group_by(play_index) %>%
      mutate(row_number = row_number()) %>%
      ungroup(),by = c("play_index","row_number")) %>%
  left_join(s_plays %>% dplyr::select(play_index,football_x_land,football_y_land,hang_time),by = c("play_index")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(dist = sqrt((x - football_x_land)^2 + (y - football_y_land)^2),
                team_off_def = ifelse(row_number <= 11,"off","def")) %>%
  dplyr::ungroup() %>%
  dplyr::group_by(play_index,team_off_def) %>%
  dplyr::arrange(dist) %>%
  dplyr::mutate(player = paste0("player_by_dist_",team_off_def,"_",ifelse(row_number() < 10,paste0("0",row_number()),row_number()))) %>%
  dplyr::ungroup() %>%
  pivot_wider(c(play_index,football_x_land,football_y_land),values_from = c(x,y,dist),names_from = player)

s_plays_ex3 <- s_plays[((nrow(s_plays)/5 * 2)+1):(nrow(s_plays)/5 * 3),] %>%
  dplyr::select(play_index,contains("x_off_land"),contains("x_def_land")) %>%
  pivot_longer(!(play_index),values_to = "x",names_to = "player_x") %>%
  group_by(play_index) %>%
  mutate(row_number = row_number()) %>%
  ungroup() %>%
  left_join(
    s_plays[((nrow(s_plays)/5 * 2)+1):(nrow(s_plays)/5 * 3),] %>%
      dplyr::select(play_index,contains("y_off_land"),contains("y_def_land")) %>%
      pivot_longer(!(play_index),values_to = "y",names_to = "player_y") %>%
      group_by(play_index) %>%
      mutate(row_number = row_number()) %>%
      ungroup(),by = c("play_index","row_number")) %>%
  left_join(s_plays %>% dplyr::select(play_index,football_x_land,football_y_land,hang_time),by = c("play_index")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(dist = sqrt((x - football_x_land)^2 + (y - football_y_land)^2),
                team_off_def = ifelse(row_number <= 11,"off","def")) %>%
  dplyr::ungroup() %>%
  dplyr::group_by(play_index,team_off_def) %>%
  dplyr::arrange(dist) %>%
  dplyr::mutate(player = paste0("player_by_dist_",team_off_def,"_",ifelse(row_number() < 10,paste0("0",row_number()),row_number()))) %>%
  dplyr::ungroup() %>%
  pivot_wider(c(play_index,football_x_land,football_y_land),values_from = c(x,y,dist),names_from = player)


s_plays_ex4 <- s_plays[((nrow(s_plays)/5 * 3)+1):(nrow(s_plays)/5 * 4),] %>%
  dplyr::select(play_index,contains("x_off_land"),contains("x_def_land")) %>%
  pivot_longer(!(play_index),values_to = "x",names_to = "player_x") %>%
  group_by(play_index) %>%
  mutate(row_number = row_number()) %>%
  ungroup() %>%
  left_join(
    s_plays[((nrow(s_plays)/5 * 3)+1):(nrow(s_plays)/5 * 4),] %>%
      dplyr::select(play_index,contains("y_off_land"),contains("y_def_land")) %>%
      pivot_longer(!(play_index),values_to = "y",names_to = "player_y") %>%
      group_by(play_index) %>%
      mutate(row_number = row_number()) %>%
      ungroup(),by = c("play_index","row_number")) %>%
  left_join(s_plays %>% dplyr::select(play_index,football_x_land,football_y_land,hang_time),by = c("play_index")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(dist = sqrt((x - football_x_land)^2 + (y - football_y_land)^2),
                team_off_def = ifelse(row_number <= 11,"off","def")) %>%
  dplyr::ungroup() %>%
  dplyr::group_by(play_index,team_off_def) %>%
  dplyr::arrange(dist) %>%
  dplyr::mutate(player = paste0("player_by_dist_",team_off_def,"_",ifelse(row_number() < 10,paste0("0",row_number()),row_number()))) %>%
  dplyr::ungroup() %>%
  pivot_wider(c(play_index,football_x_land,football_y_land),values_from = c(x,y,dist),names_from = player)


s_plays_ex5 <- s_plays[((nrow(s_plays)/5 * 4)+1):(nrow(s_plays)/5 * 5),] %>%
  dplyr::select(play_index,contains("x_off_land"),contains("x_def_land")) %>%
  pivot_longer(!(play_index),values_to = "x",names_to = "player_x") %>%
  group_by(play_index) %>%
  mutate(row_number = row_number()) %>%
  ungroup() %>%
  left_join(
    s_plays[((nrow(s_plays)/5 * 4)+1):(nrow(s_plays)/5 * 5),] %>%
      dplyr::select(play_index,contains("y_off_land"),contains("y_def_land")) %>%
      pivot_longer(!(play_index),values_to = "y",names_to = "player_y") %>%
      group_by(play_index) %>%
      mutate(row_number = row_number()) %>%
      ungroup(),by = c("play_index","row_number")) %>%
  left_join(s_plays %>% dplyr::select(play_index,football_x_land,football_y_land,hang_time),by = c("play_index")) %>%
  dplyr::rowwise() %>%
  dplyr::mutate(dist = sqrt((x - football_x_land)^2 + (y - football_y_land)^2),
                team_off_def = ifelse(row_number <= 11,"off","def")) %>%
  dplyr::ungroup() %>%
  dplyr::group_by(play_index,team_off_def) %>%
  dplyr::arrange(dist) %>%
  dplyr::mutate(player = paste0("player_by_dist_",team_off_def,"_",ifelse(row_number() < 10,paste0("0",row_number()),row_number()))) %>%
  dplyr::ungroup() %>%
  pivot_wider(c(play_index,football_x_land,football_y_land),values_from = c(x,y,dist),names_from = player)

s_plays_design_matrix <- rbind(s_plays_ex1,s_plays_ex2,s_plays_ex3,s_plays_ex4,s_plays_ex5)


s_indexv2 <- s_index %>%
  dplyr::rowwise() %>%
  dplyr::mutate(absoluteYardlineNumber = as.numeric(strsplit(play_index,"/")[[1]][1]),
                horizontal_loc = strsplit(play_index,"/")[[1]][2],
                start_y = ifelse(horizontal_loc == "L",24,ifelse(horizontal_loc == "C",26.5,ifelse(horizontal_loc == "R",29,99))),
                land_x = as.numeric(strsplit(play_index,"/")[[1]][3]),
                land_y = as.numeric(strsplit(play_index,"/")[[1]][4]),
                hang_time = as.numeric(strsplit(play_index,"/")[[1]][5]),
                play_num = as.numeric(strsplit(play_index,"/")[[1]][6]),
                kickLength = sqrt((absoluteYardlineNumber - land_x)^2 + (start_y - land_y)^2)) %>%
  dplyr::ungroup()


s_plays_design_matrixv2 <- s_plays_design_matrix %>%
  dplyr::rename(football_x = football_x_land,
                football_y = football_y_land) %>%
  left_join(s_indexv2 %>%
              ungroup() %>%
              dplyr::select(play_index,absoluteYardlineNumber,hang_time,kickLength),by = c("play_index")) %>%
  dplyr::mutate(event = 99999) %>%
  dplyr::rename(hangTime = hang_time) %>%
  dplyr::select(play_index,event,football_x,football_y,absoluteYardlineNumber,kickLength,
                contains("x_player"),contains("y_player"),contains("dist_player"),hangTime) %>%
  arrange(play_index)

s3saveRDS(x = s_plays_design_matrixv2,
          object = "s_plays_design_matrix_20220101.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354",
          multipart = TRUE,
          prefix="nfl-big-data-bowl-2022")

s3saveRDS(x = s_indexv2,
          object = "s_indexv2_20220102.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354",
          multipart = TRUE,
          prefix="nfl-big-data-bowl-2022")


