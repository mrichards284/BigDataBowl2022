# Packages
install.packages("nflfastR")
library(nflfastR)
library(aws.s3)
library(tidyverse)
library(xgboost)
library(latex2exp)


# Function to convert s3 object to csv and read into r
s3_to_csv <- function(s3_path,header_yes = TRUE) {
  
  usercsvobj <- aws.s3::get_object(s3_path)
  csvcharobj <- rawToChar(usercsvobj)
  con <- textConnection(csvcharobj) 
  data <- read.csv(con,header = header_yes) 
  close(con) 
  return(data)
  
}

s_indexv2 <- s3readRDS(object = "s_results.RDS",
          bucket = "elasticbeanstalk-us-east-1-320699877354")

demo = read.csv('/Users/marcrichards/Desktop/BigDataBowl/demo.csv')

initial_pos_v2 <- demo %>%
  filter(event == "start" & playId == 0) %>%
  mutate(x = ifelse(jerseyNumber == 10,x - 10,ifelse(jerseyNumber == 5,x -14,x-15)),
         y = ifelse(jerseyNumber == 10, y + 4,y + 2),
         epa = 0) %>%
  rename(land_x = x,
         land_y = y)

initial_pos <- demo %>%
  filter(event == "start" & playId == 0) %>%
  mutate(x = ifelse(jerseyNumber == 10,x + 9,ifelse(jerseyNumber == 5,x + 26,x+24)),
         y = ifelse(jerseyNumber == 10, y + 4,y + 2),
         epa = 0) %>%
  rename(land_x = x,
         land_y = y)

### Perfect Execution

yy<-seq(-6,56,by = 2)
xx<-seq(70,120,by = 2)
breaks_data <-list(xx=xx, yy=yy)

g1 <- s_indexv2 %>%
  dplyr::filter(absoluteYardlineNumber == 70 & horizontal_loc == 'L') %>%
  dplyr::select(land_x,land_y,epa) %>%
  ggplot(aes(x = land_x,y = land_y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -1.5,name = "EPA") +
  #geom_point(data = data.frame(land_x = 63,land_y = 26.5,epa = 0),size = 7) +
  geom_point(data = initial_pos,aes(x = land_x,y = land_y,color = team),size = 4) +
  labs(
    x = "",
    y = "",
    title = "Intended Punt Location EPA under Perfect Execution",
    subtitle = "Punts from Opponent 40 yardline, Center, Hang Time 4 Secs"
  ) +
  scale_x_continuous(limits = c(58,122),breaks = c(60,70,80,90,100,110),label = c("50","40","30","20","10","G")) +
  scale_color_manual(values = c("dark green","brown","black")) +
  geom_segment(aes(y = -2,yend = -2,x = 58,xend = 120),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 120,xend = 120),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 110,xend = 110),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 100,xend = 100),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 90,xend = 90),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 80,xend = 80),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 70,xend = 70),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 60,xend = 60),color = "black") +
  geom_segment(aes(y = 52,yend = 52,x = 58,xend = 120),color = "black") +
  coord_flip() +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank()
  ) +
  guides(colour="none")

g2 <- s_indexv2 %>%
  dplyr::filter(hang_time == 5 & absoluteYardlineNumber == 30 & horizontal_loc == 'L') %>%
  dplyr::select(land_x,land_y,epa) %>%
  ggplot(aes(x = land_x,y = land_y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = 0) +
  geom_point(data = data.frame(land_x = 40,land_y = 24,epa = 0),size = 7) +
  labs(
    x = "",
    y = "",
    title = "Punt Location EPA under Perfect Execution",
    subtitle = "Punts from the 30 yardline, Left Hash, Hang Time 5 Secs"
  ) +
  scale_x_continuous(limits = c(40,90),breaks = c(40,50,60,70,80,90),label = c(30,40,50,40,30,20)) +
  coord_flip() +
  theme_bw()

grid.arrange(g1,g2,ncol = 2)

gg_field_2(g)

yy<-seq(-6,56,by = 2)
xx<-seq(70,120,by = 2)
breaks_data <-list(xx=xx, yy=yy)

### epsilon 1
ep1_70_C <- sampled_df_y_70_loc_c_ep_1 %>%
  filter(absoluteYardlineNumber == 70 & horizontal_loc == 'C') %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -2,name = "EPA") +
  #geom_point(data = data.frame(land_x = 63,land_y = 26.5,epa = 0),size = 7) +
  geom_point(data = initial_pos,aes(x = land_x,y = land_y,color = team),size = 4) +
  labs(
    x = "",
    y = "",
    title = TeX("Optimal Punt Location EPA,  $\\epsilon  = 1$"),
    subtitle = "Punts from the opponent 40 yardline, Center, Hang Time 4 Secs"
  ) +
  scale_x_continuous(limits = c(58,120),breaks = c(60,70,80,90,100,110),label = c("50","40","30","20","10","G")) +
  scale_color_manual(values = c("dark green","brown","black")) +
  geom_segment(aes(y = -2,yend = -2,x = 58,xend = 120),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 120,xend = 120),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 110,xend = 110),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 100,xend = 100),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 90,xend = 90),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 80,xend = 80),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 70,xend = 70),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 60,xend = 60),color = "black") +
  geom_segment(aes(y = 52,yend = 52,x = 58,xend = 120),color = "black") +
  coord_flip() +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank()
  ) +
  guides(colour="none")

### epsilon 0.1
ep2_70_C <- sampled_df_y_70_loc_c_ep_0.1 %>%
  filter(absoluteYardlineNumber == 70 & horizontal_loc == 'C') %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -2,name = "EPA") +
  #geom_point(data = data.frame(land_x = 63,land_y = 26.5,epa = 0),size = 7) +
  geom_point(data = initial_pos,aes(x = land_x,y = land_y,color = team),size = 4) +
  labs(
    x = "",
    y = "",
    title = TeX("Optimal Punt Location EPA,  $\\epsilon  = 0.1$"),
    subtitle = "Punts from the opponent 40 yardline, Center, Hang Time 4 Secs"
  ) +
  scale_x_continuous(limits = c(58,120),breaks = c(60,70,80,90,100,110),label = c("50","40","30","20","10","G")) +
  scale_color_manual(values = c("dark green","brown","black")) +
  geom_segment(aes(y = -2,yend = -2,x = 58,xend = 120),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 120,xend = 120),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 110,xend = 110),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 100,xend = 100),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 90,xend = 90),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 80,xend = 80),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 70,xend = 70),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 60,xend = 60),color = "black") +
  geom_segment(aes(y = 52,yend = 52,x = 58,xend = 120),color = "black") +
  coord_flip() +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank()
  ) +
  guides(colour="none")

grid.arrange(ep1_70_C,ep2_70_C,ncol = 2)

yy<-seq(-6,56,by = 2)
xx<-seq(30,90,by = 2)
breaks_data <-list(xx=xx, yy=yy)

ep1_30_C <- sampled_df_y_30_loc_c_ep_1 %>%
  filter(absoluteYardlineNumber == 30 & horizontal_loc == 'C') %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -0.6,name = "EPA") +
  #geom_point(data = data.frame(land_x = 63,land_y = 26.5,epa = 0),size = 7) +
  geom_point(data = initial_pos_v2,aes(x = land_x,y = land_y,color = team),size = 4) +
  labs(
    x = "",
    y = "",
    title = TeX("Optimal Punt Location EPA,  $\\epsilon  = 1$"),
    subtitle = "Punts from the 20 yardline, Center, Hang Time 4 Secs"
  ) +
  scale_x_continuous(limits = c(18,90),breaks = c(20,30,40,50,60,70,80,90),label = c("10","20","30","40","50","40","30","20")) +
  scale_color_manual(values = c("dark green","brown","black")) +
  geom_segment(y = -2,yend = -2,x = 30,xend = 90,color = "black") +
  geom_segment(y = 52,yend = 52,x = 30,xend = 90,color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 90,xend = 90),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 80,xend = 80),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 70,xend = 70),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 60,xend = 60),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 50,xend = 50),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 40,xend = 40),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 30,xend = 30),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 20,xend = 20),color = "black") +
  coord_flip() +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank()
  ) +
  guides(colour="none")

### epsilon 0.1
ep2_30_C <- sampled_df_y_30_loc_c_ep_0.1 %>%
  filter(absoluteYardlineNumber == 30 & horizontal_loc == 'C') %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -0.5,name = "EPA") +
  geom_point(data = data.frame(intended_x = 25,intended_y = 24,epa = 0),size = 7) +
  labs(
    x = "",
    y = "",
    title = TeX("Optimal Punt Location EPA,  $\\epsilon  = 0.1$"),
    subtitle = "Punts from the 20 yardline, Center, Hang Time 4 Secs"
  ) +
  scale_x_continuous(limits = c(30,90),breaks = c(30,40,50,60,70,80,90),label = c("20","30","40","50","40","30","20")) +
  geom_segment(y = -2,yend = -2,x = 30,xend = 90,color = "black") +
  geom_segment(y = 52,yend = 52,x = 30,xend = 90,color = "black") +
  coord_flip() +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
    #panel.grid.major = element_blank(),
    #panel.grid.minor = element_blank(),
    #panel.border = element_blank(),
    #panel.background = element_blank()
  )

grid.arrange(ep1_70_C,ep2_70_C,ncol = 2)


grid.arrange(ep1_70_C,ep1_30_C,ncol = 2)


ep1_30_C_t_4 <- sampled_df_y_30_loc_c_ep_1 %>%
  filter(absoluteYardlineNumber == 30 & horizontal_loc == 'C' & intended_t == 4) %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -0.6,name = "EPA") +
  #geom_point(data = data.frame(land_x = 63,land_y = 26.5,epa = 0),size = 7) +
  geom_point(data = initial_pos_v2,aes(x = land_x,y = land_y,color = team),size = 4) +
  labs(
    x = "",
    y = "",
    title = TeX("Optimal Punt Location EPA,  $\\epsilon  = 1$"),
    subtitle = "Punts from the 20 yardline, Center, Hang Time 4 Secs"
  ) +
  scale_x_continuous(limits = c(18,90),breaks = c(20,30,40,50,60,70,80,90),label = c("10","20","30","40","50","40","30","20")) +
  scale_color_manual(values = c("dark green","brown","black")) +
  geom_segment(y = -2,yend = -2,x = 30,xend = 90,color = "black") +
  geom_segment(y = 52,yend = 52,x = 30,xend = 90,color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 90,xend = 90),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 80,xend = 80),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 70,xend = 70),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 60,xend = 60),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 50,xend = 50),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 40,xend = 40),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 30,xend = 30),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 20,xend = 20),color = "black") +
  coord_flip() +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank()
  ) +
  guides(colour="none")



ep1_30_C_t_5 <- sampled_df_y_30_loc_c_ep_1 %>%
  filter(absoluteYardlineNumber == 30 & horizontal_loc == 'C' & intended_t == 5) %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa)) +
  #stat_binhex()
  stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -0.75,name = "EPA") +
  #geom_point(data = data.frame(land_x = 63,land_y = 26.5,epa = 0),size = 7) +
  geom_point(data = initial_pos_v2,aes(x = land_x,y = land_y,color = team),size = 4) +
  labs(
    x = "",
    y = "",
    title = TeX("Optimal Punt Location EPA,  $\\epsilon  = 1$"),
    subtitle = "Punts from the 20 yardline, Center, Hang Time 5 Secs"
  ) +
  scale_x_continuous(limits = c(18,90),breaks = c(20,30,40,50,60,70,80,90),label = c("10","20","30","40","50","40","30","20")) +
  scale_color_manual(values = c("dark green","brown","black")) +
  geom_segment(y = -2,yend = -2,x = 30,xend = 90,color = "black") +
  geom_segment(y = 52,yend = 52,x = 30,xend = 90,color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 90,xend = 90),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 80,xend = 80),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 70,xend = 70),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 60,xend = 60),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 50,xend = 50),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 40,xend = 40),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 30,xend = 30),color = "black") +
  geom_segment(aes(y = -2,yend = 52,x = 20,xend = 20),color = "black") +
  coord_flip() +
  theme_bw() +
  theme(
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank()
  ) +
  guides(colour="none")


grid.arrange(ep1_30_C_t_4,ep1_30_C_t_5,ncol = 2)


v <- ggplot(volcano3d, aes(x, y, z = z))
v + stat_contour(geom="polygon", aes(fill=..level..)) 

sampled_df %>%
  filter(intended_t == 4.5) %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa)) +
  #stat_binhex()
  #stat_summary_2d(fun=mean,na.rm = T, breaks=breaks_data) +
  #scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -2) +
  stat_contour(aes(fill=..level..)) +
  geom_point(data = data.frame(intended_x = 65,intended_y = 24,epa = 0),size = 7) +
  labs(
    x = "",
    y = "",
    title = "Punt Location EPA w/ \epsilon = 1",
    subtitle = "Punts from the opponent 45 yardline, Left Hash, Hang Time 4 Secs"
  ) +
  scale_x_continuous(limits = c(60,120),breaks = c(60,70,80,90,100,110,120),label = c("50","40","30","20","10","G","")) +
  coord_flip() +
  theme_bw()

sampled_df %>%
  filter(intended_t == 4.5) %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa,fill = epa)) + 
  geom_tile() + 
  #geom_contour(color = "white", alpha = 0.5) + 
  scale_fill_gradient2(low = "blue",mid = "white",high = "red",midpoint = -2) +
  scale_x_continuous(limits = c(60,120),breaks = c(60,70,80,90,100,110,120),label = c("50","40","30","20","10","G","")) +
  coord_flip() +
  theme_bw()

filled.contour(z = sampled_df$epa,x = sampled_df$intended_x,y = sampled_df$intended_y)


# library
library(latticeExtra) 

# create data
set.seed(1) 
data <- data.frame(x = rnorm(100), y = rnorm(100)) 
data$z <- with(data, x * y + rnorm(100, sd = 1)) 

# showing data points on the same color scale 
levelplot(epa ~ intended_x * intended_y, sampled_df %>%
            filter(intended_t == 4.5) %>% mutate(epa = epa+5), 
          panel = panel.levelplot.points, cex = 1.2
) +
  layer_(panel.2dsmoother(..., n = 100)) 


sampled_df_output <- sampled_df %>%
  filter(intended_t == 4.5) %>%
  arrange(epa) %>%
  mutate(percentile = round((row_number()/nrow(sampled_df))*100,0),
         percentile = ifelse(percentile == 0,1,percentile)) %>%
  dplyr::select(intended_x,intended_y,percentile) %>% as.matrix()

for (i in 1:nrow(sampled_df_output)){
  if (i ==1){
    sampled_df_output2 <- matrix(rep(sampled_df_output[i,],sampled_df_output[i,3]),ncol = 3,byrow = TRUE)
  } else {
    sampled_df_output2 <- rbind(sampled_df_output2,matrix(rep(sampled_df_output[i,],sampled_df_output[i,3]),ncol = 3,byrow = TRUE))
    
  }
}

sampled_df_output2_df <- data.frame(sampled_df_output2) 
names(sampled_df_output2_df) <- c("intended_x","intended_y","percentile")

sampled_df  %>%
  ggplot(aes(x = intended_x,y = intended_y,z = epa,fill = epa)) +
  stat_density2d(aes(fill=..level..), contour=TRUE, n=200)

sampled_df_output2_df %>%
  ggplot(aes(x = intended_x,y = intended_y)) +
  geom_bin2d() +
  theme_bw()

               