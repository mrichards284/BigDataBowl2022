# This is an adaption of Marshall Furman's gg field 
# https://github.com/mlfurman3/gg_field
gg_field_2 <- function(p,yardmin=0, yardmax=120, buffer=5, direction="horiz",
                     field_color="white",line_color="black",
                     sideline_color=field_color, endzone_color="lightgrey"){
  
  ## field dimensions (units=yards)
  xmin <- 0
  xmax <- 120
  
  ymin <- 0
  ymax <- 53.33
  
  
  ## distance from sideline to hash marks in middle (70 feet, 9 inches)
  hash_dist <- (70*12+9)/36
  
  ## yard lines locations (every 5 yards) 
  yd_lines <- seq(15,105,by=5)
  
  ## hash mark locations (left 1 yard line to right 1 yard line)
  yd_hash <- 11:109
  
  ## field number size
  num_size <- 5
  
  ## rotate field numbers with field direction
  ## first element is for right-side up numbers, second for upside-down
  angle_vec <- switch(direction, "horiz" = c(0, 180), "vert" = c(270, 90))
  num_adj <- switch(direction, "horiz" = c(-1, 1), "vert" = c(1, -1))
  
  ## list of annotated geoms
  p <- p +
    
    ## add field background 
    #annotate("rect", xmin=xmin, xmax=xmax, ymin=ymin-buffer, ymax=ymax+buffer) +
    
    ## add end zones
    annotate("segment", x=xmin, xend=xmin, y=ymin, yend=ymax) +
    annotate("segment", x=xmin+10, xend=xmin+10, y=ymin, yend=ymax) +
    annotate("segment", x=xmin, xend=xmin+10, y=ymin, yend=ymin) +
    annotate("segment", x=xmin, xend=xmin+10, y=ymax, yend=ymax) +
    
    
    annotate("segment", x=xmax, xend=xmax, y=ymin, yend=ymax) +
    annotate("segment", x=xmax-10, xend=xmax-10, y=ymin, yend=ymax) +
    annotate("segment", x=xmax, xend=xmax-10, y=ymin, yend=ymin) +
    annotate("segment", x=xmax, xend=xmax-10, y=ymax, yend=ymax) +
    
    ## add yardlines every 5 yards
    annotate("segment", x=yd_lines, y=ymin, xend=yd_lines, yend=ymax,
             col=line_color) +
    
    ## add thicker lines for endzones, midfield, and sidelines
    annotate("segment",x=c(0,10,60,110,120), y=ymin, xend=c(0,10,60,110,120), yend=ymax,
             lwd=1.3, col=line_color) +
    annotate("segment",x=0, y=c(ymin, ymax), xend=120, yend=c(ymin, ymax),
             lwd=1.3, col=line_color) +
    
    ## add field numbers (every 10 yards)
    ## field numbers are split up into digits and zeros to avoid being covered by yard lines
    ## numbers are added separately to allow for flexible ggplot stuff like facetting
    
    ## 0
    annotate("text",x=seq(20,100,by=10) + num_adj[2], y=ymin+12, label=0, angle=angle_vec[1],
             col=line_color, size=num_size) +
    
    ## 1
    annotate("text",label=1,x=c(20,100) + num_adj[1], y=ymin+12, angle=angle_vec[1],
             colour=line_color, size=num_size)+
    ## 2
    annotate("text",label=2,x=c(30,90) + num_adj[1], y=ymin+12, angle=angle_vec[1],
             colour=line_color, size=num_size)+
    ## 3
    annotate("text",label=3,x=c(40,80) + num_adj[1], y=ymin+12, angle=angle_vec[1],
             colour=line_color, size=num_size)+
    ## 4
    annotate("text",label=4,x=c(50,70) + num_adj[1], y=ymin+12, angle=angle_vec[1],
             colour=line_color, size=num_size)+
    ## 5
    annotate("text",label=5,x=60 + num_adj[1], y=ymin+12, angle=angle_vec[1],
             colour=line_color, size=num_size)+
    
    
    ## upside-down numbers for top of field
    
    ## 0
    annotate("text",x=seq(20,100,by=10) + num_adj[1], y=ymax-12, angle=angle_vec[2],
             label=0, col=line_color, size=num_size)+
    ## 1
    annotate("text",label=1,x=c(20,100) + num_adj[2], y=ymax-12, angle=angle_vec[2],
             colour=line_color, size=num_size)+
    ## 2
    annotate("text",label=2,x=c(30,90) + num_adj[2], y=ymax-12, angle=angle_vec[2],
             colour=line_color, size=num_size)+
    ## 3
    annotate("text",label=3,x=c(40,80) + num_adj[2], y=ymax-12, angle=angle_vec[2],
             colour=line_color, size=num_size)+
    ## 4
    annotate("text",label=4,x=c(50,70) + num_adj[2], y=ymax-12, angle=angle_vec[2],
             colour=line_color, size=num_size)+
    ## 5
    annotate("text",label=5,x=60 + num_adj[2], y=ymax-12, angle=angle_vec[2],
             colour=line_color, size=num_size)+
    
    
    ## add hash marks - middle of field
    #annotate("segment", x=yd_hash, y=hash_dist - 0.5, xend=yd_hash, yend=hash_dist + 0.5,
    #         color=line_color),
    #annotate("segment", x=yd_hash, y=ymax - hash_dist - 0.5, 
    #         xend=yd_hash, yend=ymax - hash_dist + 0.5,color=line_color),
    
    ## add hash marks - sidelines
    #annotate("segment", x=yd_hash, y=ymax, xend=yd_hash, yend=ymax-1, color=line_color),
    #annotate("segment", x=yd_hash, y=ymin, xend=yd_hash, yend=ymin+1, color=line_color),
    
    ## add conversion lines at 2-yard line
    #annotate("segment",x=12, y=(ymax-1)/2, xend=12, yend=(ymax+1)/2, color=line_color),
    #annotate("segment",x=108, y=(ymax-1)/2, xend=108, yend=(ymax+1)/2, color=line_color),
    
    ## cover up lines outside of field with sideline_color
    #annotate("rect", xmin=0, xmax=xmax, ymin=ymax, ymax=ymax+buffer, fill=sideline_color),
    #annotate("rect",xmin=0, xmax=xmax, ymin=ymin-buffer, ymax=ymin, fill=sideline_color),
    
    ## remove axis labels and tick marks
    labs(x="", y="") +
    theme(axis.text.x = element_blank(),axis.text.y = element_blank(),
          axis.ticks = element_blank()) +
    coord_flip(xlim=c(yardmin, yardmax), ylim = c(ymin-buffer,ymax+buffer), expand = FALSE) +
    coord_cartesian(xlim=c(yardmin, yardmax), ylim = c(ymin-buffer,ymax+buffer), 
                    expand = FALSE)
  
    
    ## clip axes to view of field
   # if(direction=="horiz"){
   #   coord_cartesian(xlim=c(yardmin, yardmax), ylim = c(ymin-buffer,ymax+buffer), 
  #                    expand = FALSE)
  #    
  #  } else if (direction=="vert"){
  #    ## flip entire plot to vertical orientation
  #    coord_flip(xlim=c(yardmin, yardmax), ylim = c(ymin-buffer,ymax+buffer), expand = FALSE)
  #    
  #  }
  #)
  
  return(p)
  
}
