woe=function(x,t,a,plot=True){
  temp=cbind(x,as.numeric(t))  %>% 
    apply(2,as.numeric) %>% 
    data.frame()
  colnames(temp)=c('x','t')
  IV <- create_infotables(temp, y='t', ncore=2,bins=a)  
  
  woe=data.frame(data.frame(IV$Tables)[,1],data.frame(IV$Tables)[,4])
  colnames(woe)=c('varible','woe')
  
  
  if(plot==T)
  {
    ggplot(woe,aes(x=varible,y=woe))+
      geom_bar(stat='identity',fill='red')+
      theme(axis.title.x=element_blank(),axis.title.y=element_blank())
  }
  else {return(IV$Tables$x)}
}