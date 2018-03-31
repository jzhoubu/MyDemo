na.plot=function(data){
  missing=sapply(data,function(x)sum(is.na(x))/nrow(data))
  #missing=missing[order(missing,decreasing = T)]
  nadata=missing[missing>0]
  na_df=data.frame(var=names(nadata),na=nadata,row.names = NULL)
  ggplot(na_df)+
    geom_bar(aes(x=reorder(var,na),y=na),stat='identity', fill='red')+
    labs(y='% Missing',x=NULL,title='Percent of Missing Data by Feature') +
    coord_flip(ylim = c(0,1))  
}
