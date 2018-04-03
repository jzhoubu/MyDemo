###########################################################
#    Powered by ZJW
#
# 用来描绘变量var对于标签y的数量分布
# 会得到一个柱状图，X是var的levels,Y是数目，fill是labels
#
###########################################################
library(ggplot2)
library(data.table)


bar=function(var,y,plot=T){
  a=cbind(var,y)%>%data.table
  df=a[,.N,by=.(var,y)]
  
  df[is.na(var),var:=-99999]
  if(df$var%>%class=="numeric"){
    levels(df$var)=df$var%>%levels%>%as.numeric%>%sort
    df=df[order(y,var)]
  }
  
  df$var=factor(df$var)
  levels(df$var)=sub(x=levels(df$var),"-99999","NA")
  
  if(plot==T){
    ggplot(df,aes(x=var,y=N,fill=factor(y)))+
      geom_bar(stat='identity')
  }
  
  else{
    good=df[y==0]
    bad=df[y==1]
    df1=merge(good,bad,all=T,by='var')
    df1=mutate(df1,'good%'=N.x/(N.x+N.y))
    return(df1)
  } 
}

#bar(data1$stage_count,data1$y)
