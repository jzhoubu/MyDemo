



xgb.cv.plot=function(input,output){
  if(output!="ks"){
    history=input
    train_history=history[,1:6]%>%mutate(id=row.names(history),class="train")
    test_history=history[,7:12]%>%mutate(id=row.names(history),class="test")
    colnames(train_history)=c("auc.mean","auc.std","rmse.mean","rmse.std","error.mean","error.std","id","class")
    colnames(test_history)=c("auc.mean","auc.std","rmse.mean","rmse.std","error.mean","error.std","id","class")
    
    his=rbind(train_history,test_history)
    his$id=his$id%>%as.numeric
    his$class=his$class%>%factor
    
    if(output=="auc"){ 
      auc=ggplot(data=his,aes(x=id, y=auc.mean,ymin=auc.mean-auc.std,ymax=auc.mean+auc.std,fill=class),linetype=class)+
        geom_line()+
        geom_ribbon(alpha=0.5)+
        labs(x="nround",y=NULL,title = "XGB Cross Validation AUC")+
        theme(title=element_text(size=15))+
        theme_bw()
      return(auc)
    }
    
    
    if(output=="rmse"){
      rmse=ggplot(data=his,aes(x=id, y=rmse.mean,ymin=rmse.mean-rmse.std,ymax=rmse.mean+rmse.std,fill=class),linetype=class)+
        geom_line()+
        geom_ribbon(alpha=0.5)+
        labs(x="nround",y=NULL,title = "XGB Cross Validation RMSE")+
        theme(title=element_text(size=15))+
        theme_bw()
      return(rmse)
    }
    
    if(output=="error"){
      error=ggplot(data=his,aes(x=id,y=error.mean,ymin=error.mean-error.std,ymax=error.mean+error.std,fill=class),linetype=class)+
        geom_line()+
        geom_ribbon(alpha=0.5)+
        labs(x="nround",y=NULL,title = "XGB Cross Validation ERROR")+
        theme(title=element_text(size=15))+
        theme_bw()
      return(error)
    }
  }  
  
  
  else{
    history=input
    train_history=history[,1:2]%>%mutate(id=row.names(history),class="train")
    test_history=history[,3:4]%>%mutate(id=row.names(history),class="test")
    colnames(train_history)[1:2]=c("ks.mean","ks.std")
    colnames(test_history)[1:2]=c("ks.mean","ks.std")
    
    his=rbind(train_history,test_history)
    his$id=his$id%>%as.numeric
    his$class=his$class%>%factor
    
    ks=ggplot(data=his,aes(x=id, y=ks.mean,ymin=ks.mean-ks.std,ymax=ks.mean+ks.std,fill=class),linetype=class)+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      labs(x="nround",y=NULL,title = "XGB Cross Validation KS")+
      theme(title=element_text(size=15))+
      theme_bw()
    return(ks)
  }
  
  
  
}





