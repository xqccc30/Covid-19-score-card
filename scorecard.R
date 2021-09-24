library(readxl)
library(scorecard)



data_train <- read_excel("data_traink.xlsx")
data_test <- read_excel("data_test.xlsx")
data_test = data_test[,c(2:32)]
data_train = data_train[,c(2:32)]
bins = woebin(data_train,'outcome',bin_num_limit = 3)
breaks_adj = woebin_adj(data_train, "outcome", bins)
bins_adj = woebin(data_train, "outcome", breaks_list=breaks_adj, print_step=0)

aa = c("outcome")
for(i in bins_adj){
  if(i$total_iv[1]>=0.02)
  {
    aa = append(aa,i$variable[1])
  }
}



train_woe <-  woebin_ply(data_train[aa], bins_adj[aa[-1]])
test_woe <-  woebin_ply(data_test[aa], bins_adj[aa[-1]])




m1 <- glm( outcome ~ ., family = binomial(), data = train_woe[,c(-15,-19)])
m_step <- step(m1, direction="both", trace = FALSE)
m2 <- eval(m_step$call)

name1 = names(m2$coefficients)[-1]

#lasso
x= data.matrix(train_woe[,name1])
y=as.matrix(train_woe$outcome)
alpha1_fit <- glmnet(x,y,alpha=1,family="binomial")
set.seed(2324)
fit = cv.glmnet(x, y, family="binomial", alpha=1,nfolds = 5,type.measure = "auc")
plot(fit)
nam = coef(alpha1_fit,s=fit$lambda.min)
nam = data.frame(value = as.vector(nam),name =  row.names(nam))
formula = paste("outcome",paste(nam$name[-1],collapse = "+"),sep = "~")

m2 = glm(formula = formula, family = binomial(), 
         data = train_woe)

train_pred <- predict(m2, train_woe, type = 'response')
test_pred <- predict(m2, test_woe, type = 'response')
train_perf <- perf_eva( train_pred, train_woe$outcome,title = 'train',show_plot = c("roc","ks"))
test_perf <- perf_eva( test_pred, test_woe$outcome,title = 'test',show_plot = c("roc","ks"))
card = scorecard(bins_adj, m2,points0 = 600, odds0 = 1/19, pdo = 50)
write.csv(card,"scorecard.csv")


train_score <- scorecard_ply(data_train, card, print_step = 0)
# 验证集评分
test_score <- scorecard_ply(data_test, card, print_step = 0)
psi_result <- perf_psi(
  score = list(train = train_score, test = test_score),
  label = list(train = data_train$outcome, test = data_test$outcome))




color_2 <- colorRampPalette(c("magenta", "cyan"))


cuttable = data.frame()
minz = min(train_score)
maxz = max(train_score)
bin = 126.8

for (z in c(1:10))
{
  cut = round((minz+z*126.8),0)
  TP = 0
  TN = 0
  FP  = 0
  FN =0
  N = 0
  d = 0
  t  = 0
  for(i in test_score$score)
  {  
    
    t = t+1
    if((i<=(minz+z*126.8)))
    {
      N = N+1
      if(data_test[t,]$outcome==1)
      {
        d = d+1
        TP = TP+1
      }
      else
      {
        FP = FP+1
      }
      
    }
    else
    {
      if (data_test[t,]$outcome==0)
      {
        TN = TN+1
      }
      else
      {
        
        FN = FN+1
      }
      
    }
    
  }
  acc = (TP+TN)/(97)
  sen = TP/(TP+FN)
  spe = TN/(FP+TN)
  PPV = TP/(TP+FP)
  npv = TN/(TN+FN)
  F1 = 2*TP/(2*TP+FP+FN)
  cc = cbind(cut,N,acc,TP,TN,FP,FN,sen,spe,PPV,npv,F1,d)
  cuttable = rbind(cuttable,cc)
}


xb = Score(list("fit" = m2),formula = outcome~1,null.model = FALSE,plots = c("calibration","boxplot"),conf.int = TRUE,
           metrics = c("brier"),B = 10000,M = 20,data = test_woe)

plotCalibration(axes = TRUE,xb,col = "red",brier.in.legend =TRUE,show.frequencies = TRUE,percent = TRUE,type = "l",xlab = "Predicted ",bty="l")

abline(h = c(0.25,0.5,0.75,1 ),lty = 2,bty="7")
abline(0,1,col = "black",lty = 2,lwd = 2,)
legend(0.85,0.35,c("","q"),col =c("black","red"),bty = "n",lty = c(2,1),lwd = c(2,2))
m3 = lrm(formula = outcome ~ RR_woe + temperature_woe + Age_woe + 
           Cluster_disease_woe + leukocyte_woe + Lymphocytes_woe + TNI_woe + 
           il_10_woe, data = train_woe,x=TRUE,y=TRUE)
cal1 = calibrate(m3,method = "boot",B = 1000,data = test_woe)
plot(1,type = "n",xlim = c(0,1),ylim = c(0,1),xlab = "Predocted Probability", ylab = "Observed Probability"
)
abline(0,1,col = "black",lty = 2,lwd = 2)
lines(cal1[,c("predy","calibrated.corrected")],type = "l",lwd = 2,col = "red",pch = 16)
legend(0.55,0.35,lty = c(1),lwd=c(2),col =c("red"),bty = "n")
abline(h = c(0.2,0.4,0.6,0.8,1 ),lty = 2)


#dca
pred = predict(c, data, type = 'response')
thresholds = seq(min(pred), max(pred), by = .005)
# net = A - B*(pt/(pt-1))
all = (dim(data[data[,"outcome"]==1,])[1] - (dim(data)[1]-dim(data[data[,"outcome"]==1,])[1])*(thresholds/(1-thresholds)))/dim(data)[1]
null = 0*thresholds
data$pred = pred
model =c()
for (i in thresholds)
{
  A = dim(data[(data[,"pred"]>=i)&(data[,"outcome"]==1),])[1]
  B = dim(data[(data[,"pred"]>=i)&(data[,"outcome"]==0),])[1]
  net = (A - B*(i/(1-i)))/dim(data)[1]
  model = append(model,net)
}
thre = c(thresholds,thresholds,thresholds,thresholds)
net = c(all,null,model,model1)
model = c(rep("ALL",length(thresholds)),rep("None",length(thresholds)),rep("Model",length(thresholds)),rep("Model1",length(thresholds)))
df = data.frame(thre,net,model)
ggplot(data=df, aes(x=thre, y=net, group=model, shape=model, colour=model)) + scale_y_continuous(limits=c(-0.1,0.15), breaks=seq(-0.1,0.15,0.05))+
  geom_line(aes(linetype=model), size=1.2) +    
  
  expand_limits(y=0) +                       
  scale_colour_hue(name="model",      
                   l=30)  + geom_hline(yintercept=seq(-0.1,0.3,by = 0.05), colour="grey", linetype="solid",size = 0.6) +                 
  
  xlab("Threshold") + ylab("Net benifit") + scale_colour_manual(values= c('#0023a0', 'darkred', 'black',"blue"))+
  ggtitle("DCA") +scale_linetype_manual(values = c('twodash', 'solid', 'solid',"solid"))+
  
  theme(axis.title.x = element_text(size = 15, color = "black", vjust = 0.5, hjust = 0.5, angle = 0),
        axis.title.y = element_text(size = 15,  color = "black",  vjust = 0.5, hjust = 0.5, angle = 90),
        axis.text.x = element_text(size = 12,  color = "black",  vjust = 0.5, hjust = 0.5, angle = 0),
        axis.text.y = element_text(size = 12,  color = "black", vjust = 0.5, hjust = 0.5, angle = 90),
        legend.position = c(0.93,0.82),panel.background=element_rect(fill="white",color="grey50"),
        axis.line = element_line(colour = "black"))




plot_decision_curve(dca,curve.names = "dca model",standardize = FALSE)