---
title: "Predicting and Understanding the Risk of Myocardial Infarction & Coronary Heart Disease"
output: html_document
---

# Introduction

This project is based on part of the data from the Framingham Heart Study, focusing on the risk factors for myocardial infarction (heart attack) and coronary heart disease (CHD).

The objective is to assess both causal relationships and predictive models for CHD risk, particularly evaluating the impact of smoking, demographics, and health indicators. More specifically, considering the influence of gender, age, and BMI. It also examines whether smoking affects BMI and whether gender influences smoking behavior.

## **Data Description**

-   **Key Variables:**

    -   Outcome Variable**:** `MI_FCHD` (1 = Event Occurred, 0 = No Event)

    -   Predictors:

        -   `CIGPDAY`: Cigarettes smoked per day

        -   `SEX`: Gender

        -   `AGE`: Age

        -   `TOTCHOL`: Total Cholesterol

        -   `SYSBP`: Systolic Blood Pressure

        -   `DIABP`: Diastolic Blood Pressure

        -   `BMI`: Body Mass Index

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

```{r, message=FALSE, warning=FALSE}
library(tidyverse)
library(ggplot2)
library("mgcv")
library(glmtoolbox)
library(statmod)
library(MuMIn)
library(pROC)
library(ggdag)
df <- read.csv('frmgham2.csv')
```

## 1. Analysis of Smoking & Heart Disease Risk

The proposed hypothesis are:

-   The number of cigarettes smoked per day affects the risk of hospitalized myocardial infarction or fatal coronary heart disease (CHD).

-   Gender and age influence the risk of myocardial infarction or CHD.

### 1.1 Data Initial Inspections

```{r}
head(df)

str(df)
```

### 1.2 Data Exploration & Visualizations

Since our main focus is around the influence of gender, age and number of cigarettes, we will be specifically explore these variables.

```{r}
q1=df %>% group_by(CIGPDAY) %>% 
  summarise(n=n(),event=sum(MI_FCHD), p=sum(MI_FCHD)/n())

#the relationship between the probability of MI_FCHD and CIGPDAY 
plot(p~CIGPDAY, data=q1, main="risk ~ the number of cigarettes")
table(q1$n<10)

#the relationship between the probability of MI_FCHD and quantile of CIGPDAY (20 groups)
df %>% mutate(cig_group = ntile(CIGPDAY, 20))%>% 
  group_by(cig_group) %>%
  summarise(n=n(),event=sum(MI_FCHD), p=sum(MI_FCHD)/n()) %>% 
  ggplot(aes(x=cig_group, y=p)) +
  geom_point()+
  ggtitle("risk of MI vs the quantile of the number of smoked cigarettes")

#the relationship between the cigarettes and probability of MI_FCHD depending on sex
df %>% mutate(cig_group = ntile(CIGPDAY, 20)) %>%
  group_by(SEX,cig_group) %>%
  summarise(n=n(),event=sum(MI_FCHD), p=sum(MI_FCHD)/n()) %>%
  ggplot(aes(x=cig_group,y=p,color=SEX))+geom_point()

#the relationship between the cigarettes and probability of MI_FCHD depending on age group
df %>% mutate(cig_group = ntile(CIGPDAY, 20))%>%  
  mutate(age_group = as.factor(ntile(AGE, 4))) %>% 
  group_by(age_group,cig_group) %>%
  summarise(n=n(),event=sum(MI_FCHD), p=sum(MI_FCHD)/n()) %>%
  ggplot(aes(x=cig_group,y=p,color=age_group))+geom_point(size=2)

#the relationship between the age and probability of MI_FCHD depending on sex 
df %>% mutate(cig_group = ntile(CIGPDAY, 20)) %>%  
  mutate(age_group = as.factor(ntile(AGE, 50))) %>% 
  group_by(AGE,SEX) %>% 
  summarise(n=n(),event=sum(MI_FCHD), p=sum(MI_FCHD)/n()) %>%
  ggplot(aes(x=AGE,y=p,color=SEX))+geom_point(size=2)
```

Insights:

The data is grouped based on the number of smoked cigarettes per day and further calculated the risk of having an event of hospitalised myocardial infarction or fatal coronary heart disease for each group.

There is a noticable large variability from the first plot, which might due to insufficient number of observations per group. Therefore, I further split the data based on the quantiles of the number of smoked cigarettes per day to ensure a sufficient number of observations in each group. And proceed to calculate the risk of myocardial infarction for each group. It is found that as the number of cigarettes smoked per day increases, the risk of having an event of hospitalised myocardial infarction or fatal coronary heart disease increases.

From the above plots, it was unclear on whether the relationship between the number of smoked cigarettes and the risk of hospitalised myocardial or fatal coronary heart disease depends on age and gender. It seems that the relationships between the risk of hospitalised myocardial or fatal coronary heart disease and age depends on gender.

### 1.3 Logistic Regression Analysis

```{r}
a4q1 = df %>% select(SEX, CIGPDAY, AGE, MI_FCHD, BMI) # a new data with selected variables

a4q1.gam = gam(MI_FCHD ~ s(CIGPDAY)+s(AGE), data=a4q1)
summary(a4q1.gam)$s.table
plot(a4q1.gam) #all linear relationships 
detach('package:mgcv')

a4q1.glm <- glm(MI_FCHD~., family='binomial', data=a4q1)
a4q1.int.glm <- glm(MI_FCHD~CIGPDAY*AGE*SEX, family='binomial', data=a4q1)
summary(a4q1.int.glm)
anova(a4q1.glm, a4q1.int.glm, test='Chisq') #evidence of interaction 

gam_model <- gam(MI_FCHD ~ s(BMI) + AGE + SEX + CIGPDAY, data=df, family=binomial)
summary(gam_model)

options(na.action = "na.fail")
a4q1.fits <-dredge(a4q1.int.glm) #find possible models 
a4q1.fits[1:10,]

#find the best model 
model1 <-get.models(a4q1.fits, 1)[[1]]
hltest(model1) #no evidence against 
plot(qresiduals(model1)~predict(model1)) #looks good 
summary(model1)
100*(exp(confint(model1))-1)
```

Insights:

After assumption checks, there are no evidence against the model and the residual plot looks reasonable.

It is found that the impact of the number of smoked cigarettes per day on the outcome depends on gender. For men, given the same age, the number of smoked cigarettes per day does not significantly affect the risk of having an event of hospitalized myocardial infarction or fatal coronary heart disease. However, given the same age, for women, the number of smoked cigarettes per day increases the risk of having an event of hospitalized MI of fatal CHD. For females, the odds of having an event of hospitalized MI of fatal CHD by between 0.186% and 3.70%.

## 2. Building a CHD Risk Prediction Model

The research aims to develop a risk prediction model for hospitalized myocardial infarction or fatal coronary heart disease using all collected variables.

Since the sample size of the data is reasonably large, I will randomly select 1000 samples to serve as the test set and use the remaining data to build prediction models.

```{r}
set.seed(1234)
test.index = sample(1:nrow(df), 1000)
FirstFram.tr = df[-test.index,]
FirstFram.te = df[test.index,]
summary(FirstFram.tr)
summary(FirstFram.te)
```

### 2.1 Model Selection

```{r, message=FALSE, warning=FALSE}
#AICc
df.glm = glm(MI_FCHD ~., family='binomial', data = FirstFram.tr)
df.fits <- dredge(df.glm)
print(df.fits[1:10, ])

#BIC 
df.fits2 <- dredge(df.glm, rank="BIC")
print(df.fits2[1:10, ])

options(na.action = "na.omit")
```

### 2.1 Model Evaluation

```{r, message=FALSE, warning=FALSE}
options(width=66)
out = rep(0,20)
for (i in 1:20) {
  newpreds = predict(get.models(df.fits, i) [[1]],
                     newdata=FirstFram.te, type='response')
  my.roc = roc(response=FirstFram.te$MI_FCHD,
               predictor=newpreds, ci=TRUE, quiet=TRUE)
  out[i] = my.roc$auc
}
print(which.max(out)) #the first one 

model2 = get.models(df.fits, 2)[[1]] #first model is chosen 

#assumption check 
hltest(model2) #no evidence against 
plot(qresiduals(model2)~predict(model2)) #looks good 
summary(model2)

model2.roc = roc(response = FirstFram.te$MI_FCHD,
                 predictor=predict(model2,newdata = FirstFram.te, type='response'),ci='TRUE')
model2.roc
```

Final Model:

$log(p_i/1-p_i) = \beta_0 + \beta_1 \times AGE_i + \beta_2 \times BMI_i+\beta_3 \times CIGPDAY_i + \beta_4 \times SEXWomen_i +\beta_5 \times SYSBP_i + \beta_6 \times TOTCHOL_i$

where $Y_i \sim Binomial(1, p_i)$

$AGE_i$ is the age at exam for ith participant; $BMI_i$ is the Body Mass Index of the ith participant; $CIGPDAY_i$ is the number of cigarettes smoked each day for the ith particpant; $SEXWomen_i$ =1 when the ith participant is a women; $SYSBP_i$ is the systolic blood pressure of the ith participant; $TOTCHOL_i$ is the serum total cholesterol of the ith participant; $p_i$ is the probability that the ith participant has hospitalised myocardial infarction or fatal coronary heart disease.

Based on AUC, it is recommended to use the 2nd model for prediction. Its estimated AUC is 0.721-0.7955 (95% confidence interval)

## 3. Classification

```{r}
pred = predict(model2,newdata = FirstFram.te, type='response')
predict = ifelse(pred <= 0.2, 0, 1)
conf_matrix=table(predict,FirstFram.te$MI_FCHD )

library(caret)
sensitivity(conf_matrix) #sensitivity
specificity(conf_matrix) #specificty 
1-(sum(diag(conf_matrix))/sum(conf_matrix))#prediction error 
```

Insights:

Using a 0.2 threshold, the confusion matrix was generated to classify predicted probabilities into event (1) or no event (0). The model achieved a sensitivity of 75.1%, meaning it correctly identified 75.1% of individuals who experienced myocardial infarction or fatal coronary heart disease. The specificity was 60.9%, indicating that 60.9% of non-events were correctly classified. The overall prediction error was 27.3%, meaning the model misclassified 27.3% of cases.

## Conclusion

This study examined the relationship between smoking and the risk of hospitalised myocardial infarction (MI) or fatal coronary heart disease (CHD) while considering the roles of gender, age, and BMI. The analysis showed that as the number of cigarettes smoked per day increases, the risk of CHD generally increases. However, when analysing the impact by gender, a significant interaction was observed: for women, smoking was associated with a higher risk of CHD, with odds increasing between 0.186% and 3.70%, whereas for men, smoking did not have a statistically significant effect.

Additionally, the predictive modeling phase aimed to estimate CHD risk based on all collected variables. Using logistic regression with model selection via AICc and BIC, the final model demonstrated a sensitivity of 75.1% and specificity of 60.9%, indicating a stronger ability to detect at-risk individuals. However, with a misclassification rate of 27.3%, further improvements. Overall, this study highlights smoking as a key risk factor for CHD, particularly among women.
