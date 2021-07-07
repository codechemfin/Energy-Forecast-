%Modelaci�n y pron�stico de los precios spot en el mercado el�ctrico
%Colombiano con 1 capa oculta de base radial

clc
clear all
rng('default')
format long
DataSet=xlsread('DataSet.xls');
DataSet = DataSet(1600:end,:); 

plot(DataSet)
figure

[R,P] = corr(DataSet,'Type','Pearson'); % Matr�z de Coorrelaci�n y valores-P
hR = heatmap(R)%Mapa de Calor de las Correlaciones
figure
hP = heatmap(P)%Mapa de Calor de los valores-P
figure 
%Funciones de Entrenamiento
% X = ["traingd" "traingdm" "traingdx" "trainrp" "trainscg" "traincgf" "traincgb" ...
%     "trainoss" "trainbfg" "trainlm" "trainbr"];

[i,j]=size(DataSet);
p=DataSet(:,1:j-1);          %extraer las variables de entrada
t=DataSet(:,j);              %extrae la variable de salida

% % TRANSPONER MATRICES
% % en p y t, los DataSet estan en columnas, variables en filas
p=p';
t=t';
plot(DataSet(:,end-1))
legend('Precios spot de la energ�a el�ctrica')
figure 

plot(DataSet(:,1))
legend('Volumen embalsado en Kwh')
figure 

plot(DataSet(:,2))
legend('Caudal en Kwh')
figure 

plot(DataSet(:,3))
legend('Demanda en Kwh')
figure 

plot(DataSet(:,4))
legend('Precio del carb�n en US$/Ton')
figure 

plot(DataSet(:,5))
legend('Precio del gas en US$/BTU')
figure 

plot(DataSet(:,6))
legend('Precio del petr�leo en US$/Barril')
figure 

plot(DataSet(:,7))
legend('Tasa representativa del mercado en Cop$/US$')
figure 

tic    
       
net = newrb(p,t,1.5,5,i,200);               
net.trainFcn="traingd"; 
% net.trainParam.showWindow = 0;
net = configure(net,p,t);
net.divideFcn='dividerand';       
net=init(net);
%[net,tr]=train(net,p,t,'useParallel','no','useGPU','yes');  
[net,tr]=train(net,p,t);  
                
% Mostrar tiempo de commputo
toc

tic
             
[ptrain,pval,ptest]=divideind(p,tr.trainInd,tr.valInd,tr.testInd);         
[ttrain,tval,ttest]=divideind(t,tr.trainInd,tr.valInd,tr.testInd);                
a=sim(net,p);% Simulaci�n todos los DataSet                
b=sim(net,ptest); % Simulaci�n DataSet de test
c=sim(net,pval); % Simulaci�n DataSet de validaci�n
d=sim(net,ptrain); % Simulaci�n DataSet de entrenamiento              
    
toc
plot(t)%Todos los DataSet reales
hold on
plot(a)%Todos los DataSet simulados
legend('DataSet reales','DataSet simulados')
figure 

plot(ttest)%DataSet de test reales
hold on
plot(b) %DataSet de test simulados
legend('DataSet test reales','DataSet test simulados')
figure
 
plot(tval)%DataSet de validaci�n reales 
hold on
plot(c) %DataSet de validaci�n simulados 
legend('DataSet validaci�n reales','DataSet validaci�n simulados')
figure

plot(ttrain)%DataSet de entrenamiento reales 
hold on
plot(d) %DataSet de entrenamiento Simulados
legend('DataSet entrenamiento reales','DataSet entrenamiento simulados')
figure
 
%Errores
errorDataSet=t-a;%Error entre DataSet reales Vs Simulados
RMSEDataSet=sqrt(mse(t,a))
 
errortest =ttest-b; %Error entre DataSet de test reales Vs Simulados
RMSEtest=sqrt(mse(ttest,b))

errorval=tval-c;%Error entre DataSet de validaci�n reales Vs Simulados
RMSEval=sqrt(mse(tval,c))

errortrain=ttrain-d;%Error entre DataSet de entrenamiento reales Vs Simulados
RMSEtrain=sqrt(mse(ttrain,d))

%Pruebas PACF
parcorr(errorDataSet)
legend('PACF Error DataSet')
figure

parcorr(errortest)
legend('PACF Error Test')
figure

parcorr(errorval)
legend('PACF Error Validaci�n')
figure

parcorr(errortrain)
legend('PACF Error Entrenamiento')
figure
 
%Histogramas
 
histfit(errorDataSet)
legend('Error DataSet')
figure

histfit(errortest)
legend('Error Test')
figure

histfit(errorval)
legend('Error Validaci�n')
figure

histfit(errortrain)
legend('Error Entrenamiento')
figure  

%Pruebas de normalidad

NDataSet=jbtest(errorDataSet)
Ntest=jbtest(errortest)
Nval=jbtest(errorval)
Ntrain=jbtest(errortrain)

[H,pvalor,Q,CV]=lbqtest(errortest,'lags',[10 15 20],'Alpha',0.05)%Ljung-Box Q-Test

%DataSet para validaci�n cruzada

VAL=p(:,58);% DataSet de la fila (Poner cualquier valor que no supere el n�mero de filas de la base de DataSet original)
RVAL=t(58);
 
%Pesos de las matrices y Bias

IW1=net.IW{1,1};			%pesos capa entrada � capa oculta 0 a 1
LW2=net.LW{2,1};			%pesos capa oculta 1 a la 2 a capa de salida
 
B1=net.b{1,1};			    %pesos del Bias a capa oculta 1
B2=net.b{2,1};              %pesos del Bias a capa salida 
   
[H,K]=size(IW1);

for h=1:H
    for k=1:K
        D(h,k)=(IW1(h,k) - VAL(k)).^2; %Diferencia de los pesos y los DataSet de entrada
        DE(h)=sqrt(sum(D(h,:)));% Distancia Euclidea
        Net1(h)=DE(h)*B1(h);% Neto 1
    end
end
 
for h=1:H
    A(h)=exp(-Net1(h)^2);%Funci�n de activaci�n Gaussiana
end
  
Net2=dot(LW2,A)+B2;% Neto 2; Funci�n lineal (Debe ser muy aproximada a RVAL) 

Sol=[RVAL Net2] %Valor real Vs. Valor estimado
view(net)
 
 
