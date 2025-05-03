nct=20; //tamaño del conjunto de trabajo
x=2*rand(2,nct)-1;
x1=x(1,:); //Arreglo x1 contiene coordenadas x
y1=x(2,:); //Arreglo y1 contiene coordenadas y
plot(x1,y1,'*');

//Graficamos una linea arbitraria y-2x=0 f(x): y=2x
//Salvamos los coheficientes de x y y en el arreglo F
F=[1;-2];
//la función hipotesis es g(x,y)=w1*x+w2*y 
//pesos iniciales son w1=0 and w2=0
w=[0;0];

//mostramos área de trabajo para fines visuaes solamente
x2=linspace(-1,1,100);
for i=1:100
    y2(i)=2*x2(i);
end
plot(x2,y2,'r') //trazamos una línea roja

//Clasificamos los puntos a la derecha e izquierda de la linea
//los puntos tienen la misma x, solo hay que calcular la y de la recta
//y la y del punto y restar,
//clasificamos segun el resultado
for i=1:nct
    //y del punto: y=2*x(1) y la y de la recta: y1(i)
    l(i)=-F(2)*x1(i)-y1(i);//l(i)=F(2)*y1(i)+F(1)*x1(i);
    class_F(i)=sign(l(i));  
end

for i=1:nct
        if class_F(i)==1 then
            plot(x1(i),y1(i),'gre*');
        else
            plot(x1(i),y1(i),'blu*');    
        end
end
