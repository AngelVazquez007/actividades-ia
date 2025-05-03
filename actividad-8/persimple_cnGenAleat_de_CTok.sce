//Jesus Alejandro Flores Hernandez 
//codigo en SCILAB
//Ejemplode perceptron simple
//Probado y revisado 22-sep 2016
//Genera unn conjunto de numeros dentro de los limites del cuadro
//(-1,1),(1,1),(1,-1),(-1,-1)
//Traza una recta y-2x=0
//Clasifica los puntos segun la recta (este sera el conjunto de trabajo)
//a continuacion aplica el algoritmo de perceptron
//referencia:
//http://programming-with-scilab.blogspot.mx/2012/04/perceptron-using-scilab.html

////////////// Genera el conjunto de trabajo /////////////
//////////////////////////////////////////////////////////
nct=20; //tamaño del conjunto de trabajo
x=2*rand(2,nct)-1;
x1=x(1,:);//Arreglo x1 contiene coordenadas x
y1=x(2,:);//Arreglo y1 contiene coordenadas y
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
//los puntos tienen la misma x, solo hay que calcular la y del punto
//y la y de la recta y restar,
//clasificamos segun el resultado
for i=1:nct
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
///// fin de generacion de conjunto de trabajo ////

//////Algoritmo de perceptron /////////////////////
///////////////////////////////////////////////////
//Calculamos la salida del perceptron con los pesos
//actuales para todos los puntos g(x)=w1*x1+w2*y1
for i=1:nct
    g(i)=sign(w(2)*y1(i)+w(1)*x1(i));
end
//~= operador relacionl es distonto que
//buscamos las salidas que no concuerdan con nuestra
//clasificación
ind = find(g ~= class_F);
iter=1;
while ~isempty(ind) //si es vacio => fin
    //actualizamos los pesos w
    //for i=1:nct
     //operador .* es multiplicación de matrices elemento a elemento
     w(1) = w(1) + (class_F(ind(1))-g(ind(1))).*x1(ind(1));  // 
     w(2) = w(2) + (class_F(ind(1))-g(ind(1))).*y1(ind(1));  // 
    //end
    //temp = sign(w'*x);
    for i=1:nct
         temp(i)=sign(w(2)*y1(i)+w(1)*x1(i));
    end
    ind = find(temp ~= class_F);
    iter = iter + 1;
    g=temp;
end
for i=1:100
    y3(i)=(-w(1)*x2(i))/w(2);
    //y3(i)=-(w(2)/w(1))*x2(i);
end
plot(x2,y3,'g');
mfprintf(6,'- - Al final los pesos son:\nvector w - - -'); //escribe a consola
disp(w);

