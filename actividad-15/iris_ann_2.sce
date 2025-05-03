// iris_ann.sce – Clasificación Iris con ANN Toolbox en Scilab 2025

// 0. (Solo la primera vez) Instalar y cargar el ANN Toolbox
//--> atomsInstall("ANN_Toolbox");
//--> atomsLoad("ANN_Toolbox");

// 1. Leer y convertir datos
M        = csvRead("iris.csv", ",", [], "string");
disp("Datos leídos: " + string(size(M,1)-1) + " muestras, " + string(size(M,2)-1) + " columnas");
numData  = evstr(M(2:$,1:4));
species  = M(2:$,5);
disp("Primeras 5 muestras de datos numéricos:");
disp(numData(1:5,:));
disp("Primeras 5 etiquetas de especies:");
disp(species(1:5));

// 2. One‑hot encoding
disp("Clases únicas encontradas:");
classes    = unique(species);
disp(classes);
numClasses = size(classes, "r");
numSamples = size(numData, "r");
disp("Número de clases: " + string(numClasses) + ", Número de muestras: " + string(numSamples));
T   = zeros(numClasses, numSamples);
idx = zeros(1, numSamples);
for i = 1:numClasses
    rows      = vectorfind(species, classes(i), 'r');
    idx(rows) = i;
    T(i, rows) = 1;
end

// 3. Normalización de entradas
P     = numData';
mu    = mean(P, 2);
sigma = stdev(P, 'c');
disp("Media de cada feature:");
disp(mu');
disp("Desviación estándar de cada feature:");
disp(sigma');
P     = (P - mu * ones(1, numSamples)) ./ (sigma * ones(1, numSamples));
disp("Primeras 5 columnas normalizadas:");
disp(P(:,1:5)');

// 4. Configuración y entrenamiento de la red (batch + menor número de épocas)
N       = [4, 5, 3];            // 4 entradas, 5 ocultas, 3 salidas
lp      = [0.1, 0];             // [learning_rate, momentum]
Tepochs = 100;                  // épocas reducidas

disp("Inicializando red con arquitectura: " + string(N));
W = ann_FF_init(N);
disp("Entrenamiento por lotes: " + string(Tepochs) + " épocas");
// Bucle de entrenamiento para mostrar SSE cada 10 épocas
for e = 1:Tepochs
    W = ann_FF_Std_batch(P, T, N, W, lp, 1);
    if modulo(e,10) == 0 then
        Y_tmp = ann_FF_run(P, N, W);
        err  = sum((T - Y_tmp).^2);
        sse  = sum(err);
        disp("Época " + string(e) + ": SSE = " + string(sse));disp("Época " + string(e) + ": SSE = " + string(sse));;
    end
end

// 5. Prueba y evaluación
disp("Evaluando red entrenada...");
Y                 = ann_FF_run(P, N, W);
[maxVals, predIdx] = max(Y, 'r');
accuracy          = sum(predIdx == idx) / numSamples * 100;
disp("Exactitud total: " + string(accuracy) + "%");
