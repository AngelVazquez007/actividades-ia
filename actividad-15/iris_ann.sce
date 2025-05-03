// iris_ann.sce – Clasificación Iris con ANN Toolbox en Scilab 2025

// 0. Instalar y cargar el Toolbox (ejecutar solo la primera vez)
//--> atomsInstall("ANN_Toolbox");     // instala ANN Toolbox :contentReference[oaicite:9]{index=9}
//--> atomsLoad("ANN_Toolbox");        // carga funciones ann_FFBP_gdx, ann_FFBP_run :contentReference[oaicite:10]{index=10}

// 1. Leer y convertir datos
M       = csvRead("iris.csv", ",", [], "string");
numData = evstr(M(2:$,1:4));          // 150×4 de double
species = M(2:$,5);                   // 150×1 de strings

// 2. One‑hot encoding corregido
classes    = unique(species);         // 3×1 vector de clases :contentReference[oaicite:11]{index=11}
numClasses = size(classes, "r");      // escalar = 3 :contentReference[oaicite:12]{index=12}
numSamples = size(numData, "r");      // escalar = 150 :contentReference[oaicite:13]{index=13}

T = zeros(numClasses, numSamples);    // matriz 3×150 :contentReference[oaicite:14]{index=14}
idx = zeros(1, numSamples);           // vector 1×150

for i = 1:numClasses
    rows      = vectorfind(species, classes(i), 'r');  // índices de muestras de la clase i :contentReference[oaicite:15]{index=15}
    idx(rows) = i;
    T(i, rows) = 1;
end

// 3. Normalización de entradas
P  = numData';                        
mu = mean(P, 2);
sigma = stdev(P, 0, 2);
P = (P - mu * ones(1, numSamples)) ./ (sigma * ones(1, numSamples));

// 4. Entrenamiento de la RN
N = [4, 5, 3];                        // 4 entradas, 5 ocultas, 3 salidas
W = ann_FFBP_gdx(P, T, N);            // entrenamiento con GDX :contentReference[oaicite:16]{index=16}

// 5. Prueba y evaluación
Y = ann_FFBP_run(P, W);              
[~, predIdx] = max(Y);                // índice de fila máxima (clase) :contentReference[oaicite:17]{index=17}
accuracy = sum(predIdx == idx) / numSamples * 100;
disp("Exactitud total: " + string(accuracy) + "%");
