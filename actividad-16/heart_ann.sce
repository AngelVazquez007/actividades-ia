// heart_ann.sce – Clasificación de Enfermedad Cardíaca con ANN Toolbox en Scilab

// 0. (Sólo la primera vez) Instalar y cargar el ANN Toolbox
//--> atomsInstall("ANN_Toolbox");
//--> atomsLoad("ANN_Toolbox");

///////////////////////////////////////////////////////
// 1. Leer datos
///////////////////////////////////////////////////////
disp("1. Leyendo datos...");
M_str = csvRead("reprocessed.hungarian.data", " ", [], "string");
// Convertimos todas las columnas a numérico
M_num = evstr(M_str);
// Separamos X (1–13) y y (14)
X_raw = M_num(:,1:13)';
y_raw = M_num(:,14)';

[nFeat, nSamp] = size(X_raw);

///////////////////////////////////////////////////////
// 2. Tratamiento de missing (-9) e imputación por media
///////////////////////////////////////////////////////
disp("2. Tratamiento de missing (-9) por NaN e imputación por media...");
X_raw(X_raw == -9) = %nan;
for i = 1:nFeat
    col = X_raw(i,:);
    mu_col = mean(col(~isnan(col)));
    col(isnan(col)) = mu_col;
    X_raw(i,:) = col;
end

///////////////////////////////////////////////////////
// 3. One‑hot encoding de variables categóricas
///////////////////////////////////////////////////////
disp("3. One‑hot encoding...");
function P_enc = onehot(col, levels)
    m = length(col);
    k = length(levels);
    P_enc = zeros(k, m);
    for j = 1:k
        P_enc(j, col == levels(j)) = 1;
    end
end

// Variables que ya son numéricas/binarizadas
P_age    = X_raw(1,:);           // A1: edad
P_bin_sex= X_raw(2,:);           // A2: sexo
P_trest  = X_raw(4,:);           // A4: trestbps
P_chol   = X_raw(5,:);           // A5: chol
P_fbs    = X_raw(6,:);           // A6: fbs
P_thalch = X_raw(8,:);           // A8: thalach
P_exang  = X_raw(9,:);           // A9: exang
P_oldp   = X_raw(10,:);          // A10: oldpeak

// One‑hot para los demás
P_cp     = onehot(X_raw(3,:), [1 2 3 4]);   // A3: cp
P_ecg    = onehot(X_raw(7,:), [0 1 2]);     // A7: restecg
P_slope  = onehot(X_raw(11,:), [1 2 3]);    // A11: slope
P_ca     = onehot(X_raw(12,:), [0 1 2 3]);  // A12: ca
P_thal   = onehot(X_raw(13,:), [3 6 7]);    // A13: thal

// Concatenamos todas las características en una matriz P
P = [ P_age;
      P_bin_sex;
      P_cp;
      P_trest;
      P_chol;
      P_fbs;
      P_ecg;
      P_thalch;
      P_exang;
      P_oldp;
      P_slope;
      P_ca;
      P_thal ];

// 4. Normalización (tras calcular Pn)…
Pn = (P - mu_vec * ones(1,nSamp)) ./ (sigma_vec * ones(1,nSamp));

// Comprobación de NaN usando sum en lugar de any
if sum(Pn(:) == %nan) > 0 then
    error("¡He encontrado NaN en Pn tras la normalización!");
end


///////////////////////////////////////////////////////
// 5. Preparar target T (clasificación binaria)
///////////////////////////////////////////////////////
disp("5. Codificando targets (0 = sano, 1 = enfermo)...");
T = zeros(2, nSamp);
T(1, y_raw == 0) = 1;   // sano
T(2, y_raw > 0)  = 1;   // enfermo

///////////////////////////////////////////////////////
// 6. Configuración y entrenamiento de la red
///////////////////////////////////////////////////////
disp("6. Inicializando y entrenando la red...");
N       = [ size(Pn,1), 10, 2 ];   // [#entradas, #ocultas, #salidas]
params  = [ 0.05, 0 ];             // [learning_rate, momentum]
epochs  = 200;

W = ann_FF_init(N);
for e = 1:epochs
    W = ann_FF_Std_batch(Pn, T, N, W, params, 1);
    if modulo(e,20) == 0 then
        Ytmp = ann_FF_run(Pn, N, W);
        sse  = sum(sum((T - Ytmp).^2));
        disp("  >> Época " + string(e) + " / " + string(epochs) + "  SSE = " + string(sse));
    end
end

///////////////////////////////////////////////////////
// 7. Evaluación final
///////////////////////////////////////////////////////
disp("7. Evaluando desempeño...");
Y = ann_FF_run(Pn, N, W);
[ maxVals, pred ] = max(Y, 'r');
// pred == 1 → sano, pred == 2 → enfermo
labels    = (pred == 1);
true_lbls = (y_raw == 0);
accuracy  = sum(labels == true_lbls) / nSamp * 100;
disp("Exactitud total: " + string(accuracy) + "%");
