/////////////////////////////////////////////
// Red Neuronal para Afecciones Cardiacas
// Basado en el ejemplo de la flor Iris, adaptado para
// el archivo "reprocessed.hungarian.data"
// Se eliminan filas con datos faltantes (-9)
// La etiqueta se convierte a binaria: 0 = saludable, 1 = enfermo
/////////////////////////////////////////////

// 1. Cargar y Preprocesar los Datos
// Leer el archivo (se asume que está en el directorio actual)
// El archivo tiene 14 columnas: 13 atributos y 1 etiqueta
data = read('reprocessed.hungarian.data', -1, 14);

// Reemplazar los valores -9 por %nan para facilitar el filtrado
data(data == -9) = %nan;

// Eliminar filas que tengan %nan en cualquiera de las primeras 13 columnas
valid_rows = [];
for i = 1:size(data,1)
    // Si en las primeras 13 columnas NO hay %nan, se guarda el índice
    if sum(isnan(data(i,1:13))) == 0 then
        valid_rows = [valid_rows; i];
    end
end
data = data(valid_rows, :);

// Verificar que ya no queden %nan en los atributos
if sum(isnan(data(:,1:13))) > 0 then
    error("Aún existen valores %nan en los datos. Verifica el archivo de entrada.");
end

// Separar atributos (X) y etiquetas (y_raw)
X = data(:, 1:13);   // 13 atributos
y_raw = data(:, 14); // Etiqueta original

// Convertir la etiqueta a binaria: 0 => 0 (saludable); cualquier otro valor => 1 (enfermo)
y = zeros(size(y_raw,1), 1);
for i = 1:size(y_raw,1)
    if y_raw(i) == 0 then
        y(i) = 0;
    else
        y(i) = 1;
    end
end

// Usaremos todos los datos para entrenamiento
n_num_dat_ent = size(X,1);

// Normalizar cada columna de X para tener media 0 y desviación estándar 1
for j = 1:size(X,2)
    X(:,j) = (X(:,j) - mean(X(:,j))) / stdev(X(:,j));
end

// 2. Definir la Red Neuronal
// Parámetros de la red
n_entradas = 13;   // 13 atributos
n_ocultas  = 10;    // Número de neuronas en la capa oculta (ajusta según consideres)
n_salidas  = 1;     // Una salida (sigmoide para clasificación binaria)

// Función de activación sigmoide
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
endfunction

// Derivada de la sigmoide
function y = sigmoid_derivada(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
endfunction

// Inicializar pesos y sesgos con valores pequeños
W1 = 0.01 * rand(n_entradas, n_ocultas);  // [13 x 10]
b1 = 0.01 * rand(1, n_ocultas);           // [1 x 10]
W2 = 0.01 * rand(n_ocultas, n_salidas);    // [10 x 1]
b2 = 0.01 * rand(1, n_salidas);            // [1 x 1]

// 3. Entrenamiento de la Red Neuronal
tasa_aprendizaje = 0.01; // Tasa de aprendizaje reducida para estabilidad
max_iter = 1000;

for iter = 1:max_iter
    // Propagación hacia adelante
    b1_exp = repmat(b1, n_num_dat_ent, 1); // [n_num_dat_ent x 10]
    Z1 = X * W1 + b1_exp;                  // [n_num_dat_ent x 10]
    A1 = sigmoid(Z1);                      // Activación de la capa oculta
    
    b2_exp = repmat(b2, n_num_dat_ent, 1);  // [n_num_dat_ent x 1]
    Z2 = A1 * W2 + b2_exp;                 // [n_num_dat_ent x 1]
    A2 = sigmoid(Z2);                      // Salida de la red
    
    // (Opcional) Monitorear activaciones
    // disp("Max A1: " + string(max(abs(A1))));
    // disp("Max A2: " + string(max(abs(A2))));
    
    // Calcular el error (diferencia entre etiqueta real y salida)
    error = y - A2;
    
    // Retropropagación
    dZ2 = error .* sigmoid_derivada(Z2);
    dW2 = A1' * dZ2;
    db2 = sum(dZ2, 1);
    
    dZ1 = (dZ2 * W2') .* sigmoid_derivada(Z1);
    dW1 = X' * dZ1;
    db1 = sum(dZ1, 1);
    
    // Actualizar pesos y sesgos
    W2 = W2 + tasa_aprendizaje * dW2;
    b2 = b2 + tasa_aprendizaje * db2;
    W1 = W1 + tasa_aprendizaje * dW1;
    b1 = b1 + tasa_aprendizaje * db1;
end

// 4. Probar la Red Neuronal
// Calcular la salida final de la red
b1_exp = repmat(b1, n_num_dat_ent, 1);
b2_exp = repmat(b2, n_num_dat_ent, 1);
Y_pred = sigmoid(sigmoid(X * W1 + b1_exp) * W2 + b2_exp);

// Convertir las predicciones a 0 o 1 (umbral 0.5)
Y_pred_bin = Y_pred > 0.5;

// Mostrar ejemplos de predicción (atributos | etiqueta real | predicción)
disp("Ejemplo de predicciones (entradas | etiqueta real | predicción):")
disp(cat(2, X, y, Y_pred_bin));

// Calcular la precisión del modelo
accuracy = sum(Y_pred_bin == y) / n_num_dat_ent;
disp("Precisión del modelo: " + string(accuracy));
