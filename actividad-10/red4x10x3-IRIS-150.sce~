// Autor: Jflores
// Marzo 2025
// Red neuronal 4×10×3 que lee el dataset Iris desde iris.csv
// y muestra predicciones para todas las muestras.

// -----------------------------------------------------------------------------
// 1. Parámetros de la red
n_entradas    = 4;
n_ocultas     = 10;
n_salidas     = 3;

// -----------------------------------------------------------------------------
// 2. Cargar datos desde CSV
//    El CSV tiene 1 línea de encabezado + 150 datos, leemos filas 2 a 151.

// 2.1. Leer las 4 columnas numéricas (filas 2–151, columnas 1–4)
X = csvRead("iris.csv", ",", ".", "double", [], [], [2 1 151 4], 0);

// 2.2. Leer la columna de especies (filas 2–151, columna 5) como cadenas
species = csvRead("iris.csv", ",", ".", "string", [], [], [2 5 151 5], 0);

// Número de muestras realmente cargadas
n_num_dat_ent = size(X, 1);  // debería ser 150

// -----------------------------------------------------------------------------
// 3. One‑hot encoding de las especies
Y = zeros(n_num_dat_ent, n_salidas);
for i = 1:n_num_dat_ent
    select species(i)
    case "setosa"     then Y(i,:) = [1 0 0];
    case "versicolor" then Y(i,:) = [0 1 0];
    case "virginica"  then Y(i,:) = [0 0 1];
    end
end

// -----------------------------------------------------------------------------
// 4. Funciones de activación
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
endfunction

function y = sigmoid_derivada(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
endfunction

// -----------------------------------------------------------------------------
// 5. Inicializar pesos y sesgos aleatoriamente
W1 = rand(n_entradas, n_ocultas);
b1 = rand(1, n_ocultas);
W2 = rand(n_ocultas, n_salidas);
b2 = rand(1, n_salidas);

// -----------------------------------------------------------------------------
// 6. Muestra un pequeño vistazo de los datos
disp("Primeras 5 muestras (X | Y):");
disp([X(1:min(5,n_num_dat_ent),:), Y(1:min(5,n_num_dat_ent),:)]);

// -----------------------------------------------------------------------------
// 7. Entrenamiento por backpropagation
tasa_aprendizaje = 0.1;
max_iter         = 1000;

for iter = 1:max_iter
    // Forward pass
    Z1 = X * W1 + repmat(b1, n_num_dat_ent, 1);
    A1 = sigmoid(Z1);
    Z2 = A1 * W2 + repmat(b2, n_num_dat_ent, 1);
    A2 = sigmoid(Z2);

    // Cálculo del error
    error = Y - A2;

    // Backward pass
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

// -----------------------------------------------------------------------------
// 8. Prueba de la red y mostrar predicciones
Y_pred  = sigmoid(sigmoid(X * W1 + repmat(b1, n_num_dat_ent,1)) * W2 + repmat(b2, n_num_dat_ent,1));
Y_class = fix(Y_pred + 0.5);  // redondea a 0 o 1

disp("Predicciones para todas las muestras (X | Y_predicted):");
disp([X, Y_class]);
