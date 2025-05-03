// Función de activación sigmoide
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
endfunction

// Derivada de la función sigmoide
function y = sigmoid_derivative(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
endfunction

// Inicialización de pesos y sesgos
function [W1, b1, W2, b2] = initialize_parameters(input_size, hidden_size, output_size)
    W1 = rand(hidden_size, input_size, "normal") * 0.01;
    b1 = zeros(hidden_size, 1);
    W2 = rand(output_size, hidden_size, "normal") * 0.01;
    b2 = zeros(output_size, 1);
endfunction

// Propagación hacia adelante (forward propagation)
function [Z1, A1, Z2, A2] = forward_propagation(X, W1, b1, W2, b2)
    b1_expanded=repmat(b1, 1,4);
    Z1 = W1 * X + b1_expanded;
    A1 = sigmoid(Z1);
    //b2_expanded=repmat(b2, n_num_dat_ent,1);
    Z2 = W2 * A1 + b2;
    A2 = sigmoid(Z2);
endfunction

// Propagación hacia atrás (backpropagation)
function [dW1, db1, dW2, db2] = backward_propagation(X, Y, Z1, A1, Z2, A2, W2)
    m = size(X, 2);  // Número de ejemplos

    // Cálculo de gradientes en la capa de salida
    dZ2 = A2 - Y;
    dW2 = (dZ2 * A1') / m;
    db2 = sum(dZ2, 2) / m;

    // Cálculo de gradientes en la capa oculta
    dZ1 = (W2' * dZ2) .* sigmoid_derivative(Z1);
    dW1 = (dZ1 * X') / m;
    db1 = sum(dZ1, 2) / m;
endfunction

// Actualización de parámetros
function [W1, b1, W2, b2] = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    W1 = W1 - learning_rate * dW1;
    b1 = b1 - learning_rate * db1;
    W2 = W2 - learning_rate * dW2;
    b2 = b2 - learning_rate * db2;
endfunction

//
n_num_dat_ent = 4;
// Datos de entrenamiento para la función AND
X = [0 0; 0 1; 1 0; 1 1]';  // Entradas (transpuesta para que cada columna sea un ejemplo)
Y = [0 1 1 1];               // Salidas esperadas

// Hiperparámetros
input_size = 2;    // Número de entradas
hidden_size = 2;   // Número de neuronas en la capa oculta
output_size = 1;   // Número de neuronas en la capa de salida
learning_rate = 0.1;
epochs = 2000;

// Inicialización de parámetros
[W1, b1, W2, b2] = initialize_parameters(input_size, hidden_size, output_size);

// Entrenamiento de la red
for i = 1:epochs
    // Propagación hacia adelante
    [Z1, A1, Z2, A2] = forward_propagation(X, W1, b1, W2, b2);

    // Propagación hacia atrás
    [dW1, db1, dW2, db2] = backward_propagation(X, Y, Z1, A1, Z2, A2, W2);

    // Actualización de parámetros
    [W1, b1, W2, b2] = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate);
end

// Probar la red con las entradas de entrenamiento
[Z1, A1, Z2, A2] = forward_propagation(X, W1, b1, W2, b2);
//Z2 da salida en -1, 1, (sign(Z2')+1)*0.5 lo canvierte a 0,1
disp("Salidas: Z salida de la red, Y salida crrecta")
disp("x1,  x2,  Z,   Y")
disp([X',(sign(Z2')+1)*0.5,Y']);
