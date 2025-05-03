// RN-CARDIACO-MASK.sce
// Red neuronal con masking para datos de cardiopatía

// 1) Cargar y convertir -9 → NaN
data = csvRead("reprocessed.hungarian.data", " ");
data(data == -9) = %nan;

// 2) Separar X_raw y Y_raw
X_raw = data(:, 1:13);
Y_raw = data(:, 14);

// 3) Máscara M y X_temp
M      = ~isnan(X_raw);   
X_temp = X_raw;           
X_temp(~M) = 0;           

// 4) Normalización con repmat
[n, d] = size(X_temp);
mu     = sum(X_temp, "r") ./ sum(M, "r");
diff2  = (X_temp - repmat(mu, n, 1)) .* M;
sigma  = sqrt(sum(diff2.^2, "r") ./ sum(M, "r"));
Xn     = (X_temp - repmat(mu, n, 1)) ./ repmat(sigma, n, 1);

// 5) Preparar P y T
P = Xn';
classes     = unique(Y_raw);
num_classes = length(classes);
N           = size(P, 2);
T = zeros(num_classes, N);
for j = 1:N
    idx       = find(classes == Y_raw(j));
    T(idx, j) = 1;
end

// 6) Entrenamiento
layers = [13 10 num_classes];
W      = ann_FFBP_gd(P, T, layers);

// 7) Forward con masking
function A_out = forward_masked(P, W, M)
    A    = P;
    M_t  = M';
    for i = 1:length(W)
        A = W(i).w * (A .* M_t) + repmat(W(i).b, 1, size(A,2));
        A = 1 ./ (1 + exp(-A));
    end
    A_out = A;
endfunction

Y_pred = forward_masked(P, W, M);

// 8) Predicción y precisión
[Mv, K]    = max(Y_pred, "r");
predicted  = classes(K)';
accuracy   = sum(predicted == Y_raw) / N * 100;
disp("Precisión con masking: " + string(accuracy) + "%");

// 9) Ejemplos
disp("Reales → Predichos:");
for i = 1:min(10,N)
    printf("%d → %d\n", Y_raw(i), predicted(i));
end
