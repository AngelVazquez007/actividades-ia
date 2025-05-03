// RN-CARDIACO.sce
// Autor: Jflores
// Actividad: red neuronal multicapa para datos de cardiopatía

// -----------------------------------------------------------------------------
// 1) Cargar y limpiar datos
data = csvRead("reprocessed.hungarian.data", " ");
data(data == -9) = %nan;
mask = ~any(isnan(data), 2);
data_clean = data(mask, :);

// -----------------------------------------------------------------------------
// 2) Separar características y etiquetas
X     = data_clean(:, 1:13);
Y_raw = data_clean(:, 14);

// -----------------------------------------------------------------------------
// 3) Normalizar características (media 0, varianza 1)
[Xn, mu, sigma] = mat2normal(A = X, type = "mean_stdev");

// -----------------------------------------------------------------------------
// 4) Preparar P y T (one‑hot)
P = Xn';
classes    = unique(Y_raw);        // ej. [0 1 2 3 4]
num_classes = length(classes);
N          = size(P, 2);
T = zeros(num_classes, N);
for j = 1:N
    idx = find(classes == Y_raw(j));
    T(idx, j) = 1;
end

// -----------------------------------------------------------------------------
// 5) Arquitectura y entrenamiento
layers = [13 10 num_classes];
W = ann_FFBP_gd(P, T, layers);

// -----------------------------------------------------------------------------
// 6) Predicción
Y_pred = ann_FFBP_run(P, W);

// -----------------------------------------------------------------------------
// 7) Obtener clase predicha
[M, K]     = max(Y_pred, "r");  // CORRECCIÓN: usar M y K, no ~
predicted  = classes(K)';       // índice → etiqueta original

// -----------------------------------------------------------------------------
// 8) Precisión y muestras
accuracy = sum(predicted == Y_raw) / N * 100;
disp("Precisión en entrenamiento: " + string(accuracy) + "%");

disp("Primeros 10 reales → predichos:");
for i = 1:min(10, N)
    printf("%d → %d\n", Y_raw(i), predicted(i));
end
