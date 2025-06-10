import matplotlib.pyplot as plt
import numpy as np

# Datos
synthetic_counts = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]
accuracies_mean = [
    0.6182809297916226,
    0.6190116891872643,
    0.6488391936009108,
    0.6514513536955135,
    0.6702765948044569,
    0.6822441378202794,
    0.6903042794246943,
    0.6999159450409883,
    0.694553592116977,
    0.7065764225991621,
    0.7072692936646094,
    0.7211815297563623,
    0.7339364186227905
]

accuracies_std = [
    0.03467290540664273,
    0.021095332982992945,
    0.016322664907427183,
    0.009376417891659192,
    0.016767947522360976,
    0.013009333306563967,
    0.01714413352376367,
    0.01272067309970901,
    0.014092168481134032,
    0.007975332376317833,
    0.008725780810516568,
    0.005967368414078239,
    0.012515932306476168
]

# Pasar a porcentaje
accuracies_mean_pct = [a * 100 for a in accuracies_mean]
accuracies_std_pct = [s * 100 for s in accuracies_std]

# Convertir eje X a categórico
labels = [str(c) for c in synthetic_counts]
x_pos = np.arange(len(labels))

# Crear figura con tamaño y resolución adecuados
plt.figure(figsize=(10, 6), dpi=300)

# Plot media con barras de error
plt.errorbar(
    x_pos,
    accuracies_mean_pct,
    yerr=accuracies_std_pct,
    fmt='-o',
    ecolor='gray',
    elinewidth=2,
    capsize=5,
    capthick=2,
    markersize=8,
    linewidth=2.5,
    color='royalblue',
    label='Mean Accuracy ± Std Dev'
)

# Etiquetas en cada punto con media
for x, y in zip(x_pos, accuracies_mean_pct):
    plt.text(x, y + 1.5, f'{y:.1f}%', ha='center', fontsize=10, family='serif')

# Título y etiquetas de ejes
plt.title('Accuracy Evolution with Increasing Synthetic Images', fontsize=18, weight='bold', family='serif')
plt.xlabel('Synthetic Images per Class', fontsize=14, family='serif')
plt.ylabel('Accuracy (%)', fontsize=14, family='serif')

# Ajuste de ticks categóricos
plt.xticks(x_pos, labels, rotation=45, fontsize=12, family='serif')
plt.yticks(fontsize=12, family='serif')

# Cuadrícula suave
plt.grid(True, linestyle='--', alpha=0.5)

# Bordes de los ejes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Ajuste del layout
plt.tight_layout()

# Guardar figura
plt.savefig('accuracy_vs_synthetic_with_std3000.pdf')
plt.savefig('accuracy_vs_synthetic_with_std3000.png', dpi = 300)

# Mostrar gráfico
plt.show()



''''
clip 

accuracies_mean = [
    0.6182809297916226,
    0.6190116891872643,
    0.6488391936009108,
    0.6514513536955135,
    0.6702765948044569,
    0.6822441378202794,
    0.6903042794246943,
    0.6999159450409883,
    0.694553592116977,
    0.7065764225991621,
    0.7072692936646094,
    0.7211815297563623,
    0.7339364186227905
]

accuracies_std = [
    0.03467290540664273,
    0.021095332982992945,
    0.016322664907427183,
    0.009376417891659192,
    0.016767947522360976,
    0.013009333306563967,
    0.01714413352376367,
    0.01272067309970901,
    0.014092168481134032,
    0.007975332376317833,
    0.008725780810516568,
    0.005967368414078239,
    0.012515932306476168
]









resnet

accuracies_mean = [
    0.2726940478967853,
    0.4867683970968527,
    0.5215683241796583,
    0.5308239825480681,
    0.5185863830055869,
    0.551834288499353,
    0.5846474175137596,
    0.5991266717349155,
    0.615995203193535,
    0.656909045940309,
    0.64083565156979,
    0.7241791744194059,
    0.7497313949974151
]
accuracies_std = [
    0.09607820319306995,
    0.047480723230958005,
    0.025358006676385117,
    0.030059947804607252,
    0.014688160873879051,
    0.03703234811424313,
    0.019635385690969127,
    0.026269586024404167,
    0.013648427286025657,
    0.025424145418804403,
    0.024204695984242762,
    0.009565161557851358,
    0.014837818739626198
]
'''