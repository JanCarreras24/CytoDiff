import matplotlib.pyplot as plt
import numpy as np

# Datos
synthetic_counts = list(range(0, 1100, 100))
synthetic_counts.append(2000)
synthetic_counts.append(3000)
accuracies = [
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

# Convertir accuracy a porcentaje
accuracies_pct = [a * 100 for a in accuracies]

# Crear eje X categórico
labels = [str(x) for x in synthetic_counts]
x_pos = np.arange(len(labels))

# Crear figura con tamaño y resolución adecuados
plt.figure(figsize=(10, 6), dpi=300)

# Plot con estilo profesional
plt.plot(
    x_pos,
    accuracies_pct,
    marker='o',
    markersize=8,
    linewidth=2.5,
    color='royalblue',
    label='Accuracy'
)

# Etiquetas en cada punto
for x, y in zip(x_pos, accuracies_pct):
    plt.text(x, y + 1, f'{y:.1f}%', ha='center', fontsize=10, fontfamily='serif')

# Personalización de título y ejes con tipografía serif
plt.title('Accuracy Evolution with Increasing Synthetic Images', fontsize=18, weight='bold', family='serif')
plt.xlabel('Synthetic Images per Class', fontsize=14, family='serif')
plt.ylabel('Accuracy (%)', fontsize=14, family='serif')

# Ajustar ticks categóricos
plt.xticks(x_pos, labels, rotation=45, fontsize=12, family='serif')
plt.yticks(fontsize=12, family='serif')

# Cuadrícula suave
plt.grid(True, linestyle='--', alpha=0.5)

# Añadir borde a los ejes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# Ajuste del layout para evitar recortes
plt.tight_layout()

# Guardar figura
plt.savefig('accuracy_vs_synthetic3000.pdf')
plt.savefig('accuracy_vs_synthetic3000.png', dpi = 300)

# Mostrar gráfico
plt.show()
