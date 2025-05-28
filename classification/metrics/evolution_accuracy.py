import matplotlib.pyplot as plt

# Datos
synthetic_counts = list(range(0, 1100, 100))
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
    0.64083565156979
]

# Convertir accuracy a porcentaje
accuracies_pct = [a * 100 for a in accuracies]

# Crear figura con tamaño y resolución adecuados
plt.figure(figsize=(10, 6), dpi=300)

# Plot con estilo profesional
plt.plot(
    synthetic_counts,
    accuracies_pct,
    marker='o',
    markersize=8,
    linewidth=2.5,
    color='royalblue',
    label='Accuracy'
)

# Etiquetas en cada punto (opcional, para presentar)
for x, y in zip(synthetic_counts, accuracies_pct):
    plt.text(x, y + 1, f'{y:.1f}%', ha='center', fontsize=10, fontfamily='serif')

# Personalización de título y ejes con tipografía serif (más profesional)
plt.title('Accuracy Evolution with Increasing Synthetic Images', fontsize=18, weight='bold', family='serif')
plt.xlabel('Synthetic Images per Class', fontsize=14, family='serif')
plt.ylabel('Accuracy (%)', fontsize=14, family='serif')

# Ajustar ticks
plt.xticks(synthetic_counts, rotation=45, fontsize=12, family='serif')
plt.yticks(fontsize=12, family='serif')

# Cuadrícula suave
plt.grid(True, linestyle='--', alpha=0.5)

# Añadir borde a los ejes
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

# Ajuste del layout para que no se corten etiquetas
plt.tight_layout()

# Guardar figura en alta calidad (PDF para vectorial, PNG para raster)
plt.savefig('accuracy_vs_synthetic_images.pdf')  # PDF vectorial ideal para papers
plt.savefig('accuracy_vs_synthetic_images.png')  # O PNG a 300 dpi si prefieres

# Mostrar gráfico
plt.show()
