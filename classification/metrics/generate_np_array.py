def formato_numpy(texto):
    # Quitamos saltos de línea iniciales/finales y corchetes externos
    texto = texto.strip()
    if texto.startswith('['):
        texto = texto[1:]
    if texto.endswith(']'):
        texto = texto[:-1]
    # Separamos filas por salto de línea
    filas = texto.strip().split('\n')
    resultado = "np.array(["
    for i, fila in enumerate(filas):
        # Quitamos corchetes de cada fila y espacios extra
        fila = fila.strip()
        if fila.startswith('['):
            fila = fila[1:]
        if fila.endswith(']'):
            fila = fila[:-1]
        # Convertimos espacios múltiples a comas y formateamos números con espacios
        nums = fila.split()
        nums_formateados = [f"{int(n):4d}" for n in nums]
        resultado += "    [" + ", ".join(nums_formateados) + "]"
        if i != len(filas)-1:
            resultado += ",\n"
    resultado += "])"
    return resultado


input_text = """[[370   0  11  28   1  36 127   3   2  15   0   0   0   6   1]
 [  0 470   0  46  34  11  32   0   0   1   0   0   0   6   0]
 [  4   0 465  30   7  20  58   5   3   4   1   0   0   3   0]
 [  0   3   0 573   3   1  18   1   0   0   0   0   0   1   0]
 [  0   7   0  41 510  22  13   4   1   0   1   0   0   1   0]
 [  1   1   1  25  32 513  23   0   1   0   2   0   0   1   0]
 [  0   0   0  47   1   1 540   1   0   2   0   0   0   3   5]
 [  0   0   0  45   9   1   1 539   0   0   2   0   0   3   0]
 [  0   1   0  52  69   5  58   9 401   0   3   0   0   2   0]
 [  0   0   0  31   2   2  93   1   1 463   1   0   0   3   3]
 [  0   1   0  48  19   4  17   4   0   0 506   0   0   1   0]
 [ 27   1   1  44  23   8  60   2   2   3   0 423   6   0   0]
 [ 48   1   5  35  14  14  55   0   2   6   0   6 414   0   0]
 [  0   0   0  33   1   0   3   0   0   0   0   0   0 563   0]
 [  0   0   0  69   3   0  27   2   0   0   0   0   0   0 499]]
"""

print(formato_numpy(input_text))
