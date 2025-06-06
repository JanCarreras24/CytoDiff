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


input_text = """[[399   7   3  59  19   8  54   0  33   5   0   2  11   0   0]
 [  0 517   0  38  21  13   5   0   4   0   1   0   1   0   0]
 [  5   0 358  88  27  12  28   3  64   2  13   0   0   0   0]
 [  0   1   0 531  61   0   4   3   0   0   0   0   0   0   0]
 [  0  11   0  38 531   7   1   1   9   0   1   0   1   0   0]
 [  0  32   0 113  17 403   1   0  22   0   4   0   7   1   0]
 [  0   2   0  88  15   2 459   6  20   0   1   0   0   3   4]
 [  0   7   0  72 117   2   0 396   3   0   3   0   0   0   0]
 [  0   7   2 151  53  39   5   1 311   0  26   0   4   1   0]
 [  1  17   0  58  11   3  62   9  18 412   1   0   1   0   7]
 [  0  10   1 165  48 123   3   0  76   0 170   0   2   2   0]
 [  5   2   0  85  22   1  20   0  18   0   1 430  16   0   0]
 [  1  28   0 100   6  12   0   0  39   0   0   0 413   1   0]
 [  0   5   0  50  20   5   1  11  30   0  33   0   2 443   0]
 [  0   0   0  45  29   0  14   1   1   0   4   0   0   0 506]]

"""

print(formato_numpy(input_text))
