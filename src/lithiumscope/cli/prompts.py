def choose(prompt: str, allowed: set[str]) -> str:
    while True:
        try:
            value = input(prompt).strip()
        except EOFError:
            return "0"
        if value in allowed:
            return value
        print("[ADVERTENCIA] Opción no válida. Seleccione una de las opciones mostradas.")
