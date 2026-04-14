import os
import subprocess
import sys
import time


def run_script(script_path, description):
    print(f"\n" + "="*60)
    print(f"🚀 INICIANDO: {description}")
    print(f"📦 Usando intérprete: {sys.executable}")
    print("="*60 + "\n")

    # Configuramos el PYTHONPATH para que reconozca la carpeta 'models'
    custom_env = os.environ.copy()
    custom_env["PYTHONPATH"] = os.getcwd()

    start_time = time.time()

    # Ejecutamos usando el mismo Python que lanza este script
    result = subprocess.run(
        [sys.executable, "-m", script_path],
        env=custom_env,
        capture_output=False  # Para que veas el progreso en tiempo real
    )

    end_time = time.time()
    duration = (end_time - start_time) / 60

    if result.returncode == 0:
        print(f"\n✅ FINALIZADO: {description}")
        print(f"⏱️  Tiempo transcurrido: {duration:.2f} minutos")
    else:
        print(f"\n❌ ERROR CRÍTICO en {description}.")
        print(
            f"El proceso se detuvo con código de salida: {result.returncode}")
        sys.exit(1)


def main():
    # 1. Asegurar estructura de carpetas
    for d in ['data', 'checkpoints', 'outputs']:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"📁 Carpeta creada: {d}")

    print("\n" + "🤖" * 15)
    print("  ORQUESTADOR DE ENTRENAMIENTO GLOBAL")
    print("  (Sin entorno virtual)")
    print("🤖" * 15 + "\n")

    # 2. Secuencia de entrenamiento completa
    # Fase 1: CNN
    run_script("training.train_cnn", "Entrenamiento de la CNN (Clasificador)")

    # Fase 2: LSTM
    run_script("training.train_lstm", "Entrenamiento de la LSTM (Texto)")

    # Fase 3: GAN
    run_script("training.train_gan", "Entrenamiento de la GAN (Generador)")

    print("\n" + "🌟"*20)
    print("¡PROCESO COMPLETADO EXITOSAMENTE!")
    print("Lanza la interfaz con: python main_app.py")
    print("🌟"*20)


if __name__ == "__main__":
    main()
