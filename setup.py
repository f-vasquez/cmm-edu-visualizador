#!/usr/bin/env python3
"""
Script de setup para CMM-EDU Visualizador
Facilita la instalación y configuración inicial del proyecto
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Ejecutar comando y manejar errores"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"Salida del error: {e.stderr}")
        return False

def check_python_version():
    """Verificar versión de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print("❌ Python 3.11 o superior es requerido")
        print(f"Versión actual: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detectado")
    return True

def create_virtual_environment():
    """Crear entorno virtual"""
    if os.path.exists("venv"):
        print("📦 Entorno virtual ya existe")
        return True
    
    return run_command("python -m venv venv", "Creando entorno virtual")

def activate_and_install():
    """Activar entorno virtual e instalar dependencias"""
    system = platform.system()
    
    if system == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Instalar dependencias
    return run_command(f"{pip_cmd} install -r requirements.txt", "Instalando dependencias")

def check_data_file():
    """Verificar si existe el archivo de datos"""
    data_file = "data/capitulos_keywords_with_embeddings.csv"
    if not os.path.exists(data_file):
        print("⚠️  Archivo de datos no encontrado")
        print(f"Por favor, coloca tu archivo CSV en: {data_file}")
        print("Estructura requerida: id,curso,numero,titulo,keywords,keywords_embedding")
        return False
    print("✅ Archivo de datos encontrado")
    return True

def create_directories():
    """Crear directorios necesarios"""
    directories = ["data", ".streamlit"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Directorio {directory} creado")

def main():
    """Función principal de setup"""
    print("🎓 CMM-EDU Visualizador - Setup")
    print("=" * 40)
    
    # Verificar Python
    if not check_python_version():
        sys.exit(1)
    
    # Crear directorios
    create_directories()
    
    # Crear entorno virtual
    if not create_virtual_environment():
        sys.exit(1)
    
    # Instalar dependencias
    if not activate_and_install():
        sys.exit(1)
    
    # Verificar datos
    data_exists = check_data_file()
    
    print("\n🎉 Setup completado!")
    print("=" * 40)
    print("Para ejecutar la aplicación:")
    
    system = platform.system()
    if system == "Windows":
        print("1. venv\\Scripts\\activate")
    else:
        print("1. source venv/bin/activate")
    
    print("2. streamlit run app.py")
    print("\nO simplemente ejecuta: python run.py")
    
    if not data_exists:
        print("\n⚠️  No olvides colocar tu archivo de datos en data/")

if __name__ == "__main__":
    main() 