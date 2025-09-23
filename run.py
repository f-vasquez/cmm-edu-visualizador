#!/usr/bin/env python3
"""
Script de ejecución para CMM-EDU Visualizador
Inicia la aplicación automáticamente con la configuración correcta
"""

import os
import sys
import subprocess
import platform

def check_virtual_env():
    """Verificar si estamos en un entorno virtual"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def check_requirements():
    """Verificar si las dependencias están instaladas"""
    try:
        import streamlit
        import pandas
        import numpy
        import plotly
        import sklearn
        import umap
        return True
    except ImportError as e:
        print(f"❌ Dependencia faltante: {e}")
        return False

def run_streamlit():
    """Ejecutar la aplicación Streamlit"""
    print("🚀 Iniciando CMM-EDU Visualizador...")
    
    # Configurar variables de entorno para Streamlit
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = 'localhost'
    
    try:
        # Ejecutar Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al ejecutar la aplicación: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Aplicación cerrada por el usuario")

def main():
    """Función principal"""
    print("🎓 CMM-EDU Visualizador")
    print("=" * 30)
    
    # Verificar archivo de datos
    if not os.path.exists("data/capitulos_keywords_with_embeddings.csv"):
        print("⚠️  Archivo de datos no encontrado")
        print("Por favor ejecuta primero: python setup.py")
        print("Y coloca tu archivo CSV en data/")
        sys.exit(1)
    
    # Verificar entorno virtual (recomendado pero no obligatorio)
    if not check_virtual_env():
        print("⚠️  No estás en un entorno virtual")
        print("Recomendado: activa tu entorno virtual primero")
        response = input("¿Continuar de todos modos? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Verificar dependencias
    if not check_requirements():
        print("❌ Dependencias faltantes")
        print("Ejecuta: pip install -r requirements.txt")
        sys.exit(1)
    
    print("✅ Todo listo!")
    print("📊 La aplicación se abrirá en http://localhost:8501")
    print("Press Ctrl+C para cerrar la aplicación")
    print("-" * 30)
    
    # Ejecutar aplicación
    run_streamlit()

if __name__ == "__main__":
    main() 