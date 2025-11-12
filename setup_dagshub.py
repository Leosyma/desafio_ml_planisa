"""
Script para configurar DagsHub no projeto.
Execute este script após criar o repositório no DagsHub.
"""

import os
import subprocess
from pathlib import Path


def setup_dagshub():
    """Configura DagsHub e DVC no projeto."""
    
    print("🚀 Configurando DagsHub...")
    
    # Solicitar informações do usuário
    username = input("Digite seu usuário do DagsHub: ")
    repo_name = input("Digite o nome do repositório no DagsHub: ")
    
    repo_url = f"https://dagshub.com/{username}/{repo_name}.git"
    dvc_remote = f"https://dagshub.com/{username}/{repo_name}.dvc"
    
    print(f"\n📦 Configurando repositório: {repo_url}")
    
    # Inicializar DVC se não estiver inicializado
    if not Path('.dvc').exists():
        print("\n1. Inicializando DVC...")
        subprocess.run(['dvc', 'init'], check=True)
    
    # Configurar remote DVC
    print("\n2. Configurando remote DVC...")
    try:
        subprocess.run(['dvc', 'remote', 'add', 'origin', dvc_remote], check=True)
        subprocess.run(['dvc', 'remote', 'default', 'origin'], check=True)
    except subprocess.CalledProcessError:
        print("   Remote já configurado ou erro ao configurar. Continuando...")
    
    # Configurar git remote se não existir
    print("\n3. Configurando git remote...")
    try:
        result = subprocess.run(['git', 'remote', 'get-url', 'origin'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(['git', 'remote', 'add', 'origin', repo_url], check=True)
            print(f"   Git remote configurado: {repo_url}")
        else:
            print(f"   Git remote já existe: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"   Erro ao configurar git remote: {e}")
    
    # Criar arquivo .env.example se não existir
    env_example = Path('.env.example')
    if not env_example.exists():
        print("\n4. Criando .env.example...")
        env_example.write_text("""# DagsHub Configuration
DAGSHUB_USER_TOKEN=your_token_here

# MLflow Configuration
MLFLOW_TRACKING_URI=https://dagshub.com/{username}/{repo_name}.mlflow
""".format(username=username, repo_name=repo_name))
        print("   Arquivo .env.example criado. Configure suas credenciais!")
    
    print("\n✅ Configuração concluída!")
    print("\n📝 Próximos passos:")
    print("   1. Configure suas credenciais no arquivo .env")
    print("   2. Execute: dagshub login")
    print("   3. Execute: dvc push para enviar dados versionados")
    print("   4. Execute: git push para enviar código")


if __name__ == '__main__':
    setup_dagshub()

