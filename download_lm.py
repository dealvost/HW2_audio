import subprocess
import sys

def main():
    # 1) Устанавливаем через pip необходимые пакеты
    # Если нужно больше библиотек, добавьте их в список
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "wget"], check=True)
    
    # 2) Выполняем все требуемые команды через subprocess
    # wget 4-gram.arpa.gz
    subprocess.run("wget https://us.openslr.org/resources/11/4-gram.arpa.gz -O 4-gram.arpa.gz", shell=True, check=True)
    
    # gunzip 4-gram.arpa.gz
    subprocess.run("gunzip 4-gram.arpa.gz", shell=True, check=True)
    
    # wget librispeech-vocab.txt
    subprocess.run("wget https://us.openslr.org/resources/11/librispeech-vocab.txt -O librispeech-vocab.txt", shell=True, check=True)
    
    # Преобразование словаря в нижний регистр
    subprocess.run("cat librispeech-vocab.txt | tr '[:upper:]' '[:lower:]' > librispeech-vocab_lower.txt", shell=True, check=True)
    
    # Преобразование ARPA в нижний регистр
    subprocess.run("cat 4-gram.arpa | tr '[:upper:]' '[:lower:]' > 4-gram-lower.arpa", shell=True, check=True)

if __name__ == "__main__":
    main()
