#!/usr/bin/env python3

import subprocess
import os
import shutil

def main():
    # 1. Установим/обновим gdown
    subprocess.run(["pip", "install", "--upgrade", "gdown"], check=True)

    # 2. Скачаем папку по ссылке Google Drive (ID: 1fbJsI2gpO8j9rURufUbGIUORkyDbt_S7)
    #    Параметры:
    #      --folder        -> указывает, что нужно скачать папку
    #      --remaining-ok  -> продолжать скачивание, даже если часть файлов уже скачана
    #      -O template_asr -> сохранить файлы в папку с именем template_asr
    subprocess.run([
        "gdown",
        "--folder",
        "1fbJsI2gpO8j9rURufUbGIUORkyDbt_S7",
        "--remaining-ok",
        "-O",
        "template_asr"
    ], check=True)


    if os.path.exists("template_asr"):
        print("Папка 'template_asr' успешно скачана.")
    else:
        print("Не удалось найти папку 'template_asr' после скачивания.")

if __name__ == "__main__":
    main()
