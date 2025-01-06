# HW2

## About

Этот репозиторий содержит файлы для обучения/инференса моделей обученных в рамках ДЗ2

## Установка и загрузка


1. Скопировать репозиторий в среду

   ```bash
   git clone https://github.com/dealvost/HW2_audio.git
   ```

2. Исполнить файл download.py:
   ```bash
   python HW2_audio/download.py
   ```


3. Исполнить файл download_lm.py:
   ```bash
   python HW2_audio/download_lm.py
   ```
4. Установить необходимые библиотеки
   ```bash
   pip install -r HW2_audio/requirements.txt
   ```
5. Запустить инференс модели с LM командой ниже. По результатам выполнения команды будут получены метрики test_CER_(LM) = 0.1841 test_WER_(LM) = 0.3368
   ```bash
   python HW2_audio/inference.py -cn inference \
   inferencer.from_pretrained="template_asr/saved/DeepSpeech2_clean360_2/model_best_wer.pth" \
   inferencer.save_path="inference_outputs_lm" \
   model=deepspeech2 
   ```
6. Запустить инференс модели на датасете train-other 500  с LM командой ниже. По результатам выполнения команды будут получены метрики test_CER_(LM) = 0.2894 test_WER_(LM) = 0.4693

   ```bash
   python HW2_audio/inference.py -cn inference_other \
   inferencer.from_pretrained="template_asr/saved/DeepSpeech2_trainother/model_best_wer.pth" \
   inferencer.save_path="inference_outputs_lm" \
   model=deepspeech2
   ```
7. Запустить инференс модели на датасете train-other 500  с LM с BPE токенизацией командой ниже. По результатам выполнения команды будет получена метрика Average WER beam+LM = 0.4791
   ```bash
   python HW2_audio/inference_with_lm_beam.py -cn inference_BPE
   ```


   Всего к данном отчету прикреплено 4 модели (на самом деле обучалось больше, но прикреплять все не имеет смысла)

   Отчет по модели DeepSpeech2_clean360_2: https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/f31rsvua?nw=nwusermarkoavro01

   Отчет по модели DeepSpeech2_trainother: https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/pbmwty22?nw=nwusermarkoavro01

   Отчет по модели DeepSpeech2_BPE_main:  https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/yue2u1sz?nw=nwusermarkoavro01

   Отчет по модели DeepSpeech2_BPE_finetune_1: https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/qnkkkno8?nw=nwusermarkoavro01 (неудачная попытка исполльзовать аугментации для повышения метрик относительно модели BPE_main)

## Ответы на вопросы:

### Как воспроизвести вашу модель? (например: обучите 50 эпох с конфигурацией train_1.yaml и 50 эпох с train_2.yaml)

1. Модель DeepSpeech2_clean360_2. Выполнить команду:
   ```bash
   python3 train.py -cn deepspeech2 writer.run_name="DeepSpeech2_clean360_2" trainer.override=True trainer.n_epochs=100 dataloader.batch_size=30
   ```
2. Модель DeepSpeech2_trainother. Выполнить команду:
   ```bash
   python3 train.py -cn finetune_deepspeech2 writer.run_name="DeepSpeech2_trainother" trainer.override=True trainer.n_epochs=100 dataloader.batch_size=30
   ```
3. Модель DeepSpeech2_BPE_main. Выполнить команду для токенизации:
   ```bash
   python /content/template_asr/train_bpe_tokenizer.py
   ```
   Затем команду:
   ```bash   
   python3 train.py -cn deepspeech2_BPE writer.run_name="DeepSpeech2_BPE_main" trainer.override=True trainer.n_epochs=100 dataloader.batch_size=40
   ```
 4. Модель  DeepSpeech2_BPE_finetune_1. Выполнить команду:
   ```bash   
   python3 train.py \
  -cn deepspeech2_BPE \
  writer.run_name="DeepSpeech2_BPE_finetune_1" \
  trainer.n_epochs=100 \
  trainer.override=True \
  dataloader.batch_size=40 \
  +trainer.from_pretrained="/template_asr/saved/DeepSpeech2_BPE_main/model_best_wer.pth"
   ```
### Прикрепите журналы обучения (logs) обученных моделей.
   
Журналы прикреплены.
   
### Как вы обучили свою окончательную модель?

Все файлы конфигураций .yaml приложены. 
   
### Что вы пробовали?

Аугментации, BPE-токенизацию, увеличение числа hidden_size параметров в модели, инференс с LM.
### Что сработало, а что нет?
Аугментации оказались бесполезны. Пробовал использовать их как для train-other 500 так и для train-clean 360, метрики становились только хуже. (вот один из отчетов с аугментациями для train clear 360: https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/hggc1iwa?nw=nwusermarkoavro01). Для train-other прикладывал выше.
BPE-токенизация не дала значительного прироста метрик (можно сравнить два отчета с моделью которая обучалась с BPE и без BPE для train-other, отчеты приложены выше)
Инференс с LM дает значительный прирост качества всегда.
Увеличение числа параметров в модели сработало и дало наилучшие метрики для train-clean 360. К сожалению у меня перезагрузилась среда в коллабе и я безвозвратно потерял эти модели (соответственно они не приложены к работе), но метрики там были самые высокие. У меня сохранился отчет в wandb: https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/o7xxh3bt?nw=nwusermarkoavro01. Если бы эту модель я заинференсил с LM, то получилось бы пробить WER 30 на test clean 100%, но вот так вот :(((

### Также прикрепите краткий обзор всех бонусных задач, которые вы реализовали.

Реализован инференс с LM как для обычной токенизации так и для BPE.

Реализована BPE токенизация.

Произведен sanity check (обучены идентичные модели с двумя разными токенизациями, для обеих произведен инференс с LM). Все файлы приложены.

### Что бы я сделал, если бы было больше времени и вычислительных ресурсов.

Я обучил бы conformer и подкрутил побольше hidden size параметров. И сделал бы инференс с LM и BPE токенизацию. К сожалению, много времени ушло на попытки затьюнить метрики с помощью аугментаций, которые в моем случае были полностью бесполезны. Это хороший урок на будущее: лучше сразу брать самую сильную SOTA модель, даже если она "тяжелая", чем пытаться моделью полегче выжать обходными приемами вменяемые метрики. В конечном счете на все попытки с аугментациями ушло больше вычислительных ресурсов, чем если бы я сразу взял conformer и обучил его и сразу получил бы вменяемый результат.

