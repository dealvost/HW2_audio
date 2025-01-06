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
6. Запустить инференс модели на датасете train-other 500  с LM командой ниже.

   ```bash
   python HW2_audio/inference.py -cn inference_other \
   inferencer.from_pretrained="template_asr/saved/DeepSpeech2_trainother/model_best_wer.pth" \
   inferencer.save_path="inference_outputs_lm" \
   model=deepspeech2
   ```
7. Запустить инференс модели на датасете train-other 500  с LM с BPE токенизацией командой ниже.
   ```bash
   python HW2_audio/inference_with_lm_beam.py -cn inference_BPE
   ```


   Всего к данном отчету прикреплено 4 модели (на самом деле обучалось больше, но прикреплять все не имеет смысла)

   Отчет по модели DeepSpeech2_clean360_2: https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/f31rsvua?nw=nwusermarkoavro01

   Отчет по модели DeepSpeech2_trainother: https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/pbmwty22?nw=nwusermarkoavro01

   Отчет по модели DeepSpeech2_BPE_main:  https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/yue2u1sz?nw=nwusermarkoavro01

   Отчет по модели DeepSpeech2_BPE_finetune_1: https://wandb.ai/markoavro01-hse-university/pytorch_template_asr_example/runs/qnkkkno8?nw=nwusermarkoavro01 (неудачная попытка исполльзовать аугментации для повышения метрик относительно модели BPE_main)

