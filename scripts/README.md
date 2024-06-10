## Training 

Скрипт запускается из корня директории

```bash
python -m scripts.train
```

Также скрипту можно передать параметры обучения (см. код scripts/train.py)

```bash
python -m scripts.train --scaling_factor=4 --l_adv=0.3
```

Остальные параметры можно посмотреть в `scripts/opts.yml`

## Evaluate

Тест-скрипту передается путь к логгам `pytorch-lightning'a`
```bash
python -m scripts.test --model_path ./runs/down=True@sc=2@cod1=15@res=True@gan=False@vgg=False
```