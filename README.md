# hackathon-linq

Рабочий репо команды Pokus для хакатона «Открой#Моспром» (кейс LinQ).
Здесь только продакшен-код для VPS; все артефакты (модели, .tpu, видео, логи) живут вне гита.

## Структура
```
hackathon-linq/
├── tpu_pycloud_format/   # ИХ код как есть (4 файла), скопирован из Jupyter, не правим
├── linq/                 # прод: конфиг, реестр программ, пре/постпроцессинг, бэкенды
│   └── backend/          #   base (контракт), cloud (RabbitMQ -> TPU)
├── service/              # прод: пайплайн и смоук-проверка тракта
├── tests/                # тесты на моках, гоняются БЕЗ VPN (pytest -q)
├── tools/                # не прод: бенч, экспорт ONNX, компиляция в .tpu
└── (requirements, .env.example, README, conftest)
```

## Одноразовый монтаж (руками, дальше не трогаем)
1. VPS: OpenVPN под systemd с автореконом, импорт командного `Hackathon.ovpn`, подключено.
2. Jupyter (`http://hackathon.lin-q.ru/`, с любой машины с VPN):
   - убедиться, что `torch_resnet50_b1_o5_h1m.tpu` лежит в `/auto/cache/<логин>/`
     (ячейка 1 их примерного ноутбука делает это, либо вручную `!cp ... /auto/cache/<логин>/`);
   - скачать папку `tpu_pycloud_format` (4 файла как есть) в корень этого репо;
   - скачать `ILSVRC2012_val_00000045.JPEG` в любую папку на VPS (в гит не кладём).
3. VPS: `git clone`, `python -m venv .venv`, `pip install -r requirements.txt`,
   `cp .env.example .env`, заполнить.

## Проверки
- Без VPN (пайплайн, пре/пост, мок): `pytest -q`
- С VPN (весь тракт до TPU): `python -m service.smoke --image /путь/ILSVRC2012_val_00000045.JPEG`
  Ожидание: класс 257, latency ~1 с.
- Замеры скорости (обязательное требование кейса):
  `python -m tools.bench --image /путь/картинка.jpg --runs 20` → JSON в `logs/`.

## SDK и компиляция своих моделей (понадобится на шаге YOLO)
- SDK (`tpu_framework`) и доки — по ссылкам из PDF «Процедура подготовки и инференса сетей на LinQ TPU».
- Компиляция: `python -m tools.compile_tpu --onnx ... --name ... --input-nodes '{"images":[1,3,640,640]}'`.
- Готовый `.tpu` заливаем в `/auto/cache/<логин>/` вручную и регистрируем в `linq/registry.py`.

## Правила гита
- Не коммитим: `.env`, `.tpu`, `.onnx`, веса, видео, калибровку, логи, `artifacts/`.
- `tpu_pycloud_format/` — чужой код как есть, не редактируем.
- Python >= 3.9.