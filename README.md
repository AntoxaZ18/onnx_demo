# Программа для запуска инференса моделей ONNX на видео.

## Описание
Распознавание объектов на видео при помощи экспортированных 
моделей YOLO11 в формате ONNX 

## Установка

### Требования
- Python 3.11 или выше
- `poetry` менеджер пакетов

### Шаги установки на Windows
1. Клонируйте репозиторий:
   ```Powershell
   git clone https://github.com/AntoxaZ18/onnx_demo.git
   cd onnx_demo
   ```
2. Установка при помощи пакетного менеджера
   Если хотите чтобы вирутальная среда создалась в папке с проектом
   ```Powershell
   poetry config settings.virtualenvs.in-project true
   ```
   Создайте преднастроенную виртуальную среду
   ```Powershell
   poetry install
   ```
   Активируйте среду при помощи poetry (опционально)
   ```Powershell
   poetry env activate
   ```
   Для сборки в exe файл (будет находиться в папке dist):
      ```Powershell
   poetry run build-script 
   ```
