
# Detekcja uszkodzeń konstrukcji budowlanych: 3 architektury × 2 konfiguracje

Projekt z przedmiotu *Przetwarzanie i analiza cyfrowych danych przestrzennych i budowlanych metodami sztucznej inteligencji* (2026/2027), dr inż. Karolina Tomaszkiewicz.

## Problem i dane

- **Rodzaj uszkodzeń:** pęknięcia konstrukcji betonowych.
- **Zadanie:** detekcja obiektów (notebook wykrywa typ zadania na podstawie plików etykiet i obsługuje również segmentację instancji).
- **Zbiór danych:** SHM Dataset, wersja 4, Roboflow Universe: https://universe.roboflow.com/shm-agftj/shm-bafca/dataset/4
  (12 001 obrazów: 7 893 treningowe / 2 049 walidacyjne / 2 059 testowych, zgodnie z opublikowanymi danymi; zbiór treningowy zawiera kopie obrazów z odbiciami i obrotami wykonanymi przez Roboflow).
- **Licencja:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)

## Plan eksperymentu

Trzy architektury, każda trenowana w dwóch konfiguracjach (łącznie 6 treningów).

| Architektura | Typ |
|---|---|
| YOLOv8n | jednofazowy detektor CNN |
| YOLO11n | jednofazowy detektor CNN (nowsza generacja) |
| RT-DETR-l | detektor oparty na transformerze |

- **Konfiguracja 1 (wspólna baza):** wagi wstępnie wytrenowane na zbiorze COCO.
- **Konfiguracja 2 (jedna zmiana):** losowa inicjalizacja (trening od podstaw), taka sama liczba epok.

We wszystkich przebiegach zachowane są te same ustawienia: seed 42, podzbiór i podział danych, rozmiar wejściowy, liczba epok, augmentacja i metryki.
Hiperparametry dla poszczególnych architektur: optymalizator AdamW dla wszystkich modeli, learning rate 1e-3 dla YOLO i 1e-4 dla RT-DETR.
Domyślnie używany podzbiór: 2000 obrazów treningowych / 400 walidacyjnych / 500 testowych, rozmiar 512 px, 15 epok, batch 8. Metryki są obliczane na podzbiorze testowym z użyciem punktu kontrolnego, który uzyskał najlepszy wynik na walidacji: precision, recall, F1, mAP@0.5, mAP@0.5:0.95, a także liczba parametrów, GFLOPs i czas wnioskowania.

### Hipotezy (zapisane przed treningiem)

1. Wstępne treningowanie zwiększy testowe mAP@0.5 dla wszystkich trzech architektur przy tak krótkim treningu.
2. Największy wzrost wystąpi dla RT-DETR-l (transformera wymagającego wielu danych i wolno osiągającego zbieżność), a mniejszy — dla małych modeli YOLO opartych na CNN.
3. Przy wagach wstępnie wytrenowanych YOLOv8n i YOLO11n osiągną zbliżone wyniki; RT-DETR-l może być porównywalny lub nieco lepszy pod względem mAP@0.5:0.95, ale będzie znacznie wolniejszy.

## Odtwarzanie eksperymentu (VS Code + karta NVIDIA)

### Konfiguracja GPU (Windows, testowana konfiguracja: RTX 2060)

1. Zainstaluj aktualny sterownik NVIDIA (GeForce Experience lub nvidia.com). Sprawdź jego działanie: polecenie `nvidia-smi` uruchomione w terminalu powinno wyświetlić kartę GPU. Osobna instalacja CUDA Toolkit **nie jest** potrzebna.
2. Utwórz środowisko wirtualne i najpierw zainstaluj wersję PyTorch z obsługą CUDA, a następnie pozostałe pakiety:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install ultralytics jupyter ipykernel pandas matplotlib pyyaml
   ```
   (`cu126` to jedna z wersji CUDA wymienionych na stronie https://pytorch.org/get-started/locally/; jeśli podane tam polecenie będzie inne, użyj aktualnego polecenia ze strony.)
   Jeśli `ultralytics` został zainstalowany jako pierwszy, mógł pobrać PyTorch działający wyłącznie na CPU: uruchom `pip uninstall -y torch torchvision`, a następnie ponownie wykonaj powyższe polecenie instalujące PyTorch.
3. Sprawdź działanie w Pythonie: `import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))` powinno wyświetlić `True` i nazwę karty GPU.
4. Otwórz notebook w VS Code i wybierz interpreter `.venv` jako kernel (w prawym górnym rogu).

### Uruchomienie

1. W komórce ustawień podaj w `COCO_ROOT` ścieżkę do folderu zawierającego podfoldery `train`, `valid` i `test` (każdy z plikiem `_annotations.coco.json`). Notebook automatycznie konwertuje COCO do formatu YOLO.
2. Uruchom wszystkie komórki. Komórka mierząca czas oszacuje całkowity czas przed rozpoczęciem pełnych treningów. Zakończone przebiegi są zapisywane w `results/runs` i pomijane przy ponownym uruchomieniu notebooka.
3. Przy 6 GB pamięci VRAM domyślny rozmiar batch wynosi 8. Jeśli zabraknie pamięci podczas treningu RT-DETR-l, ustaw `BATCH = 4` dla **wszystkich** modeli i uruchom eksperyment ponownie (najpierw usuń `results/runs`, aby wszystkie przebiegi używały tego samego rozmiaru batch).

## Wyniki

*Uzupełnij po zakończeniu treningu: wstaw `results_summary.csv`, `comparison_map.png`, `curves_and_speed.png` oraz wykresy predykcji.*

## Ograniczenia

Użyto podzbioru danych i jednego seeda dla każdego przebiegu (niewielkie różnice mogą wynikać z szumu). IoU jest surową miarą dla cienkich pęknięć. Dla każdego modelu przetestowano tylko jedno ustawienie hiperparametrów.
