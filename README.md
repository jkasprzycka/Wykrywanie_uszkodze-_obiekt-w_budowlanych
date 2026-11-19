# Detekcja uszkodzeń obiektów budowlanych - porównanie architektur sieci neuronowych
## Zbiór danych i licencja
* **Nazwa zbioru:** SHM Computer Vision Model (`SHM-4`)
* **Źródło:** https://universe.roboflow.com/shm-agftj/shm-bafca/dataset/4
* **Liczba obrazów:** ~6,9k (z możliwością pracy na całym zbiorze lub wybranym podzbiorze testowym)
* **Licencja:** Creative Commons Attribution 4.0 International (**CC BY 4.0**)
* **Klasy uszkodzeń (6):**
  * `crack` (pęknięcia)
  * `efflorescence` (wykwity)
  * `Exposed_reinforcement` (odsłonięte zbrojenie)
  * `Ruststrain` (zacieki rdzy)
  * `Scaling` (łuszczenie)
  * `Spalling` (odpryski)
 
---

## Metodologia i warunki porównania (Konfiguracja 1)
Aby zapewnić rzetelne porównanie wszystkich trzech architektur, zastosowano wspólne parametry:
* **Ten sam zbiór danych oraz podział** na zbiory treningowy / walidacyjny / testowy
* **Wspólny seed:** `42`
* **Rozdzielczość wejściowa:** `640x640` px
* **Liczba epok:** `10
* **Hiperparametry (np. `batch_size`, `lr`):** dobrane z uwzględnieniem specyfiki architektury oraz dostępnej pamięci

---
