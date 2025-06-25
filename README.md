# Analiza skupień metodami FCM i PCM na zbiorze danych Wine Quality

Projekt zawiera zestaw skryptów do analizy skupień (clustering) na zbiorze danych **Wine Quality**. Wykorzystano metody rozmyte (fuzzy) FCM i PCM, a także analizę głównych składowych (PCA) i wskaźnik jakości grupowania Xie-Beni.

## Zawartość projektu

- `fcm.py` – implementacja algorytmu **Fuzzy C-Means (FCM)** do grupowania danych
- `pcm.py` – implementacja algorytmu **Possibilistic C-Means (PCM)**
- `pca.py` – skrypt wykonujący **redukcję wymiarowości PCA** dla danych wejściowych
- `xie_beni.py` – obliczanie **wskaźnika Xie-Beni**, który ocenia jakość uzyskanego podziału na grupy

## Dane

Analiza prowadzona jest na publicznie dostępnym zbiorze danych:  
**Wine Quality Dataset** – dane dotyczące jakości wina, zawierające cechy chemiczne i sensoryczne.

## Wymagania

- Python 3.x
- NumPy
- Pandas
- scikit-learn
- scikit-fuzzy
- sklearn
- seaborn
- Matplotlib (opcjonalnie, do wizualizacji)

## Uruchomienie

Każdy skrypt można uruchomić niezależnie. Dane wejściowe powinny znajdować się w katalogu roboczym w pliku `.csv`.

---

> W przyszłości planowany jest jeden skrypt integrujący cały proces analizy.


