# Podstawy Sztucznej Inteligencji

## 1. Określenie tematu i celu projektu, analiza wymagań 
### Temat projektu
Estymacja wieku osoby na podstawie zdjęcia twarzy z wykorzystaniem sztucznej inteligencji.

### Cel projektu
Stworzenie modelu AI, który na podstawie zdjęcia twarzy osoby będzie w stanie oszacować jej wiek. Model ma za zadanie uczyć się wzorców wizualnych związanych z procesem starzenia (np. zmarszczki, kształt twarzy, struktura skóry), aby z jak największą dokładnością przypisać przewidywany wiek.

### Zakres projektu
- Pozyskanie, analiza i przygotowanie zbioru danych (IMDB-WIKI)
- Budowa i trenowanie modelu konwolucyjnej sieci neuronowej (CNN)
- Ocena jakości modelu (np. poprzez wskaźnik MAE)
- Testowanie modelu na nieznanych zdjęciach
- Możliwe rozszerzenie: przygotowanie prostego interfejsu użytkownika lub API

### Wymagania funkcjonalne i niefunkcjonalne
Funkcjonalne:
- Możliwość wprowadzenia zdjęcia i uzyskania przewidywanego wieku
- Obsługa różnych formatów obrazów (np. JPG, PNG)
- Pre-processing twarzy: detekcja i kadrowanie

Niefunkcjonalne:
- Wydajność: model powinien osiągnąć rozsądnie niską wartość MAE (Mean Absolute Error). Dla dobrych modeli do estymacji wieku przyjmuje się, że MAE na poziomie ok. 4-5 lat jest zadowalający.
- Skalowalność: możliwość użycia większego zbioru danych w przyszłości lub przeniesienia modelu do środowiska produkcyjnego.
- Bezpieczeństwo: model i dane będą przetwarzane lokalnie, bez przesyłania prywatnych zdjęć do chmury.
- Zasoby: model będzie trenowany z wykorzystaniem frameworków takich jak TensorFlow lub PyTorch, z możliwym wsparciem GPU (CUDA). Dane będą wstępnie analizowane w Pythonie.

