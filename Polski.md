# Podstawy sztucznej inteligencji
## 1. Temat projektu i cel
### Temat projektu
Estymacja wieku na podstawie obrazu twarzy osoby przy użyciu sztucznej inteligencji.

### Cel projektu
Stworzenie modelu AI zdolnego do estymacji wieku osoby na podstawie obrazu jej twarzy. Model ma na celu nauczenie się wizualnych wzorców związanych ze starzeniem się (np. zmarszczki, kształt twarzy, tekstura skóry), aby przewidywać wiek jak najdokładniej.

### Zakres projektu
- Pozyskiwanie i przygotowanie danych
- Trenowanie modelu
- Ocena wydajności modelu
- Testowanie modelu na nieznanych obrazach
- Możliwe rozszerzenie: przygotowanie prostego interfejsu użytkownika lub API

### Wymagania
- Możliwość wprowadzenia obrazu i uzyskania przewidywanego wieku
- Obsługa różnych formatów obrazów (np. JPG, PNG)
- Na razie brak wstępnego przetwarzania: użytkownik musi dostarczyć dobrze przycięte zdjęcie twarzy z jak najmniejszym tłem

Model będzie trenowany z użyciem frameworków i bibliotek Pythona (PyTorch, Torchvision, matplotlib, pandas, Pillow).

## 2. Zbiór danych i przygotowanie danych
### Zbiór danych używany w projekcie
Projekt wykorzystuje publicznie dostępny zbiór danych IMDB-WIKI, zawierający zdjęcia twarzy celebrytów wraz z metadanymi, takimi jak data urodzenia, data zdjęcia, płeć i imię osoby. Dane pochodzą z IMDb: zdjęcia aktorów i aktorek z powiązanymi metadanymi.

### Etapy przygotowania danych:
Pobieranie danych:
- Już przycięte zdjęcia twarzy
- Plik .mat z metadanymi dla każdego zdjęcia

Wiek dla każdego zdjęcia oblicza się za pomocą wzoru:
  - wiek = data_zdjęcia - rok urodzenia

Weryfikacja i czyszczenie danych:
- Usuwanie uszkodzonych obrazów
- Filtrowanie obrazów z wieloma twarzami
- Odrzucanie przypadków z niewystarczającymi danymi
- Usuwanie odstających wartości – (obrazy z wiekiem poniżej 0 lub powyżej 100, itd.)

Podział na zbiory treningowe i testowe

### Oczekiwany wynik:
Czysty i gotowy do użycia zbiór danych zawierający:
- zdjęcia twarzy o odpowiedniej jakości
- odpowiadające etykiety wiekowe
- jednolity format danych wejściowych
