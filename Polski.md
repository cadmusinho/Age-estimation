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

### Zbiór danych używany w projekcie
Projekt wykorzystuje publicznie dostępny zbiór danych IMDB-WIKI, zawierający zdjęcia twarzy celebrytów wraz z metadanymi, takimi jak data urodzenia, data zdjęcia, płeć i imię osoby. Dane pochodzą z IMDb: zdjęcia aktorów i aktorek z powiązanymi metadanymi.

### Etapy przygotowania danych:
**Pobieranie danych**:
- Już przycięte zdjęcia twarzy
- Plik .mat z metadanymi dla każdego zdjęcia

**Wiek dla każdego zdjęcia oblicza się za pomocą wzoru**:
  - wiek = data_zdjęcia - rok urodzenia

**Weryfikacja i czyszczenie danych**:
- Usuwanie uszkodzonych obrazów
- Filtrowanie obrazów z wieloma twarzami
- Odrzucanie przypadków z niewystarczającymi danymi
- Usuwanie odstających wartości – (obrazy z wiekiem poniżej 0 lub powyżej 100, itd.)

**Podział na zbiory treningowe i testowe**

### Oczekiwany wynik:
Czysty i gotowy do użycia zbiór danych zawierający:
- zdjęcia twarzy o odpowiedniej jakości
- odpowiadające etykiety wiekowe
- jednolity format danych wejściowych

## 2. Przygotowanie danych
### Obliczanie wieku:
Wiek dla każdego zdjęcia obliczany jest jako różnica między rokiem wykonania zdjęcia a rokiem urodzenia osoby na zdjęciu.

### Czyszczenie danych:
- Usuwanie zdjęć z niepoprawnymi lub brakującymi danymi (np. niewłaściwe daty, brakujące imiona lub ścieżki do zdjęcia).
- Usuwanie zdjęć z wykrytą wieloma twarzami (na podstawie `second_face_score`).
- Usuwanie rekordów z wartościami wieku poza określonym zakresem (poniżej 10, powyżej 95).

### Wstępna analiza danych:
- Określenie liczby zdjęć, które spełniają kryteria wieku i jakości.

Zbiór danych został przeanalizowany pod kątem rozkładu liczby zdjęć w klasach wiekowych. Wystąpiły duże różnice – niektóre grupy wiekowe miały mniej niż 1000 zdjęć.
Jak pokazano, liczba próbek znacznie się różniła w zależności od wieku — niektóre klasy zawierały mniej niż 1000 zdjęć.

![2](https://github.com/user-attachments/assets/ec374735-61b3-429d-99e3-6b3d96892b64)

W celu zbalansowania klas wiekowych zastosowano różne techniki augmentacji obrazów, takie jak:

- obrót
- przesunięcie
- zoomowanie
- odbicie lustrzane

Celem było zwiększenie liczby przykładów w każdej klasie wiekowej do co najmniej **5500**, uzyskując bardziej jednolity rozkład.

Drugi histogram przedstawia zbiór danych po zbalansowaniu, z porównywalną liczbą obrazów w każdej klasie.

![1](https://github.com/user-attachments/assets/bf77ae92-9c70-43ee-8b6d-ce19bd61b714)

80% zdjęć, które spełniają kryteria, zostanie użytych w zbiorze treningowym, a pozostałe 20% w zbiorze testowym.
