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

## 2. Zbiór danych i ich przygotowanie
### Zbiór danych użyty w projekcie
Do realizacji projektu wykorzystano publicznie dostępny zbiór danych IMDB-WIKI, który zawiera łącznie ponad 500 000 zdjęć twarzy celebrytów wraz z metadanymi – datą urodzenia, datą wykonania zdjęcia, płcią oraz nazwą osoby.
Zbiór został przygotowany przez badaczy w ramach prac nad modelem DEX (Deep EXpectation), który zdobył 1. miejsce w konkursie LAP 2015 na estymację wieku. Dane pochodzą z dwóch źródeł:
- IMDb: zdjęcia aktorów i aktorek z przypisanymi metadanymi,
- Wikipedia: profile osób publicznych z analogicznymi informacjami.

### Etapy przygotowania danych:
Pobranie danych:
- Dane zostały pobrane w formie archiwum .tar, które zawiera:
  - Obrazy twarzy (surowe lub już przycięte – cropped)
  - Plik .mat zawierający metadane dla każdego zdjęcia (np. dob, photo_taken, gender, face_location, itp.)

Wczytanie i analiza metadanych:
- Metadane są przetwarzane w Pythonie (np. za pomocą scipy.io.loadmat()).
- Dla każdego zdjęcia wyliczana jest wartość wieku według wzoru:
wiek = photo_taken - rok urodzenia
(z uwzględnieniem, że zdjęcia uznaje się za wykonane w połowie roku).

Weryfikacja i czyszczenie danych:
- Usuwanie uszkodzonych obrazów (np. plików, które nie otwierają się lub są puste).
- Filtrowanie zdjęć z wieloma twarzami – jeśli second_face_score jest większy niż określony próg, zdjęcie jest odrzucane.
- Odrzucanie przypadków bez wystarczających danych – np. brak dob, photo_taken, nieznana płeć (jeśli potrzebna).
- Usunięcie danych odstających (outliers) – np. zdjęcia z wiekiem poniżej 0 lub powyżej 100 lat, lub o bardzo niskim score detekcji twarzy.

Kadrowanie i wyodrębnienie twarzy:
- Twarze są przycinane na podstawie współrzędnych face_location z dodatkowym marginesem 40% (aby model mógł lepiej uchwycić cechy otaczające twarz).
- Można wykorzystać gotowe przycięte obrazy (wiki_crop, imdb_crop) lub wykonać własne kadrowanie.

Transformacja i normalizacja danych:
- Obrazy są przeskalowywane do standardowego rozmiaru (np. 224x224 piksele) odpowiedniego dla sieci VGG-16.
- Dane są normalizowane (np. przez odjęcie średniej ImageNet mean i podzielenie przez std).
- Obrazy mogą być konwertowane do formatu tensorów (torch.Tensor, np.array), a metadane przekształcane do etykiet liczbowych.

Podział na zbiory treningowe i testowe:
- Dane są dzielone na zbiór treningowy, walidacyjny i testowy (np. 70%-15%-15%).
- Podział powinien zapewniać równomierne rozłożenie wieku w każdej grupie (stratyfikacja), aby uniknąć biasu wiekowego.

### Oczekiwany wynik:
Gotowy i czysty zbiór danych, zawierający:
- obrazy twarzy o odpowiedniej jakości,
- przypisane etykiety wieku (liczbowe),

jednorodny format danych wejściowych.

Dane są w pełni gotowe do użycia w procesie trenowania modelu AI, a także odpowiednio przygotowane do walidacji i testowania.

Zbiór może być dodatkowo wzbogacony o augmentację (obracanie, skalowanie, zmiana jasności) podczas treningu, aby zwiększyć różnorodność próbek.
