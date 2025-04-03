from prepare_dataset import prepare_dataset


def main():
    # path_to_dataset = input("Podaj sciezke do katalogu zawierajacego zbior: ")
    path_to_dataset = 'F:\\imdb_crop\\'
    df_clean = prepare_dataset(path_to_dataset + 'imdb.mat')

if __name__ == '__main__':
    main()
