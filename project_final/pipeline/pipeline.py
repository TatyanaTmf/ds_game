import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# Базовые пути к файлам
BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED_DIR = BASE_DIR / "data_processed"
OUTPUT_DIR = BASE_DIR / "output"

# Входной файл с данными (лежит в data_processed/)
INPUT_CSV = DATA_PROCESSED_DIR / "data_for_clustering.csv"

# Выходной файл с результатом (лежит в output/)
OUTPUT_CSV = OUTPUT_DIR / "regions_clusters_report.csv"

# На всякий случай создаём папку output, если её нет
OUTPUT_DIR.mkdir(exist_ok=True)


# Список признаков 
FEATURES = ["birth_rate_per_1000",
            "infant_mortality_per_1000",
            "cash_income_per_capita",
            "poverty_percent",
            "poor_children_share",
            "poor_old_share",
            "disabled_per_1000",
            "drug_alco_rate",
            "area_living_per_capita",
            "hh_high_crowding",
            "hh_plan_improve",
            "grp_per_capita",
            "industry_total",
            "retail_to_income",
            "crime_prev_convicted",
            "crime_intox_alcohol",
            "welfare_expense_share",
            "workers_share"
            ]
DROP_REGIONS = ["Архангельская область без АО",
                "Тюменская область без АО"
                ]

N_CLUSTERS = 3
RANDOM_STATE = 42
N_PCA_COMPONENTS = 0.9


def load_data(path: str) -> pd.DataFrame:
    """Загрузка данных из CSV."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимальная предобработка:
    - проверка наличия нужных колонок
    - работа с пропусками по ключевым признакам
    - логарифмирование признаков
    """
    missing = [col for col in FEATURES + ["region"] if col not in df.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют признаки: {missing}")

    df_work = df.copy()

    # удаляем ненужные регионы
    df_work = df_work[~df_work["region"].isin(DROP_REGIONS)].copy()

    # заполняем пропуски медианой по каждому признаку
    for col in FEATURES:
        median_value = df_work[col].median()
        df_work[col] = df_work[col].fillna(median_value)
    
    # логарифмирование признаков    
    skew_values = df_work[FEATURES].skew()
    log_features = skew_values[skew_values > 1].index.tolist()

    for col in log_features:
        df_work[col] = np.log1p(df_work[col])

    return df_work


def run_clustering(df_work: pd.DataFrame) -> pd.DataFrame:
    """
    Стандартизация признаков + k-means кластеризация.
    Возвращает df_work с добавленным столбцом 'cluster'.
    """
    X = df_work[FEATURES].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    kmeans = KMeans(n_clusters=N_CLUSTERS,
                    random_state=RANDOM_STATE,
                    n_init=10
                    )

    clusters = kmeans.fit_predict(X_pca)

    df_work = df_work.copy()
    df_work["cluster"] = clusters

    return df_work


def make_summary(df_work: pd.DataFrame) -> None:
    """
    Печать:
    - количество регионов по кластерам
    """
    print("\n=== Распределение регионов по кластерам ===")
    print(df_work["cluster"].value_counts().sort_index())


def print_top5_vulnerable(df_work: pd.DataFrame) -> None:
    """
    Вывод на экран 5 наиболее уязвимых регионов внутри кластера 0.
    Критерии приоритизации:
    - высокая бедность (poverty_percent, poor_children_share),
    - низкие доходы (cash_income_per_capita),
    - высокая инвалидность (disabled_per_1000).
    """
    print("\n=== Топ-5 наиболее уязвимых регионов внутри кластера 0 ===")

    if 0 not in df_work["cluster"].unique():
        print("Кластер 0 в результате кластеризации не найден.")
        return

    priority_cols = ["poverty_percent",
                     "poor_children_share",
                     "poor_old_share",
                     "cash_income_per_capita",
                     "disabled_per_1000"
                     ]

    cluster0 = df_work[df_work["cluster"] == 0].copy()

    top5 = (cluster0.sort_values(["poverty_percent", "poor_children_share", "cash_income_per_capita"],
                                 ascending=[False, False, True])[["region"] + priority_cols].head(5))
    
    out_path = OUTPUT_DIR / "top5_vulnerable_cluster0.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(top5.to_string(index=False))

    print(top5.to_string(index=False))
    
    
def print_top5_risk(df_work: pd.DataFrame) -> None:
    """
    Вывод на экран 5 регионов скрытого риска внутри кластера 2.
    Логика такая же, как для кластера 0, но считается, что это регионы,
    которые пока не уязвимы, но имеют тревожные показатели.
    """
    print("\n=== Топ-5 регионов скрытого риска внутри кластера 2 ===")

    if 2 not in df_work["cluster"].unique():
        print("Кластер 2 в результате кластеризации не найден.")
        return

    priority_cols = ["poverty_percent",
                     "poor_children_share",
                     "poor_old_share",
                     "cash_income_per_capita",
                     "disabled_per_1000"]
    cluster2 = df_work[df_work["cluster"] == 2].copy()

    top5_risk = (cluster2.sort_values(["poverty_percent", "poor_children_share", 
                                       "cash_income_per_capita", "disabled_per_1000"],
                                      ascending=[False, False, True, False])[["region"] + priority_cols].head(5))
    
    out_path = OUTPUT_DIR / "top5_risk_cluster2.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(top5_risk.to_string(index=False))
    
    print(top5_risk.to_string(index=False))


# Запуск всего пайплайна
def main():
    print("Загрузка данных из:", INPUT_CSV)
    df = load_data(INPUT_CSV)

    print("Предобработка данных...")
    df_work = preprocess(df)

    print("Запуск кластеризации k-means...")
    df_work = run_clustering(df_work)

    # Сохраняем итоговый отчёт по регионам
    print("Сохранение результатов кластеризации в:", OUTPUT_CSV)
    cols_to_save = ["region", "cluster"] + FEATURES
    df_work[cols_to_save].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    # Сводка по кластерам
    make_summary(df_work)

    # Топ-5 наиболее уязвимых регионов в кластере 0
    print_top5_vulnerable(df_work)
    
    # Топ-5 регионов риска в кластере 2
    print_top5_risk(df_work)

    print("\nГотово.")


if __name__ == "__main__":
    main()
