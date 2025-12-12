import pandas as pd
import numpy as np

def get_useful_columns():
    return {
        "application": [
            "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
            "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
            "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
            "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE",
            "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
            "OWN_CAR_AGE", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
            "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
            "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
        ],
        "bureau": [
            "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_CURRENCY",
            "DAYS_CREDIT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "CNT_CREDIT_PROLONG"
        ],
        "previous_application": [
            "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS",
            "AMT_ANNUITY", "AMT_CREDIT", "AMT_DOWN_PAYMENT",
            "DAYS_DECISION", "NAME_PAYMENT_TYPE", "CODE_REJECT_REASON",
            "NAME_CLIENT_TYPE", "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO",
            "NAME_PRODUCT_TYPE", "CHANNEL_TYPE", "NAME_YIELD_GROUP"
        ],
        "installments_payments": [
            "SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_VERSION",
            "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
            "AMT_INSTALMENT", "AMT_PAYMENT"
        ],
        "credit_card_balance": [
            "SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE",
            "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL",
            "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT",
            "AMT_PAYMENT_TOTAL_CURRENT", "SK_DPD", "SK_DPD_DEF"
        ],
        "POS_CASH_balance": [
            "SK_ID_PREV", "SK_ID_CURR", "MONTHS_BALANCE",
            "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
            "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"
        ]
    }


def create_domain_features(df):
    # 1. Pourcentage du crédit par rapport au revenu
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # 2. Taux d'effort (Annuité / Revenu)
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']

    # 3. Durée du crédit (Annuité / Montant total)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

    # 4. Pourcentage des jours travaillés par rapport à l'âge
    df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    return df


def load_and_aggregate_table(file_path, table_name, group_col='SK_ID_CURR'):
    # Récupération de la liste des colonnes utiles
    useful_cols = get_useful_columns().get(table_name)

    # Chargement optimisé (on ne lit que ce qui est nécessaire)
    if useful_cols:
        # Vérification rapide des colonnes présentes dans le fichier
        header = pd.read_csv(file_path, nrows=0).columns.tolist()
        final_cols = [c for c in useful_cols if c in header]
        df = pd.read_csv(file_path, usecols=final_cols)
    else:
        df = pd.read_csv(file_path)

    # Suppression des IDs secondaires qui faussent les moyennes
    cols_to_drop = [c for c in ['SK_ID_PREV', 'SK_ID_BUREAU'] if c in df.columns]
    df_clean = df.drop(columns=cols_to_drop)

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

    if group_col not in numeric_cols:
        numeric_cols.append(group_col)

    df_numeric = df_clean[numeric_cols]

    # Agrégation Numérique : Min, Max, Moyenne, Somme
    num_agg = df_numeric.groupby(group_col).agg(['min', 'max', 'mean', 'sum'])

    # Renommage des colonnes pour qu'elles soient uniques
    num_agg.columns = [f'{table_name.upper()}_{c[0]}_{c[1].upper()}' for c in num_agg.columns]

    return reduce_mem_usage(num_agg)

def missing_values_table(df):
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def reduce_mem_usage(df):
    """
    Itère sur toutes les colonnes d'un DataFrame et réduit la précision
    des types numériques (int et float) pour diminuer la consommation de mémoire.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Usage mémoire initial du DataFrame: {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        # Traiter uniquement les colonnes numériques
        if col_type != object and col_type != str and col_type != bool:
            c_min = df[col].min()
            c_max = df[col].max()

            # --- Conversion des entiers (Integers) ---
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)

            # --- Conversion des décimaux (Floats) ---
            else:
                # La majorité de vos colonnes d'agrégats (mean, var, proportions) sont ici
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    # Conversion principale : float64 -> float32
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64) # Garder float64 si la précision est nécessaire

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Usage mémoire final du DataFrame: {end_mem:.2f} MB")
    print(f"Mémoire réduite de {(start_mem - end_mem) / start_mem * 100:.1f} %")

    return df
