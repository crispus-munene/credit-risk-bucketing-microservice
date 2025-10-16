import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 
logging.basicConfig(format=log_format, level=logging.INFO)

def build_features(df: pd.DataFrame):
    df= df.copy()

    df= df.loc[df['loan_status'].isin(['Fully Paid', 'Charged Off'])]
    logging.info(f'Fully paid or charged off data: {df["loan_status"].value_counts(normalize=True)}')

    df['charged_off']= (df['loan_status'] == 'Charged Off').apply(np.uint8)

    df.drop('loan_status', axis=1, inplace=True)

    #Removing values with more than 30% missing data
    missing_fractions= df.isnull().mean().sort_values(ascending=False)
    drop_list= sorted(list(missing_fractions[missing_fractions > 0.3].index))
    df.drop(labels= drop_list, axis=1, inplace=True)

    #Keeping domain knowledge data and data dictionary information
    keep_list= ['charged_off','funded_amnt','addr_state', 'annual_inc', \
                'application_type','dti', 'earliest_cr_line', 'emp_length',\
                'emp_title', 'fico_range_high',\
                'fico_range_low', 'grade', 'home_ownership', 'id', 'initial_list_status', \
                'installment', 'int_rate', 'loan_amnt',\
                'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', \
                'purpose', 'revol_bal', 'revol_util', \
                'sub_grade', 'term', 'title', 'total_acc',\
                'verification_status', 'zip_code','last_pymnt_amnt',\
                'num_actv_rev_tl', 'mo_sin_rcnt_rev_tl_op',\
                'mo_sin_old_rev_tl_op',"bc_util","bc_open_to_buy",\
                "avg_cur_bal","acc_open_past_24mths" ]
    
    drop_list_dict= [col for col in df.columns if col not in keep_list]
    df.drop(labels=drop_list_dict, axis=1, inplace=True)

    #correlation with the target variable
    correlation= df.select_dtypes(exclude='object').corr()
    corr_co= abs(correlation['charged_off'])
    drop_list_corr= sorted(list(corr_co[corr_co < 0.03].index))
    df.drop(labels=drop_list_corr, axis=1, inplace=True)

    logging.info(f'Current dataset shape: {df.shape}')

    #High cardinality & irrelevant features
    df.drop(['id', 'emp_title', 'zip_code', 'title', 'emp_length'], axis=1, inplace=True)

    df['term']= df['term'].apply(lambda s: np.int8(s.split()[0]))

    #log transform outliers
    df['log_annual_inc']= df['annual_inc'].apply(lambda x: np.log10(x+1))
    df.drop('annual_inc', axis=1, inplace=True)

    # Feature engineering fico score
    df['fico_score'] = df[['fico_range_low', 'fico_range_high']].mean(axis=1)
    df.drop(['fico_range_low', 'fico_range_high'], axis=1, inplace=True)

    df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y')

    #Feature engineering earliest_cr_line
    current_date = pd.to_datetime('today')
    df['credit_age_months'] = (current_date.year - df['earliest_cr_line'].dt.year) * 12 + \
                            (current_date.month - df['earliest_cr_line'].dt.month)

    df['credit_age_years'] = df['credit_age_months'] / 12

    df= df.drop(columns= ['earliest_cr_line'])

    logging.info(f'Final dataset shape: {df.shape}')

    X= df.loc[:, df.columns != 'charged_off']
    y= df['charged_off']

    train, test= train_test_split(df, test_size=0.3, random_state=42, stratify=y)

    train.to_csv('data/interim/train.csv', index=False)
    test.to_csv('data/interim/test.csv', index=False)




    