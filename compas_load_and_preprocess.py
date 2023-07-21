import pandas as pd
from datetime import datetime, date, timedelta

# Loading dataset
def load_compas():
    df = pd.read_csv('data/compas-scores-two-years.csv', index_col='id')
    # Not relevant
    del df['name']
    del df['first']
    del df['last']
    df['age'] = df['age_cat']
    del df['age_cat']  # Alreafy in age
    del df['dob']  # Already in age
    del df['vr_case_number']
    del df['r_case_number']
    del df['c_case_number']
    del df['days_b_screening_arrest']

    # Potentially useless
    del df['c_offense_date']
    del df['c_jail_in']
    del df['c_jail_out']
    del df['event']
    del df['start']
    del df['end']

    # Very partial and potentially useless
    del df['r_days_from_arrest']
    del df['r_jail_in']
    del df['r_jail_out']
    del df['r_offense_date']

    # There is another better cleaned column (and/or less empty)
    del df['r_charge_degree']
    del df['vr_charge_degree']
    del df['r_charge_desc']

    # Almost empty
    del df['vr_offense_date']
    del df['vr_charge_desc']
    del df['c_arrest_date']

    # Empty
    del df['violent_recid']

    # Duplicates
    del df['priors_count.1']

    # Only one unique value
    del df['v_type_of_assessment']
    del df['type_of_assessment']

    # Prediction of COMPAS
    del df['v_decile_score']
    del df['score_text']
    del df['screening_date']
    del df['decile_score.1']
    del df['v_screening_date']
    del df['v_score_text']
    del df['compas_screening_date']
    del df['c_days_from_compas']
    del df['decile_score']

    # Custody
    df = df.dropna()
    df['custody'] = (df['out_custody'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) - df['in_custody'].apply(
        lambda x: datetime.strptime(x, '%Y-%m-%d'))).apply(lambda x: x.total_seconds() / 3600 / 24).astype(int)
    del df['out_custody']
    del df['in_custody']

    def summarise_charge(x):
        drugs = ['clonaz', 'heroin', 'cocaine', 'cannabi', 'drug', 'pyrrolidin', 'Methyl', 'MDMA', 'Ethylone',
                 'Alprazolam', 'Oxycodone',
                 'Methadone', 'Methamph', 'Bupren', 'Lorazepam', 'controlled', 'Amphtamine', 'contro', 'cont sub',
                 'rapher', 'fluoro',
                 'ydromor', 'methox', 'iazepa', 'XLR11', 'steroid', 'morphin', 'contr sub', 'enzylpiper', 'butanediol',
                 'phentermine',
                 'Fentanyl', 'Butylone', 'Hydrocodone', 'LSD', 'Amobarbital', 'Amphetamine', 'Codeine', 'Carisoprodol']
        drugs_selling = ['sel', 'del', 'traf', 'manuf']
        if sum([d.lower() in x.lower() for d in drugs]) > 0:
            if sum([h in x.lower() for h in drugs_selling]) > 0:
                x = 'Drug Traffic'
            else:
                x = 'Drug Possess'
        elif 'murd' in x.lower() or 'manslaughter' in x.lower():
            x = 'Murder'
        elif 'sex' in x.lower() or 'porn' in x.lower() or 'voy' in x.lower() or 'molest' in x.lower() or 'exhib' in x.lower():
            x = 'Sex Crime'
        elif 'assault' in x.lower() or 'carjacking' in x.lower():
            x = 'Assault'
        elif 'child' in x.lower() or 'domestic' in x.lower() or 'negle' in x.lower() or 'abuse' in x.lower():
            x = 'Family Crime'
        elif 'batt' in x.lower():
            x = 'Battery'
        elif 'burg' in x.lower() or 'theft' in x.lower() or 'robb' in x.lower() or 'stol' in x.lower():
            x = 'Theft'
        elif 'fraud' in x.lower() or 'forg' in x.lower() or 'laund' in x.lower() or 'countrfeit' in x.lower() or 'counter' in x.lower() or 'credit' in x.lower():
            x = 'Fraud'
        elif 'prost' in x.lower():
            x = 'Prostitution'
        elif 'trespa' in x.lower() or 'tresspa' in x.lower():
            x = 'Trespass'
        elif 'tamper' in x.lower() or 'fabricat' in x.lower():
            x = 'Tampering'
        elif 'firearm' in x.lower() or 'wep' in x.lower() or 'wea' in x.lower() or 'missil' in x.lower() or 'shoot' in x.lower():
            x = 'Firearm'
        elif 'alking' in x.lower():
            x = 'Stalking'
        elif 'dama' in x.lower():
            x = 'Damage'
        elif 'driv' in x.lower() or 'road' in x.lower() or 'speed' in x.lower() or 'dui' in x.lower() or 'd.u.i.' in x.lower():
            x = 'Driving'

        else:
            x = 'Other'

        return x

    df['charge_desc'] = df['c_charge_desc'].apply(summarise_charge)
    del df['c_charge_desc']

    CUSTODY_RANGES = {
        (0, 1): '0 days',
        #         (1,2): '1 day',
        #         (2,5): '2-4 days',
        #         (5,10): '5-9 days',
        (1, 10): '1-9 days',

        #         (10,30): '10-29 days',
        #         (30,90): '1-3 months',
        #         (90,365): '3-12 months',
        (10, 30): '10-29 days',
        (30, 365): '1-12 months',

        #         (365,365*2): '1 year',
        #         (365*2,365*3): '2 years',
        (365 * 1, 365 * 3): '1-2 years',
        (365 * 3, 365 * 5): '3-4 years',
        #         (365*5,365*10): '5-9 years',
        (365 * 5, df['custody'].max() + 1): '5 years or more'
        #         (365*10, df['custody'].max()+1): '10 years or more'
    }

    PRIORS_RANGES = {
        (0, 1): '0 priors',
        (1, 2): '1 priors',
        #         (2,3): '2 priors',
        #         (3,5): '3-4 priors',
        (2, 5): '2-4 priors',
        (5, 10): '5-9 priors',
        (10, df['priors_count'].max() + 1): '10 priors or more',
    }
    JUV_OTHER_RANGES = {
        (0, 1): '0 juv others',
        (1, 2): '1 juv others',
        #         (2,3): '2 juv others',
        #         (3,5): '3-4 juv others',
        (2, 5): '2-4 juv others',

        (5, df['juv_other_count'].max() + 1): '5 or more juv others',
    }
    JUV_FEL_RANGES = {
        (0, 1): '0 juv fel',
        (1, 2): '1 juv fel',
        #         (2,3): '2 juv fel',
        #         (3,5): '3-4 juv fel',
        (2, 5): '2-4 juv fel',

        (5, df['juv_fel_count'].max() + 1): '5 or more juv fel',
    }
    JUV_MISD_RANGES = {
        (0, 1): '0 juv misd',
        (1, 2): '1 juv misd',
        #         (2,3): '2 juv misd',
        #         (3,5): '3-4 juv misd',
        (2, 5): '2-4 juv misd',

        (5, df['juv_misd_count'].max() + 1): '5 or more juv misd',
    }

    def get_range(x, RANGES):
        for (a, b), label in RANGES.items():
            if x >= a and x < b:
                return label

    df['custody'] = df['custody'].apply(lambda x: get_range(x, CUSTODY_RANGES))
    df['priors_count'] = df['priors_count'].apply(lambda x: get_range(x, PRIORS_RANGES))
    df['juv_other_count'] = df['juv_other_count'].apply(lambda x: get_range(x, JUV_OTHER_RANGES))
    df['juv_fel_count'] = df['juv_fel_count'].apply(lambda x: get_range(x, JUV_FEL_RANGES))
    df['juv_misd_count'] = df['juv_misd_count'].apply(lambda x: get_range(x, JUV_MISD_RANGES))

    df['is_recid'] = df['is_violent_recid'].apply(lambda x: 'Yes' if x == 1 else 'No')
    df['is_violent_recid'] = df['is_violent_recid'].apply(lambda x: 'Yes' if x == 1 else 'No')
    df['two_year_recid'] = df['two_year_recid'].apply(lambda x: 'Yes' if x == 1 else 'No')
    df['charge_degree'] = df['c_charge_degree'].apply(lambda x: 'Felony' if x == 'F' else 'Misdemeanor')
    del df['c_charge_degree']

    # df['custody'], custody_bins = pd.cut(df['custody'], bins = 10, labels = False, retbins = True)
    # df['priors_count'], custody_bins = pd.cut(df['10'], bins = 10, labels = False, retbins = True)
    print(f'Loaded {len(df)} records')
    return df