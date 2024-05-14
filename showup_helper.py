import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import datetime
from IPython.display import display, HTML, Javascript
import warnings
warnings.filterwarnings('ignore')
import math
import uuid
import json
import re

from imblearn.over_sampling import SMOTE

# SKlearn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
# from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, TransformerMixin
from tpot import TPOTClassifier


class ShowUp(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.hello()
        # self.create_code_cell(self.code_load)
        # self.create_code_cell(self.code_2)

    def hello(self):
        message = "<p><span style='color: #7b9547; font-size: 18px;'>🍏 <B>ShowUp</B>forHealth</span>  <span style='color: #cfcfcf; font-size: 18px;'> v 0.3.1 | Prediction missed appointments in Primary Care.</span></p>"
        display(HTML(message))

    code_load = """# Importing default Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

# Hi-resolution Plots and Matplotlib inline
%config InlineBackend.figure_format = 'retina'
%matplotlib inline

# Set the maximum number of rows and columns to be displayed
warnings.filterwarnings('ignore')

# "magic commands" to enable autoreload of your imported packages
%load_ext autoreload
%autoreload 2"""

    code_2 = """# Prepare Full Dataset to date
data = su.make_full_train_data()

# Define X and y
X, y = su.define_X_y(data, 'Appointment_status')

# Train and Test Split - Saving files to disk
X_train, y_train = su.train_test_split(X, y)

# Oversampling with SMOTE
X_train_o, y_train_o = su.oversample_SMOTE(X_train, y_train)

# Scale data
X_train_o_s = su.scale_df(X_train_o, 'minmax')

# specify the model to use
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(n_jobs=-1, solver='sag')



# Evaluate the model
su.evaluate_classification_model(model, X_train_o_s, y_train_o)"""

    def create_code_cell(self, code):
        cell_id = str(uuid.uuid4())
        escaped_code = json.dumps(code)
        display(
            Javascript(
                f"""
            var code = {escaped_code};
            var cell = Jupyter.notebook.insert_cell_above('code');
            cell.set_text(code);
            cell.metadata.id = '{cell_id}';
        """
            )
        )

    def create_global_appointments_list(self, surgery_list = ['ECS', 'TCP', 'TGP', 'SMW', 'KMC', 'HPVM']):
        print("=== Processing Appointments data ==============================================================")
        full_app_list = []
        for surgery_prefix in surgery_list:
            print(f'⏺️: {surgery_prefix} -------------------------------------------------------------')
            df_list = []
            for i in range(1,10,1):
                app = pd.read_csv(f'../data/raw-data/{surgery_prefix}/{surgery_prefix}_APP{i}.csv')
                print(f"df {i} - {app.shape}, ", end=' ')
                df_list.append(app)

            appointments = pd.concat(df_list, axis=0, ignore_index=True)
            print(f'{surgery_prefix}_APPS.csv created / df shape: {appointments.shape} duplicates: {appointments.duplicated().sum()}')
            full_app_list.append(appointments)
            
            global_appointments = pd.concat(full_app_list, axis=0, ignore_index=True)
            print(f'Full Appointment List - {global_appointments.shape}')
            
        print(f'✅ Appointment List - {global_appointments.shape}')
        # Filter and drop rows with 'DROP' value
        global_appointments.to_csv('../../data-showup/data/output-data/global_apps_list.csv', index=False)
        print(f'💾 Saved as ../data/output-data/global_apps_list.csv')
        return global_appointments

    def make_full_preprocess_data(self):
        register = self.make_global_disease_register()
        register['Patient ID'] = register['Patient ID'].astype('int64')
        apps = self.create_global_appointments_list()
        
        apps_weather = self.add_weather(apps)
        apps_weather['Patient ID'] = apps_weather['Patient ID'].astype('int64')
        full_df = apps_weather.merge(register, how='left', on='Patient ID')
        print(f'↔️ Merged Appointments and Global Register - Pre-process df {full_df.shape}')
        #full_df.dropna(inplace=True)
        #print(f'❌ dropna {full_df.shape}')
        return full_df
        

    def haversine_distance(self, surgery_prefix, lat2, lon2):
        R = 6371.0  # Radius of the Earth in kilometers

        if surgery_prefix == 'ECS':
            lat1, lon1 = 51.488721, -0.191873
        elif surgery_prefix == 'SMW':
            lat1, lon1 = 51.494474, -0.181931
        elif surgery_prefix == 'TCP':
            lat1, lon1 = 51.48459, -0.171887
        elif surgery_prefix == 'HPVM':
            lat1, lon1 = 51.48459, -0.171887
        elif surgery_prefix == 'KMC':
            lat1, lon1 = 51.49807, -0.159918
        elif surgery_prefix == 'TGP':
            lat1, lon1 = 51.482652, -0.178066


        # Convert degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Differences
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # Haversine formula
        a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c
        return distance  # in kilometers
    
    def extract_rota_type(self, text):
        # HOW TO APPLY IT
        # Apply extract_role function and overwrite Rota type column
        # full_appointments['Rota type'] = full_appointments['Rota type'].apply(extract_rota_type)     
        role_map = {
        'GP': ['GP', 'Registrar', 'Urgent', 'Telephone', '111', 'FY2', 'F2', 'Extended Hours', 'GP Clinic', 'Session'],
        'Nurse': ['Nurse', 'Nurse Practitioner'], 
        'HCA': ['HCA','Health Care Assistant', 'Phlebotomy'],
        'ARRS': ['Pharmacist', 'Paramedic', 'Physiotherapist', 'Physicians Associate', 'ARRS', 'PCN'],
        }

        for role, patterns in role_map.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return role
        return 'DROP'

    def extract_ethnicity(self, text):
        # HOW TO APPLY IT
        # Apply extract_role function and overwrite Rota type column
        # full_appointments['Rota type'] = full_appointments['Rota type'].apply(extract_rota_type)     
        ethnicity_dict = {
        "White": ['Other White', 'British or Mixed British', 'Irish'],
        "Black": ['African','Other Black','Caribbean'],
        "Mixed": ['Other Mixed','White & Asian','White & Black African','White & Black Caribbean'],
        "Asian": ['Other Asian','Indian or British Indian','Pakistani or British Pakistani','Chinese', 'Bangladeshi or British Bangladeshi'],
        "Other": ['Other']
        }   

        for role, patterns in ethnicity_dict.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return role
        return 'Unknown'

    def make_global_disease_register(self, surgery_list = ['ECS', 'TCP', 'TGP', 'SMW', 'KMC', 'HPVM']):
        print('=== Preparing Global Disease Register + IMD2023 info =====================================================')
        root_dir = os.path.dirname(os.path.dirname(__file__))
        disease_register = []
        for surgery in surgery_list:
            register = f'show-up-for-health-private/data/raw-data/{surgery}/{surgery}'
            register_path = os.path.join(root_dir, register)

            idnhs = pd.read_excel(f'{register_path}_NHS_PTID.xlsx', dtype='str')
            idnhs.dropna(inplace=True)
            frail = pd.read_csv(f'{register_path}_FRAILTY.csv', dtype='str')
            dep = pd.read_csv(f'{register_path}_DEPRESSION.csv', dtype='str')
            obesity = pd.read_csv(f'{register_path}_OBESITY.csv', dtype='str')
            chd = pd.read_csv(f'{register_path}_IHD.csv', dtype='str')
            dm = pd.read_csv(f'{register_path}_DM.csv', dtype='str')
            hpt = pd.read_csv(f'{register_path}_HPT.csv', dtype='str')
            ndhg = pd.read_csv(f'{register_path}_NDHG.csv', dtype='str')
            smi = pd.read_csv(f'{register_path}_SMI.csv', dtype='str')

            ptid = idnhs.merge(frail, how='left', on='NHS number')
            #ptid = ptid.drop(columns='NHS number')

            register = (ptid.merge(dep, how='left', on='Patient ID')
                        .merge(obesity, how='left', on='Patient ID')
                        .merge(chd, how='left', on='Patient ID')
                        .merge(dm, how='left', on='Patient ID')
                        .merge(hpt, how='left', on='Patient ID')
                        .merge(ndhg, how='left', on='Patient ID')
                        .merge(smi, how='left', on='Patient ID')
                        .fillna(0)
                        )
            print(f'💊 {surgery} Disease Register completed')
            # Add IMD and distance from station
            imd_path = os.path.join(root_dir, 'show-up-for-health-private/data/imd-master/imd_master.csv')
            imd = pd.read_csv(imd_path)

            full_register = register.merge(imd, how='left', on='Postcode')
            print(f'🔸 {surgery} IMD2023')
            full_register['distance_from_surg'] = full_register.apply(lambda row: self.haversine_distance(surgery, row['Latitude'], row['Longitude']), axis=1)
            disease_register.append(full_register)

        global_register = pd.concat(disease_register, axis=0, ignore_index=True)
        print(f"🦠 Concat Registers into ONE REGISTER")
        #global_register.dropna(inplace=True)
        #print(f'❌ Dropped NaN')
        output_path = f'show-up-for-health-private/data/output-data/global_disease_register.csv'
        register_out = os.path.join(root_dir, output_path)
        global_register.to_csv(register_out, index=False)
        print(f'✅ Global Disease Register Saved to output-data: {global_register.shape}')
        print()
        return global_register



    # Mapping functions
    def fix_appointment_status(self, status):
        """
        Function to categorize appointment statuses into binary format.

        Args:
            status (str): Appointment status.

        Returns:
            int: Returns 1 if status is in ['In Progress', 'Arrived', 'Patient Walked Out', 'Finished', 'Waiting'], 0 if status is 'Did Not Attend' or 'ERROR' otherwise.
        """
        if status in ['In Progress', 'Arrived', 'Patient Walked Out', 'Finished', 'Waiting']:
            return 1
        elif status == 'Did Not Attend':
            return 0
        

    def add_weather(self, global_apps):
        
        root_dir = os.path.dirname(os.path.dirname(__file__))
        weather = f'data-showup/data/weather/weather.csv'
        weather_path = os.path.join(root_dir, weather)        
        weather = pd.read_csv(weather_path)
        
        weather['app datetime'] = pd.to_datetime(weather['app datetime'])


        global_apps['app datetime'] = pd.to_datetime(global_apps['Appointment date'] + ' ' +
                                          global_apps['Appointment time'].str.split(expand=True)[0])


        #global_apps['app datetime'] = pd.to_datetime(global_apps['app datetime'])

        global_apps_weather = global_apps.merge(weather, how='left', on='app datetime')
        global_apps_weather.to_csv('../../data-showup/data/output-data/global_apps_with_weather.csv', index=False)
        print(f'🌤️ Weather Added to Apps {global_apps_weather.shape}')
        return global_apps_weather

    def add_disease_register(self):
        
        root_dir = os.path.dirname(os.path.dirname(__file__))
        disease = f'data-showup/data/output-data/global_disease_register.csv'
        disease_path = os.path.join(root_dir, disease)
        apps = 'data-showup/data/output-data/global_apps_with_weather.csv'
        apps_path = os.path.join(root_dir, apps)
        
        
        disease = pd.read_csv(disease_path)
        global_apps = pd.read_csv(apps_path)


        raw_train_data = global_apps.merge(disease, how='left', on='Patient ID')
        raw_train_data.to_csv('full_raw_train_data.csv')
        print(f'😷 Disease Register Added to Apps {raw_train_data.shape} - saved as full_raw_train_data.csv')
        print()
        return raw_train_data

    def feature_engineering(self, df):
        print('=== Feature Engineering =============================================================')
        print('➡️ Rename Columns')
        df.rename(columns={'Appointment status': 'Appointment_status', \
                        'Booked by': 'Booked_by','Appointment time': 'Appointment_time', \
                        'Rota type': 'Rota','Age in years': 'Age'}, inplace=True)
        
        # Convert Date Columns to DATETIME
        print('➡️ Columns to Datetime')
        datetime_cols = ['Appointment booked date', 'Appointment date', 'Registration date']
        for datetime_col in datetime_cols:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        print('➡️ Fix Appointment Time')
        df['Appointment_time'] = df['Appointment_time'].astype('str')
        df['Appointment_time'] = df['Appointment_time'].str.split(':').str[0].astype(int)   
        print('➡️ Map Appointment Status')
        df['Appointment_status'] = df['Appointment_status'].map(self.fix_appointment_status).astype(int)
        print('➡️ book_to_app_days')
        df['book_to_app_days'] = (df['Appointment date'] - df['Appointment booked date']).dt.total_seconds() / (60*60*24)

        # df['Appointment_time'] = df['Appointment_time'].astype('str')
        # df['Appointment_time'] = df['Appointment_time'].str.split(':').str[0].astype(int)
        print('➡️ booked_by_clinician')
        df['booked_by_clinician'] = (df['Booked_by'] == df['Clinician']).astype(int)
        
        print('➡️ Extract Rota Types')
        df['Rota'] = df['Rota'].map(self.extract_rota_type)
        
        print('➡️ registered_for_months')
        df['registered_for_months'] = ((pd.Timestamp.now() - df['Registration date']).dt.total_seconds() / (60*60*24*7*30)).apply(np.ceil)
        
        # print('week month day_of_week')
        # df[['week', 'month', 'day_of_week']] = df['Appointment date'].apply(lambda x: pd.Series([x.week, x.month, x.dayofweek]))
        print('➡️ Week')
        df['week'] = df['Appointment date'].dt.week
        print('➡️ month')
        df['month'] = df['Appointment date'].dt.month
        print('➡️ day of week')
        df['day_of_week'] = df['Appointment date'].dt.dayofweek

        
        type_cast_cols = {
            'Appointment_time': 'int',
            'Age': 'int',
            'FRAILTY': 'float',
            'DEPRESSION': 'int',
            'OBESITY': 'int',
            'IHD': 'int',
            'DM': 'int',
            'HPT': 'int',
            'NDHG': 'int',
            'SMI': 'int',
        }
        for col, col_type in type_cast_cols.items():
            df[col] = df[col].astype(col_type)
        print('➡️ Convert Cyclical data')
        # Converting Weeks to Cyclical data
        cyclical_column = 'week'
        weeks_in_a_year = 52
        df['sin_'+cyclical_column] = np.sin(2*np.pi*df[cyclical_column]/weeks_in_a_year)
        df['cos_'+cyclical_column] = np.cos(2*np.pi*df[cyclical_column]/weeks_in_a_year)
        df.drop(cyclical_column, axis=1, inplace=True)

        # Converting Appointment_time to Cyclical data
        cyclical_column = 'Appointment_time'
        hrs_day = 24
        df['sin_'+cyclical_column] = np.sin(2*np.pi*df[cyclical_column]/hrs_day)
        df['cos_'+cyclical_column] = np.cos(2*np.pi*df[cyclical_column]/hrs_day)
        df.drop(cyclical_column, axis=1, inplace=True)

        # Convertingmonth to Cyclical data
        cyclical_column = 'month'
        months_in_a_year = 12
        df['sin_'+cyclical_column] = np.sin(2*np.pi*df[cyclical_column]/months_in_a_year)
        df['cos_'+cyclical_column] = np.cos(2*np.pi*df[cyclical_column]/months_in_a_year)
        df.drop(cyclical_column, axis=1, inplace=True)

        cyclical_column = 'day_of_week'
        day_per_week = 7
        df['sin_'+cyclical_column] = np.sin(2*np.pi*df[cyclical_column]/day_per_week)
        df['cos_'+cyclical_column] = np.cos(2*np.pi*df[cyclical_column]/day_per_week)
        df.drop(cyclical_column, axis=1, inplace=True)
        print('➡️ Adding NO Shows Column')
        # Filter to only rows where status = 0 (no-show)
        noshow = df[df['Appointment_status'] == 0]

        # Group by Patient ID and count no-shows
        no_show_count = noshow.groupby('Patient ID')['Appointment_status'].count().reset_index(name='No_shows')
        df = df.merge(no_show_count, how='left', on='Patient ID').fillna(0)
        
        print('➡️ Drop Column no longer needed')
        df.drop(columns=['Appointment booked date', 'Appointment date', 'Booked_by', 'Clinician', 'app datetime', 'Postcode', 'Registration date', 'Language', 'Latitude', 'Longitude'], inplace=True)
        
        pre_drop = df.shape[0]
        boolean_mask = (df['Rota'] != 'DROP')
        # Applying the boolean filteraing
        df = df[boolean_mask].reset_index(drop=True)
        df.reset_index(inplace=True, drop=True)
        post_drop = df.shape[0]
        print(f'➡️ Rows dropped from Rotas other than spec: {pre_drop - post_drop}')

        pre_drop = df.shape[0]
        df.drop(df[df['book_to_app_days'] < 0].index, inplace=True)
        df.reset_index(inplace=True, drop=True)
        post_drop = df.shape[0]
        print(f'➡️ Rows from with Negative book_to_app_days: {pre_drop - post_drop}')
        
        print(f'➡️ Labelencode Column Sex')
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])
        
        print(f'➡️ OneHotEncode Column Rota')
        # OneHotEncode Rota
        ohe = OneHotEncoder(handle_unknown='ignore')
        encoded = ohe.fit_transform(df[['Rota']]).toarray()
        # Create feature names manually
        feature_names = [f"Rota_{category}" for category in ohe.categories_[0]]

        # Convert the encoded array back into a DataFrame
        encoded_data = pd.DataFrame(encoded, columns=feature_names)
        # Concatenate the original DataFrame and the encoded DataFrame
        df = pd.concat([df, encoded_data], axis=1)
        # Drop the original column
        df = df.drop(['Rota'], axis=1)  
        print(f'➡️ Extract Ethnicity Category')
        df['Ethnicity category'] = df['Ethnicity category'].apply(self.extract_ethnicity)    
        
        print(f'➡️ OneHotEncode Ethnicity')
        # OneHotEncode Rota
        ohe = OneHotEncoder(handle_unknown='ignore')
        encoded = ohe.fit_transform(df[['Ethnicity category']]).toarray()
        # Create feature names manually
        feature_names = [f"Ethnicity_{category}" for category in ohe.categories_[0]]
        # Convert the encoded array back into a DataFrame
        encoded_data = pd.DataFrame(encoded, columns=feature_names)
        # Concatenate the original DataFrame and the encoded DataFrame
        df = pd.concat([df, encoded_data], axis=1)
        # Drop the original column
        df = df.drop(['Ethnicity category'], axis=1)
        df.to_csv('full_train_dataset_processed.csv', index=False)
        print('✅ Full Train Dataset Complete 💾 Saved to ')
        return df

    def make_full_train_dataset(self):
        data = self.make_full_preprocess_data()
        full_df = self.feature_engineering(data)
        full_df.to_csv('full_train_dataset.csv', index=False)
        return full_df

    # Transform database function
    def transform_data(self, surgery_prefix):
        """f
        Function to read and process raw data, perform transformations and save it into a .csv file.

        Args:
            surgery_prefix (str): Prefix of the file names to read data from.

        Returns:
            DataFrame: Returns a transformed dataframe.
        """
        root_dir = os.path.dirname(os.path.dirname(__file__))

        disease_register = f'show-up-for-health-private/data/practice-register/{surgery_prefix}_disease_register.csv'
        register_path = os.path.join(root_dir, disease_register)

        appointment_data = f'show-up-for-health-private/data/raw-data/{surgery_prefix}/{surgery_prefix}_APPS.csv'
        apps_path = os.path.join(root_dir, appointment_data)

        register_df = pd.read_csv(register_path)
        app_df = pd.read_csv(apps_path)

        
        weather_data = f'show-up-for-health-private/data/weather/weather.csv'
        weather_path = os.path.join(root_dir, weather_data)
        weather = pd.read_csv(weather_path)
        weather['app datetime'] = pd.to_datetime(weather['app datetime'])
        

        app_df['app datetime'] = pd.to_datetime(app_df['Appointment date'] + ' ' +
                                          app_df['Appointment time'].str.split(expand=True)[0])


        app_df['app datetime'] = pd.to_datetime(app_df['app datetime'])

        app_df = app_df.merge(weather, on='app datetime')
        app_df.to_csv('ECS_post_weather_merge.csv')
        print('💾 Weatehr merge CSV Saved to Output Data - remove saves after debug')

        # Merge the Appointment Dataframe with the Disease Register Dataframe
        print(f'Shape in: {app_df.shape}')
        size_in = app_df.shape[0]
        df = pd.merge(app_df, register_df, on='Patient ID', how='left')
        print(f'Post merge shape: {df.shape}')
        df.to_csv('ECS_post_disease_register_merge.csv')
        print('💾 Disease Register merge CSV Saved to Output Data - remove saves after debug')

        # Rename specific columns to allow manipulation
        df.rename(columns={'Appointment status': 'Appointment_status', \
                        'Booked by': 'Booked_by','Appointment time': 'Appointment_time', \
                        'Rota type': 'Rota','Age in years': 'Age'}, inplace=True)

        # Convert Date Columns to DATETIME
        datetime_cols = ['Appointment booked date', 'Appointment date', 'Registration date']
        df[datetime_cols] = df[datetime_cols].apply(pd.to_datetime)

        df['Appointment_status'] = df['Appointment_status'].map(self.fix_appointment_status).astype(int)
        df['Rota'] = df['Rota'].map(self.fix_rota_type).astype(str)

        df['book_to_app_days'] = (df['Appointment date'] - df['Appointment booked date']).dt.total_seconds() / (60*60*24)

        df['Appointment_time'] = df['Appointment_time'].str.split(':').str[0].astype(int)

        df['booked_by_clinician'] = (df['Booked_by'] == df['Clinician']).astype(int)

        df[['week', 'month', 'day_of_week']] = df['Appointment date'].apply(lambda x: pd.Series([x.week, x.month, x.dayofweek]))

        df['registered_for_months'] = ((pd.Timestamp.now() - df['Registration date']).dt.total_seconds() / (60*60*24*7*30)).apply(np.ceil)

        df.fillna(0)

        cat_cols = ['Booked_by', 'Clinician', 'Sex']
        df[cat_cols] = df[cat_cols].astype('category')

        pre_drop = df.shape[0]
        boolean_mask = (df['Rota'] != 'DROP')
        # Applying the boolean filteraing
        df = df[boolean_mask].reset_index(drop=True)
        df.reset_index(inplace=True, drop=True)
        post_drop = df.shape[0]
        print(f'Rows dropped from Rotas other than spec: {pre_drop - post_drop}')

        pre_drop = df.shape[0]
        df.drop(df[df['book_to_app_days'] < 0].index, inplace=True)
        post_drop = df.shape[0]
        print(f'Rows from with Negative book_to_app_days: {pre_drop - post_drop}')
        df.dropna(inplace=True)
        type_cast_cols = {
            'Appointment_time': 'int',
            'Age': 'int',
            'FRAILTY': 'float',
            'DEPRESSION': 'int',
            'OBESITY': 'int',
            'IHD': 'int',
            'DM': 'int',
            'HPT': 'int',
            'NDHG': 'int',
            'SMI': 'int',
        }
        for col, col_type in type_cast_cols.items():
            df[col] = df[col].astype(col_type)

        # Converting Weeks to Cyclical data
        cyclical_column = 'week'
        weeks_in_a_year = 52
        df['sin_'+cyclical_column] = np.sin(2*np.pi*df[cyclical_column]/weeks_in_a_year)
        df['cos_'+cyclical_column] = np.cos(2*np.pi*df[cyclical_column]/weeks_in_a_year)
        df.drop(cyclical_column, axis=1, inplace=True)

        # Converting Appointment_time to Cyclical data
        cyclical_column = 'Appointment_time'
        hrs_day = 24
        df['sin_'+cyclical_column] = np.sin(2*np.pi*df[cyclical_column]/hrs_day)
        df['cos_'+cyclical_column] = np.cos(2*np.pi*df[cyclical_column]/hrs_day)
        df.drop(cyclical_column, axis=1, inplace=True)

        # Convertingmonth to Cyclical data
        cyclical_column = 'month'
        months_in_a_year = 12
        df['sin_'+cyclical_column] = np.sin(2*np.pi*df[cyclical_column]/months_in_a_year)
        df['cos_'+cyclical_column] = np.cos(2*np.pi*df[cyclical_column]/months_in_a_year)
        df.drop(cyclical_column, axis=1, inplace=True)

        cyclical_column = 'day_of_week'
        day_per_week = 7
        df['sin_'+cyclical_column] = np.sin(2*np.pi*df[cyclical_column]/day_per_week)
        df['cos_'+cyclical_column] = np.cos(2*np.pi*df[cyclical_column]/day_per_week)
        df.drop(cyclical_column, axis=1, inplace=True)

        df['Ethnicity category'] = df['Ethnicity category'].fillna('Unknown')

        # Define a mapping function
        def map_ethnicity(ethnicity):
            if 'White' in ethnicity:
                return 'White'
            elif 'Black' in ethnicity:
                return 'Black'
            elif 'Asian' in ethnicity or 'Indian' in ethnicity or 'Pakistani' in ethnicity or 'Chinese' in ethnicity:
                return 'Asian'
            elif 'Mixed' in ethnicity:
                return 'Mixed'
            else:
                return 'Other'

        # Apply the mapping function to the "Ethnicity" column
        df['Ethnicity category'] = df['Ethnicity category'].apply(map_ethnicity)

        # OneHotEncode Rota
        ohe = OneHotEncoder(handle_unknown='ignore')
        encoded = ohe.fit_transform(df[['Rota']]).toarray()
        # Create feature names manually
        feature_names = [f"Rota_{category}" for category in ohe.categories_[0]]

        # Convert the encoded array back into a DataFrame
        encoded_data = pd.DataFrame(encoded, columns=feature_names)
        # Concatenate the original DataFrame and the encoded DataFrame
        df = pd.concat([df, encoded_data], axis=1)
        # Drop the original column
        df = df.drop(['Rota'], axis=1)

        encoded = ohe.fit_transform(df[['Ethnicity category']]).toarray()
        # Create feature names manually
        feature_names = [f"Ethnicity_{category}" for category in ohe.categories_[0]]

        # Convert the encoded array back into a DataFrame
        encoded_data = pd.DataFrame(encoded, columns=feature_names)
        # Concatenate the original DataFrame and the encoded DataFrame
        df = pd.concat([df, encoded_data], axis=1)
        # Drop the original column
        df = df.drop(['Ethnicity category'], axis=1)
        df = df.drop(['Language'], axis=1)

        # Drop the original column
        df = df.drop(['Rota'], axis=1)

        # Label Encode Sex
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])
        df.drop(columns=['Appointment booked date', 'Appointment date', 'Booked_by', 'Clinician', 'Postcode', 'Registration date', 'Latitude', 'Longitude', 'app datetime'], inplace=True)

        pre_drop = df.shape[0]
        df.drop_duplicates(inplace=True, ignore_index=True)
        df.dropna(inplace=True)
        post_drop = df.shape[0]
        print(f'Rows Dropped Duplicates + NaN: {pre_drop - post_drop}')

        df.reset_index(inplace=True, drop=True)
        print(f'Shape Out: {df.shape}')

        root_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = f'data/output-data/{surgery_prefix}_transformed.csv'
        output_path = os.path.join(root_dir, output_dir)

        df.to_csv(output_path, index=False)

        size_out = df.shape[0]
        print(f'Rows dropped: {size_in - size_out}')
        return df

    def scale_df(self, df, scaler='minmax'):
        """
        Function to scale the numerical features of a dataframe.

        Args:
            df (DataFrame): The dataframe to scale.
            scaler (str, optional): The type of scaling method to use. Can be 'standard', 'minmax', or 'robust'. Default is 'minmax'.

        Returns:
            DataFrame: Returns a dataframe with the numerical features scaled.
        """
        if scaler == 'standard':
            scaler = StandardScaler()
        elif scaler == 'minmax':
            scaler = MinMaxScaler()
        elif scaler == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError('Invalid scaler type. Choose "standard" or "minimax".')

        # Get the column headers
        column_headers = df.columns
        # Fit the scaler to the data and transform the data
        scaled_values = scaler.fit_transform(df)

        # Convert the transformed data back to a DataFrame, preserving the column headers
        scaled_df = pd.DataFrame(scaled_values, columns=column_headers)

        print(f'✅ Data Scaled: {scaler} - {scaled_df.shape}')
        return scaled_df


    def oversample_SMOTE(self, X_train, y_train, sampling_strategy='auto', k_neighbors=5, random_state=42):
        """
        Oversamples the minority class in the provided DataFrame using the SMOTE (Synthetic Minority Over-sampling Technique) method.

        Parameters:
        ----------
        X_train : Dataframe
            The input DataFrame which contains the features and the target variable.
        y_train : Series
            The name of the column in df that serves as the target variable. This column will be oversampled.
        sampling_strategy : str or float, optional (default='auto')
            The sampling strategy to use. If 'auto', the minority class will be oversampled to have an equal number
            of samples as the majority class. If a float is provided, it represents the desired ratio of the number
            of samples in the minority class over the number of samples in the majority class after resampling.
        k_neighbors : int, optional (default=5)
            The number of nearest neighbors to use when constructing synthetic samples.
        random_state : int, optional (default=0)
            The seed used by the random number generator for reproducibility.

        Returns:
        -------
        X_res : DataFrame
            The features after oversampling.
        y_res : Series
            The target variable after oversampling.

        Example:
        -------
        >>> df = pd.DataFrame({'feature1': np.random.rand(100), 'target': np.random.randint(2, size=100)})
        >>> oversampled_X, oversampled_y = oversample_df(df, 'target', sampling_strategy=0.6, k_neighbors=3, random_state=42)
        """

        # Define the SMOTE instance
        smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)

        # Apply the SMOTE method
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f'✅ Data Oversampled: SMOTE - X_train:{X_train_res.shape} y_train:{y_train_res.shape}')

        return X_train_res, y_train_res


    def evaluate_classification_model(self, model, X, y, cv=5):
        """
        Evaluates the performance of a model using cross-validation, a learning curve, and a ROC curve.

        Parameters:
        - model: estimator instance. The model to evaluate.
        - X: DataFrame. The feature matrix.
        - y: Series. The target vector.
        - cv: int, default=5. The number of cross-validation folds.

        Returns:
        - None
        """
        print(model)
        # Cross validation
        scoring = {'accuracy': make_scorer(accuracy_score),
                'precision': make_scorer(precision_score, average='macro'),
                'recall': make_scorer(recall_score, average='macro'),
                'f1_score': make_scorer(f1_score, average='macro')}

        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        # Compute means and standard deviations for each metric, and collect in a dictionary
        mean_std_scores = {metric: (np.mean(score_array), np.std(score_array)) for metric, score_array in scores.items()}

        # Create a DataFrame from the mean and std dictionary and display as HTML
        scores_df = pd.DataFrame(mean_std_scores, index=['Mean', 'Standard Deviation']).T
        display(HTML(scores_df.to_html()))

        # Learning curve
        train_sizes=np.linspace(0.1, 1.0, 5)
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        # Define the figure and subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))

        axs[0].plot(train_sizes, train_scores_mean, 'o-', color="#a10606", label="Training score")
        axs[0].plot(train_sizes, test_scores_mean, 'o-', color="#6b8550", label="Cross-validation score")
        axs[0].set_xlabel("Training examples")
        axs[0].set_ylabel("Score")
        axs[0].legend(loc="best")
        axs[0].set_title("Learning curve")

        # ROC curve
        cv = StratifiedKFold(n_splits=cv)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i, (train, test) in enumerate(cv.split(X, y)):
            model.fit(X.iloc[train], y.iloc[train])
            viz = plot_roc_curve(model, X.iloc[test], y.iloc[test],
                                name='ROC fold {}'.format(i),
                                alpha=0.3, lw=1, ax=axs[1])
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        axs[1].plot(mean_fpr, mean_tpr, color='#023e8a',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.6)

        axs[1].plot([0, 1], [0, 1], linestyle='--', lw=2, color='#a10606',
                label='Chance', alpha=.6)
        axs[1].legend(loc="lower right")
        axs[1].set_title("ROC curve")

        # Show plots
        plt.tight_layout()
        plt.show()


    # Permutation feature importance
    def feature_importance(self, model, X, y):
        """
        Displays the feature importances of a model using permutation importance.

        Parameters:
        - model: estimator instance. The model to evaluate.
        - X: DataFrame. The feature matrix.
        - y: Series. The target vector.

        Returns:
        - Permutation importance plot
        """
        # Train the model
        model.fit(X, y)

        # Calculate permutation importance
        result = permutation_importance(model, X, y, n_repeats=10)
        sorted_idx = result.importances_mean.argsort()

        # Permutation importance plot
        plt.figure(figsize=(10, 5))
        plt.boxplot(result.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
        plt.title("Permutation Importances")
        plt.show()



    def sample_df(self, df, n_samples):
        """
        Samples the input DataFrame.

        Parameters:
        - df: DataFrame. The input DataFrame.
        - n_samples: int. The number of samples to generate.

        Returns:
        - resampled_df: DataFrame. The resampled DataFrame.
        """
        # Error handling: if the number of samples is greater than the DataFrame length.
        if n_samples > len(df):
            print("The number of samples is greater than the number of rows in the dataframe.")
            return None
        else:
            sampled_df = df.sample(n_samples, replace=True, random_state=42)
            print(f'Data Sampled: {sampled_df.shape}')
            return sampled_df


    def automl_tpot(self, X, y):
        # Select features and target
        features = X
        target = y

        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

        # Create a tpot object with a few generations and population size.
        tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)

        # Fit the tpot model on the training data
        tpot.fit(X_train, y_train)

        # Show the final model
        print(tpot.fitted_pipeline_)

        # Use the fitted model to make predictions on the test dataset
        test_predictions = tpot.predict(X_test)

        # Evaluate the model
        print(tpot.score(X_test, y_test))

        # Export the pipeline as a python script file
        time = datetime().now()
        root_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = f'pipelines/tpot_pipeline_{time}.csv'
        output_path = os.path.join(root_dir, output_dir)
        tpot.export(output_path)


    def train_val_test_split(self, X, y, val_size=0.2, test_size=0.2, random_state=42):
        # Calculate intermediate size based on test_size
        intermediate_size = 1 - test_size

        # Calculate train_size from intermediate size and validation size
        train_size = 1 - val_size / intermediate_size
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, train_size=train_size, random_state=random_state)

        print(f"✅ OUTPUT: X_train, X_val, X_test, y_train, y_val, y_test")
        print(f"Train Set:  X_train, y_train - {X_train.shape}, {y_train.shape}")
        print(f"  Val Set:  X_val, y_val - - - {X_val.shape}, {y_val.shape}")
        print(f" Test Set:  X_test, y_test - - {X_test.shape}, {y_test.shape}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def define_X_y(self, df, target):
        target = target

        X = df.drop(columns=target)
        y = df[target]

        print(f'X - independant variable shape: {X.shape}')
        print(f'y - dependant variable - {target}: {y.shape}')

        return X, y
    
    
# Filter to only rows where status = 0 (no-show)
# noshow = df[df['Appointment status'] == 0]

# # Group by Patient ID and count no-shows
# no_show_count = noshow.groupby('Patient ID')['Appointment status'].count().reset_index(name='No_shows')
# no_show_count.sort_values(by='No_shows', ascending=False)
