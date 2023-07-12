# 0 = male, 1 = female
# 0 = white, 1 = black, 2 = asian, 2 = native
# state codes are publicly available

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
#imputing only + hyperparameters
#Mean Absolute Error (MAE): 6.14876077601131
#Mean Squared Error (MSE): 88.51972235206492
#R-squared (R2) Score: 0.9402871163582485
#Explained Variance Score: 0.9402873251850954
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Load the demographic dataset
suicide_data = pd.read_csv(r"C:\Users\aarus\coding\suicideCCml\model_data\suicideConverted.csv")

# Load the climate change dataset (assuming it contains climate-related features)
cc_data = pd.read_csv(r"C:\Users\aarus\coding\suicideCCml\model_data\trainCC.csv")

pop_data = pd.read_csv(r"C:\Users\aarus\coding\suicideCCml\model_data\popHappened.csv")

suicide_pop_columns_merge = ["Year", "Year_Month", "race", "Race Code", "gender",  "Gender Code", "deaths", "State", "State Code"]
pop_columns_to_merge = ["Year", "Pop", "State"]

suicide_pop_merged_data = pd.merge(suicide_data[suicide_pop_columns_merge], pop_data[pop_columns_to_merge], on=["Year", "State"], how="outer",suffixes=('_suicidePop', '_pop'))

# Concatenate Year and Month columns with a forward slash separator
cc_data["Year_Month"] = cc_data['Year'].astype(str) + '/' + cc_data['Month'].astype(str).str.zfill(2)

suicide_columns_to_merge = ["Year_Month", "race", "Race Code", "gender",  "Gender Code", "deaths", "State", "State Code"]
cc_columns_to_merge = ["Year_Month", "State", "average_temp"]

# Perform the merge based on the "Year_Month" column
merged_data = pd.merge(suicide_data[suicide_columns_to_merge], cc_data[cc_columns_to_merge], on="Year_Month", how="outer", suffixes=('_suicide', '_climate'))

merged_data.drop_duplicates(subset=["Year_Month", "race", "gender", "Race Code", "Gender Code", "State_suicide"], inplace=True)

merged_data_columns_to_merge = ["Year_Month", "race", "Race Code", "gender",  "Gender Code", "deaths", "State_suicide", "State Code", "State_climate", "average_temp"]
suicide_pop_merged_data_columns_to_merge = ["Year", "Year_Month", "race", "Race Code", "gender",  "Gender Code", "deaths", "State", "State Code", "Pop"]
final_merged_data = pd.merge(merged_data[merged_data_columns_to_merge], suicide_pop_merged_data[suicide_pop_merged_data_columns_to_merge], on=["Year_Month", "State Code", "race", "Race Code", "gender", "Gender Code", "deaths"], how="outer")

final_merged_data.dropna(subset=['deaths'], inplace=True)

final_merged_data['suicide_rate'] = final_merged_data['deaths'] / final_merged_data['Pop']

X = final_merged_data[['average_temp', 'State Code', 'Race Code', 'Gender Code']]
y = final_merged_data.deaths

# Initialize the SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Impute missing values
X = imputer.fit_transform(X)

# Split the scaled and imputed dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor(
    max_depth=13,
    min_samples_leaf=10,
    max_iter=2000,
    l2_regularization=0.1
)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

predicted_temp = pd.read_csv(r"C:\Users\aarus\coding\suicideCCml\model_data\predictedTemp.csv")
for index, row in predicted_temp.iterrows():
    temp_2019 = row['temp2019']
    temp_moderate = row['tempModerateEmissions(SSP2-4.5)']
    temp_mediumHigh = row['tempMediumHighEmissions(SSP3-7.0)']
    temp_high = row['tempHighEmissions(SSP5-8.5)']
    state = row['state']

    moderate_temp_diff = temp_moderate - temp_2019
    medium_temp_diff = temp_mediumHigh - temp_2019
    high_temp_diff = temp_high - temp_2019
    
    predicted_temp.at[index, 'moderateTempDiff'] = moderate_temp_diff
    predicted_temp.at[index, 'mediumTempDiff'] = medium_temp_diff
    predicted_temp.at[index, 'highTempDiff'] = high_temp_diff

predicted_pop = pd.read_csv(r"C:\Users\aarus\coding\suicideCCml\model_data\predictedPop.csv")

for index, row in predicted_pop.iterrows():
    pop_2040 = row['2040']
    state_code = row['State Code']

for index, row in cc_data.iterrows():
    average_temp = row['average_temp']
    mod_average_temp = average_temp + moderate_temp_diff
    med_average_temp = average_temp + medium_temp_diff
    high_average_temp = average_temp + high_temp_diff
    
state_codes = {
    'Alabama': '01',
    'Alaska': '02',
    'Arizona': '04',
    'Arkansas': '05',
    'California': '06',
    'Colorado': '08',
    'Connecticut': '09',
    'Delaware': '10',
    'Florida': '12',
    'Georgia': '13',
    'Hawaii': '15',
    'Idaho': '16',
    'Illinois': '17',
    'Indiana': '18',
    'Iowa': '19',
    'Kansas': '20',
    'Kentucky': '21',
    'Louisiana': '22',
    'Maine': '23',
    'Maryland': '24',
    'Massachusetts': '25',
    'Michigan': '26',
    'Minnesota': '27',
    'Mississippi': '28',
    'Missouri': '29',
    'Montana': '30',
    'Nebraska': '31',
    'Nevada': '32',
    'New Hampshire': '33',
    'New Jersey': '34',
    'New Mexico': '35',
    'New York': '36',
    'North Carolina': '37',
    'North Dakota': '38',
    'Ohio': '39',
    'Oklahoma': '40',
    'Oregon': '41',
    'Pennsylvania': '42',
    'Rhode Island': '44',
    'South Carolina': '45',
    'South Dakota': '46',
    'Tennessee': '47',
    'Texas': '48',
    'Utah': '49',
    'Vermont': '50',
    'Virginia': '51',
    'Washington': '53',
    'West Virginia': '54',
    'Wisconsin': '55',
    'Wyoming': '56'
}

state_code_list = list(state_codes.values())

# Add a new column "State Code" based on the "State" column
cc_data['State Code'] = cc_data['State'].map(state_codes)

predicted_suicides_mod_number = []
predicted_suicides_mod_rate = []
predicted_suicides_med_number = []
predicted_suicides_high_number = []
predicted_suicides_med_rate = []
predicted_suicides_high_rate = []


for index, row in cc_data.iterrows():
    state_code = row['State Code']
    if state_code == state_code:
    #if state_code == '08' or state_code == '06' or state_code == '09':  # Display data only for Alabama (State Code '01')
        if row['Month'] == 1:  # Filter for month equal to 1
            if index in predicted_pop.index:
                pop_2040 = predicted_pop.loc[index, '2040']
                for race_code in range(2):  
                    for gender_code in range(2): 
                        if race_code == 1 and gender_code == 1:  # Filter for race and gender codes equal to 0
                            pop_2040 = predicted_pop[predicted_pop['State Code'] == 1]['2040'].values[0]

                            input_data_mod = np.array([[mod_average_temp, state_code, race_code, gender_code]])
                            input_data_med = np.array([[med_average_temp, state_code, race_code, gender_code]])
                            input_data_high = np.array([[high_average_temp, state_code, race_code, gender_code]])

                            prediction_mod = model.predict(input_data_mod)
                            prediction_med = model.predict(input_data_med)
                            prediction_high = model.predict(input_data_high)

                            

                            predicted_suicides_mod_number.append(prediction_mod)
                            pop_2040 = predicted_pop.loc[index, '2040']
                            pop_2040 = str(pop_2040).replace(",", "")  # Convert to string and replace commas
                            pop_2040 = float(pop_2040)
                            predicted_suicides_mod_rate.append((prediction_mod/pop_2040) * 100)
                            
                            predicted_suicides_med_number.append(prediction_med)
                            pop_2040 = predicted_pop.loc[index, '2040']
                            pop_2040 = str(pop_2040).replace(",", "")  # Convert to string and replace commas
                            pop_2040 = float(pop_2040)
                            predicted_suicides_med_rate.append((prediction_med/pop_2040) * 100)
                            
                            predicted_suicides_high_number.append(prediction_high)
                            pop_2040 = predicted_pop.loc[index, '2040']
                            pop_2040 = str(pop_2040).replace(",", "")  # Convert to string and replace commas
                            pop_2040 = float(pop_2040)
                            predicted_suicides_high_rate.append((prediction_high/pop_2040) * 100)

                            '''
                            print("State:", row['State Code'])
                            print("Month:", row['Month'])
                            print("2040")
                            print("Race Code:", race_code)
                            print("Gender Code:", gender_code)
                            print("Predicted Suicides (moderate):", prediction_mod)
                            prediction_mod = prediction_mod[0]
                            pop_2040 = predicted_pop.loc[index, '2040']
                            pop_2040 = str(pop_2040).replace(",", "")  # Convert to string and replace commas
                            pop_2040 = float(pop_2040)
                            print("Death Rate Mod: ", (prediction_mod/pop_2040) * 100)
                            print("Predicted Suicides (medium):", prediction_med)
                            prediction_med = prediction_med[0]
                            print("Death Rate Med: ", (prediction_med/pop_2040) * 100)
                            print("Predicted Suicides (high):", prediction_high)
                            prediction_high = prediction_high[0]
                            print("Death Rate High: ", (prediction_high/pop_2040) * 100)
                            print()'''
 


predicted_suicides_mod = {}
for state_code, prediction in zip(state_codes.values(), predicted_suicides_mod_number):
    predicted_suicides_mod[state_code] = {'State Code': state_code, 'Predicted Suicides': prediction[0]}

death_rate_mod = {}
for state_code, prediction in zip(state_codes.values(), predicted_suicides_mod_rate):
    death_rate_mod[state_code] = {'State Code': state_code, 'rate': prediction[0]}

print(predicted_suicides_mod)
print("YOOO", death_rate_mod)


suicidesArray = {'01': {'State Code': '01', 'Predicted Suicides': 10.15988602511615}, '02': {'State Code': '02', 'Predicted Suicides': 13.745470987927053}, '04': {'State Code': '04', 'Predicted Suicides': 10.994386558939269}, '05': {'State Code': '05', 'Predicted Suicides': 13.137335726176559}, '06': {'State Code': '06', 'Predicted Suicides': 11.63868054647037}, '08': {'State Code': '08', 'Predicted Suicides': 9.661660078410803}, '09': {'State Code': '09', 'Predicted Suicides': 17.0455038181433}, '10': {'State Code': '10', 'Predicted Suicides': 22.372945471629873}, '12': {'State Code': '12', 'Predicted Suicides': 15.41589712148633}, '13': {'State Code': '13', 'Predicted Suicides': 11.841369231676868}, '15': {'State Code': '15', 'Predicted Suicides': 15.207463994361287}, '16': {'State Code': '16', 'Predicted Suicides': 11.730122133065791}, '17': {'State Code': '17', 'Predicted Suicides': 8.372415373743912}, '18': {'State Code': '18', 'Predicted Suicides': 8.786078549201097}, '19': {'State Code': '19', 'Predicted Suicides': 9.491913799381132}, '20': {'State Code': '20', 'Predicted Suicides': 9.204635828273956}, '21': {'State Code': '21', 'Predicted Suicides': 6.243646880768812}, '22': {'State Code': '22', 'Predicted Suicides': 9.325611899856241}, '23': {'State Code': '23', 'Predicted Suicides': 10.154389091667523}, '24': {'State Code': '24', 'Predicted Suicides': 13.098800592849084}, '25': {'State Code': '25', 'Predicted Suicides': 10.374931616523833}, '26': {'State Code': '26', 'Predicted Suicides': 9.119274975577737}, '27': {'State Code': '27', 'Predicted Suicides': 11.835055799036882}, '28': {'State Code': '28', 'Predicted Suicides': 9.62130968027546}, '29': {'State Code': '29', 'Predicted Suicides': 9.62130968027546}, '30': {'State Code': '30', 'Predicted Suicides': 10.787160755277863}, '31': {'State Code': '31', 'Predicted Suicides': 9.218710331977}, '32': {'State Code': '32', 'Predicted Suicides': 10.901399580315838}, '33': {'State Code': '33', 'Predicted Suicides': 10.33159164912109}, '34': {'State Code': '34', 'Predicted Suicides': 16.327674324698602}, '35': {'State Code': '35', 'Predicted Suicides': 15.231030223091802}, '36': {'State Code': '36', 'Predicted Suicides': 6.149864139492505}, '37': {'State Code': '37', 'Predicted Suicides': 16.098786026675224}, '38': {'State Code': '38', 'Predicted Suicides': 11.201121062924058}, '39': {'State Code': '39', 'Predicted Suicides': 12.436380450900076}, '40': {'State Code': '40', 'Predicted Suicides': 17.589999876090214}, '41': {'State Code': '41', 'Predicted Suicides': 3.990548981839866}, '42': {'State Code': '42', 'Predicted Suicides': 11.725714442632595}, '44': {'State Code': '44', 'Predicted Suicides': 5.621121331901568}, '45': {'State Code': '45', 'Predicted Suicides': 14.741699671554496}, '46': {'State Code': '46', 'Predicted Suicides': 34.59170571101031}, '47': {'State Code': '47', 'Predicted Suicides': 9.738050444054455}, '48': {'State Code': '48', 'Predicted Suicides': -1.0683446917822683}, '49': {'State Code': '49', 'Predicted Suicides': 13.313743081393827}, '50': {'State Code': '50', 'Predicted Suicides': 13.949244927801074}, '51': {'State Code': '51', 'Predicted Suicides': 10.92282049353656}, '53': {'State Code': '53', 'Predicted Suicides': 11.102788091944348}, '54': {'State Code': '54', 'Predicted Suicides': 8.902820668536084}}
suicideArray = [data['Predicted Suicides'] for data in suicidesArray.values()]
death_rate_array = {'01': {'State Code': '01', 'rate': 0.0002009154813663859}, '02': {'State Code': '02', 'rate': 0.0016763709900710349}, '04': {'State Code': '04', 'rate': 0.00011994383499497744}, '05': {'State Code': '05', 'rate': 0.00040830436113908816}, '06': {'State Code': '06', 'rate': 2.504719542040247e-05}, '08': {'State Code': '08', 'rate': 0.00012559179616250142}, '09': {'State Code': '09', 'rate': 0.00048114348203628747}, '10': {'State Code': '10', 'rate': 0.0019215064853367967}, '12': {'State Code': '12', 'rate': 0.0014559506924204614}, '13': {'State Code': '13', 'rate': 4.0992059405015984e-05}, '15': {'State Code': '15', 'rate': 0.00011862045657506996}, '16': {'State Code': '16', 'rate': 0.0007242143857896041}, '17': {'State Code': '17', 'rate': 0.00037580831018285465}, '18': {'State Code': '18', 'rate': 7.086939457784688e-05}, '19': {'State Code': '19', 'rate': 0.00013378314023088276}, '20': {'State Code': '20', 'rate': 0.0002713004583044055}, '21': {'State Code': '21', 'rate': 0.00020588068865012952}, '22': {'State Code': '22', 'rate': 0.00019779606855694788}, '23': {'State Code': '23', 'rate': 0.00020056943204459847}, '24': {'State Code': '24', 'rate': 0.000987724744382015}, '25': {'State Code': '25', 'rate': 0.00015161596083830854}, '26': {'State Code': '26', 'rate': 0.00011778009967129684}, '27': {'State Code': '27', 'rate': 0.00011882448946660639}, '28': {'State Code': '28', 'rate': 0.0001511623252997062}, '29': {'State Code': '29', 'rate': 0.0003248072244671274}, '30': {'State Code': '30', 'rate': 0.0001696102458860319}, '31': {'State Code': '31', 'rate': 0.0007456669501980904}, '32': {'State Code': '32', 'rate': 0.0004975722313804459}, '33': {'State Code': '33', 'rate': 0.0002545748441707545}, '34': {'State Code': '34', 'rate': 0.0011717437014074124}, '35': {'State Code': '35', 'rate': 0.00016083432864807143}, '36': {'State Code': '36', 'rate': 0.00028909002506877224}, '37': {'State Code': '37', 'rate': 7.712551935103144e-05}, '38': {'State Code': '38', 'rate': 8.848396916203134e-05}, '39': {'State Code': '39', 'rate': 0.0011727378338678585}, '40': {'State Code': '40', 'rate': 0.00014968250864218829}, '41': {'State Code': '41', 'rate': 8.989670694055482e-05}, '42': {'State Code': '42', 'rate': 0.0002270647046108386}, '44': {'State Code': '44', 'rate': 4.3883640459371375e-05}, '45': {'State Code': '45', 'rate': 0.001396896449369242}, '46': {'State Code': '46', 'rate': 0.0005445367149984418}, '47': {'State Code': '47', 'rate': 0.0009336291162739451}, '48': {'State Code': '48', 'rate': -1.3655302232921978e-05}, '49': {'State Code': '49', 'rate': 3.327112161952628e-05}, '50': {'State Code': '50', 'rate': 0.0003210901572782666}, '51': {'State Code': '51', 'rate': 0.0018148289888158576}, '53': {'State Code': '53', 'rate': 0.00011241362617199084}, '54': {'State Code': '54', 'rate': 9.10669591261005e-05}}
death_rate_array = [data['rate'] for data in death_rate_array.values()]

print("HO", death_rate_array)
print("HO", suicideArray)
print(len(death_rate_array))
print(len(suicideArray))
