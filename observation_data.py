import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, FuncFormatter
import seaborn as sns
import scipy.stats as stats
import Data
import Data_btc
from cross_moment_estimator import estimator_bipower_spot_variance

def data_extract(file):
    data_brut = pd.read_csv(file)
    data_clean = data_brut[["timestamp","close"]].copy()
    data_clean["timestamp"] = pd.to_datetime(data_clean["timestamp"])
    data_clean.rename(columns={"timestamp": "Day"}, inplace=True)
    data_clean.set_index("Day", inplace=True)
    return data_clean

def retunrs1min(data):
    data['datetime'] = pd.to_datetime(data['datetime'])
    data["1min_returns"] = data["price"].pct_change()
    data.set_index('datetime', inplace=True)
    return data

def retunrs1day(df):
    df_with_returns = df.copy()
    df_with_returns["returns"] = df_with_returns["price"].pct_change()
    return df_with_returns

def from_1min_to_1day(dfilt):
    df_copy = dfilt.copy()
    df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
    df_copy.set_index('datetime', inplace=True)
    daily_close = df_copy.resample('D').last()
    daily_close.dropna(inplace=True) # pour ne pas avoir les weekend
    return daily_close

def log_variance_change(df):
    temp_df = df.copy()
    temp_df = temp_df.drop_duplicates(subset=['date'])
    print(temp_df)
    temp_df['log-variance change'] = temp_df['log-variance'] - temp_df['log-variance'].shift(1)
    return temp_df

def returns_tstats(df_day, df_min):
    df_day['date'] = df_day.index.date
    df_variance_daily = df_min[['date', 'smoothed_variance']].drop_duplicates()
    df_variance_daily.index = df_variance_daily.index.date
    print(df_day.index.equals(df_variance_daily.index))
    # to compute the t-Stat we divide the daily returns by the volatility of the previous day
    df_day['returns_tstat'] = df_day['returns'] / np.sqrt(df_variance_daily['smoothed_variance'].shift(1))
    print(df_day['returns_tstat'])
    return df_day

def vol_tstats(df):
    conditional_vol  = 0.14
    df['vol_tstats'] = df['log-variance change'] / conditional_vol
    return df

def daily_vol_estimation(df):
    hourly_variances = [] # list of hourly variances
    datetime_indices = [] # list of datetime of each hourly calcul
    for i in range(0, len(df) - 64, 65):
        hourly_variance = estimator_bipower_spot_variance(df['log_returns'].iloc[i:i+65].tolist())
        hourly_variances.append(hourly_variance)
        datetime_indices.append(df.index[i])
        
    print('hourly_variances\n', hourly_variances)
    temp_df = pd.DataFrame({'datetime': datetime_indices, 'hourly_variance': hourly_variances})
    temp_df['date'] = temp_df['datetime'].dt.date
    
    """ For log-variance: 
    The logarithmic variance changes are close-to-close. 
    They are obtained by virtue of spot (hourly) variance estimates making use of intradaily price data 
    in the last hour of the trading day before being scaled up to a daily value"""
    last_hour_variance_each_day = temp_df.groupby('date')['hourly_variance'].last()
    last_hour_variance_each_day_scaled = last_hour_variance_each_day * 6
    last_hour_variance_each_day_scaled_df = last_hour_variance_each_day_scaled.reset_index()
    last_hour_variance_each_day_scaled_df.columns = ['date', 'last_hour_variance_each_day_scaled']
    print('last_hour_variance_each_day_scaled_df\n', last_hour_variance_each_day_scaled_df)
    
    """ For Returns t-Stat standard deviation:
    To obtain the daily estimates used in this section, we scale up the hourly estimates to a 
    daily value and average them for every 6.5-hour trading day."""
    daily_variances = temp_df.groupby('date')['hourly_variance'].mean()
    print('daily_variances \n', daily_variances)
    daily_variances_scaled = daily_variances * 6
    daily_variances_scaled_df = daily_variances_scaled.reset_index()
    daily_variances_scaled_df.columns = ['date', 'daily_variances_scaled']
    
    """ Again for returns t-Stat:
    We then apply an exponential smoother with a 40-day lag to reduce measurement error."""
    smoothed_variances = daily_variances_scaled.ewm(span=40).mean()
    smoothed_variances_df = smoothed_variances.reset_index()
    smoothed_variances_df.columns = ['date', 'smoothed_variance']
    df['date'] = df.index.date

    original_datetime_index = df.index # pour garder le datetime en index du dataframe final

    df_merged = pd.merge(df, smoothed_variances_df, on='date', how='left')
    df_merged = pd.merge(df_merged, daily_variances_scaled_df, on='date', how='left')
    df_merged = pd.merge(df_merged, last_hour_variance_each_day_scaled_df, on='date', how='left')
    df_merged.index = original_datetime_index
    return df_merged
        
def table1(df):
    df['abs_standardized_rdt'] = df['returns_tstat'].abs()
    largest_rdt = df.sort_values(['abs_standardized_rdt'], ascending=False)
    print(largest_rdt)
    df.drop('abs_standardized_rdt', axis=1)
    largest_rdt_table = largest_rdt.drop('abs_standardized_rdt', axis=1)
    largest_rdt_table = largest_rdt_table.head(30)
    largest_rdt_table = largest_rdt_table.sort_index()
    return largest_rdt_table

def price_and_vol_jumps_identification(df): # test for a 5% significance level
    price_jumps = []
    vol_jumps = []
    critical_value = stats.norm.ppf(0.025)
    t_statistics_returns = df['returns_tstat']
    daily_returns = df['returns']
    t_statistics_returns_vol = df['vol_tstats']
    log_variance_changes = df['log-variance change']
    for date, (price_t_stat, daily_return) in zip(df.index, zip(t_statistics_returns, daily_returns)):
        if (abs(price_t_stat) > abs(critical_value)):
               price_jumps.append({'date': date, 'returns': daily_return})
    for date, (variance_t_stat, log_variance_change) in zip(df.index, zip(t_statistics_returns_vol, log_variance_changes)):
        if (abs(variance_t_stat) > abs(critical_value)):
               vol_jumps .append({'date': date, 'log-variance change': log_variance_change})
    df_price_jumps = pd.DataFrame(price_jumps)
    df_vol_jumps = pd.DataFrame(vol_jumps)
    df_vol_and_price_jumps = pd.concat([df_price_jumps,df_vol_jumps])
    return df_vol_and_price_jumps


def co_jump_identification(df): # test for a 0.5% significance level
    co_jumps_data = []
    critical_value = stats.norm.ppf(0.0025)
    t_statistics_returns = df['returns_tstat']
    t_statistics_returns_vol = df['vol_tstats']
    daily_returns = df['returns']
    log_variance_changes = df['log-variance change']
    for date, (price_t_stat, variance_t_stat, daily_return, log_variance_change) in zip(df.index, zip(t_statistics_returns, t_statistics_returns_vol, daily_returns, log_variance_changes)):
        if (abs(price_t_stat) > abs(critical_value)) and \
           (abs(variance_t_stat) > abs(critical_value)):
               co_jumps_data.append({'date': date, 'returns': daily_return, 'log-variance change': log_variance_change})
    df_co_jumps = pd.DataFrame(co_jumps_data)
    return df_co_jumps
               
      
def figure1(df):
    sns.kdeplot(df['returns'], bw_adjust=0.5, label='Price co-jumps')
    sns.kdeplot(df['log-variance change'], bw_adjust=0.5, label='Variance co-jumps')
    plt.xlabel('Co-jump size (%)')
    plt.ylabel('Estimated probability density function')
    plt.title('Estimated PDFs of Price Co-jumps')
    plt.legend()
    plt.show()
    
def figure6(df):
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    axs[0].plot(df.index, df['returns'], color='blue')
    axs[0].set_title('Daily Returns')
    axs[0].set_ylabel('Return (%)')
    axs[0].grid(True)
    
    axs[0].xaxis.set_major_locator(mdates.YearLocator())
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
    
    axs[1].plot(df.index, df['last_hour_variance_each_day_scaled'], color='blue')
    axs[1].set_yscale('log') 
    axs[1].set_title('Daily Volatility')
    axs[1].set_ylabel('Volatility (%)')
    axs[1].yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(5))  # Pour les graduations principales
    axs[1].yaxis.set_minor_locator(plt.AutoMinorLocator(4))  # Pour les sous-divisions
    axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    axs[1].set_xlabel('Year')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def figure2(df):
    returns_mean = df_co_jumps['returns'].mean()
    returns_std = df_co_jumps['returns'].std()
    variance_mean = df_co_jumps['log-variance change'].mean()
    variance_std = df_co_jumps['log-variance change'].std()
    correlation = df_co_jumps['returns'].corr(df_co_jumps['log-variance change'])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df_co_jumps['returns'], df_co_jumps['log-variance change'], alpha=0.7)
    
    plt.text(0.7, 0.90, f'Jumps in log-price:\nMean: {returns_mean:.2f}% Std: {returns_std:.2f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.80, f'Jumps in log-variance:\nMean: {variance_mean:.2f}% Std: {variance_std:.2f}', transform=plt.gca().transAxes)
    plt.text(0.7, 0.70, f'Correlation: {correlation:.2f}', transform=plt.gca().transAxes)
    
    plt.xlabel('Returns (%)')
    plt.ylabel('Log-variance changes')
    plt.title('Scatter plot of significant (daily) co-jumps in prices and variances')
    
    plt.show()
    
def figure3(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['returns'], df['last_hour_variance_each_day_scaled'], alpha=0.7)
    
    plt.xlabel('Returns (%)')
    plt.ylabel('Volatility Levels (%)')
    plt.title('Scatter plot of significant daily price co-jumps and the volatility level at which they occur')
    
    plt.show()
    
#%% DataImport
fp = r'C:\Users\Tim\OneDrive - De Vinci\A5\S1\Gestion Quant\Projet\code'
fn = 'SP.csv'
 
df = Data.FinancialDataHandler(fp, fn)
#print(df[df['datetime'].isin('2000-01-03')])
dfilt = df.minutePrice()
print(dfilt)

#%% BTC Import Test

Data_importer = Data_btc.Data_btc()
df_futures_btc = Data_importer.get_cryptos_historical_prices()

#%% btc
df_futures_btc.rename(columns={'Open Time': 'datetime', 'Close': 'price'}, inplace=True)
df_futures_btc['price'] = pd.to_numeric(df_futures_btc['price'], errors = 'coerce')

#%% Create df of daily prices and returns, and 1min returns
main_data_sp = dfilt.copy()
df_daily_prices = from_1min_to_1day(main_data_sp ) # get daily prices from min by min df
print(df_daily_prices)

df_daily_returns = retunrs1day(df_daily_prices) # daily df with returns column
print(df_daily_returns)

df_1min_returns = retunrs1min(main_data_sp ) # df min by min with returns
print(f'df_1min_returns \n {df_1min_returns}')

#%% Log returns 1min
df_1min_returns['log_returns'] = np.log(df_1min_returns['price'] / df_1min_returns['price'].shift(1))
print(df_1min_returns) #df min by min with log returns (not in %)

#%% Create df with daily vol (by method ² at the begining of section 2)
df_min_with_daily_smoothvar = daily_vol_estimation(df_1min_returns)
print(df_min_with_daily_smoothvar)
print(df_min_with_daily_smoothvar.columns)
df_min_with_daily_smoothvar['log-variance'] = np.log(df_min_with_daily_smoothvar['last_hour_variance_each_day_scaled']*100)

#%% Returns tstat
df_tstat_returns = returns_tstats(df_daily_returns, df_min_with_daily_smoothvar)
print(df_tstat_returns)

#%% Log-variance change
df_with_log_var = log_variance_change(df_min_with_daily_smoothvar)
print(df_with_log_var)

#%%  Vol tstat
df_with_vol_tstats = vol_tstats(df_with_log_var)
df_with_vol_tstats.index = df_with_vol_tstats.index.date
print(df_with_vol_tstats)

#%% DataFrame for table 1
"""
df_with_log_var = log_variance_change(df_min_with_daily_smoothvar)
df_with_vol_tstats = vol_tstats(df_with_log_var)
df_daily_returns['date'] = df_daily_returns.index.date
df_final = pd.merge(df_daily_returns[['date', 'returns', 'daily_returns_t_stat']], 
                     df_min_with_daily_smoothvar[['date', 'log-variance change', 'vol_tsats']], 
                     on='date', 
                     how='inner')
"""
df_daily_returns.index = df_daily_returns.index.date
columns1 = df_daily_returns[['returns', 'returns_tstat']]
columns2 = df_with_vol_tstats[['log-variance','log-variance change', 'vol_tstats']]
df_final = pd.concat([columns1, columns2], axis=1)
df_final['returns'] = df_final['returns']*100
#df_final.index = df_tstat_returns.index.date
print(df_final)

#%% Table 1
df_table1 = table1(df_final)

#%% Co-jump Identification
df_pirce_and_vol_jumps = price_and_vol_jumps_identification(df_final)
df_co_jumps = co_jump_identification(df_final)

#%% Will be usefull for Figure 3 and 6
df_co_jumps_with_daily_vol = df_co_jumps.copy()

#%% And also for Figure 3 and 6
daily_vol_co_jump = df_min_with_daily_smoothvar[['date', 'last_hour_variance_each_day_scaled']].drop_duplicates()
df_final['date'] = df_final.index
#%%
df_final = df_final.merge(daily_vol_co_jump, how='inner', on='date')
df_final.index = df_final['date']
#df_final.drop('date',axis=1, inplace=True)
df_final['last_hour_variance_each_day_scaled'] = np.sqrt(df_final['last_hour_variance_each_day_scaled'])
df_co_jumps_with_daily_vol = df_co_jumps_with_daily_vol.merge(daily_vol_co_jump, how='inner', on='date')
df_co_jumps_with_daily_vol['last_hour_variance_each_day_scaled'] = np.sqrt(df_co_jumps_with_daily_vol['last_hour_variance_each_day_scaled'])*100 

#%% Figure 1
figure1(df_pirce_and_vol_jumps)

#%% Figure 6
figure6(df_final)

#%% Figure 2
figure2(df_co_jumps)

#%% Figure 3 plot
figure3(df_co_jumps_with_daily_vol)

#%% Table 1 to Latex
df_table1_rounded = df_table1.round(2)
df_table1_rounded.insert(0, 'Day', df_table1_rounded.index)
latex_table = df_table1_rounded.to_latex(index=False, header=True, column_format='cccccc', 
                          longtable=False, float_format="%.2f")
latex_table = latex_table.replace("\\toprule", "\\toprule")
latex_table = latex_table.replace("\\midrule", "\\midrule")
latex_table = latex_table.replace("\\bottomrule", "\\bottomrule")
         
# save the table in a .tex file
with open('C:/Users/Tim/OneDrive - De Vinci/A5/S1/Gestion Quant/Projet/code/table1.tex', 'w') as file:
    file.write(latex_table)









