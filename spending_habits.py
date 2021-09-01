import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
sns.set(style='ticks')
import scipy.fftpack
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

def spending():
	# df = pd.read_csv('/home/dp/Downloads/simple_checking_jan_1_2018_to_present.csv')
	df = pd.read_csv('simple_checking_all_time.csv')

	df.Date = pd.to_datetime(df.Date)
	print(df.Activity)

	# Data types:
	print('Data types:\n', df.dtypes)
	print('Memory usage:\n', df.memory_usage())
	category_cols = ['Activity','Raw description','Description','Category folder','Category']
	df[category_cols] = df[category_cols].astype('category')
	df.Amount = df.Amount.astype('float32')
	print('Data types:\n', df.dtypes)
	print('Memory usage:\n', df.memory_usage())

	# Mask to remove all Money Transfers (protected goal account transfers and C2c):
	print('df size:', len(df))
	# Category to remove: Money Transfers: removes Round-up transfer, Protected goal account transfer, Pin purchase, Signature purchase, C2c, ACH
	# I'm not sure if it should be removing the odd pin purchase, so I may need to make that exempt.
	remove_transfers_mask = df.Category == 'Money Transfers'
	print('mask:\n', remove_transfers_mask)
	print('mask True count:\n', np.sum(remove_transfers_mask))
	df_money_transfers = df.loc[remove_transfers_mask, ['Amount','Category','Activity']]
	print('df_money_transfers:', df_money_transfers.to_string())
	unique_transfers = np.unique(df_money_transfers.Activity)
	print('Transfer types:\n', unique_transfers)
	transfers_sum = df_money_transfers.Amount.sum()
	print('Transfer sum:', transfers_sum)
	protected_goal_transfer_sum = df_money_transfers[ df_money_transfers.Activity=='Protected goal account transfer'].Amount.sum()
	print('Protect goal transfer sum:', protected_goal_transfer_sum)
	C2c_transfer_sum = df_money_transfers[ df_money_transfers.Activity=='C2c'].Amount.sum()
	print('Consumer-to-consumer (C2c) transfer sum:', C2c_transfer_sum)
	# Removed 'Protected goal account transfer' (they collectively add to $-4415.55, the current amount in that account, 
	# but this shouldn't be included in the spending stats):
	df = df.loc[ df.Activity!='Protected goal account transfer', : ]
	# Remove 'Tax refund transfer', equivalent to Pgat but automated by Simple when it detects
	# a tax refund is deposited
	df = df.loc[ df.Activity!='Tax refund transfer', : ]
	# Remove any 'Investment' in the Category folder, this is the result of moving money out of
	# the checking account to an investment account such as Vanguard
	df = df.loc[ df.Category!='Stocks & Mutual Funds', : ]
	print('df without protected goal, tax refund, or investment transfers:\n', df.loc[:, ['Amount','Category','Activity']].to_string())
	category_group = df.groupby('Category')['Amount'].agg('sum')
	print('category_group:\n', category_group)
	total_income = category_group['Other Income']
	print('total_income:', total_income)
	category_group.drop('Other Income', axis=0, inplace=True)
	print('category_group:\n', category_group)
	category_group_pct = category_group / total_income * 100 * -1
	print('category_group_pct:\n', category_group_pct)
	print('Sum of all spent percentages of income:', category_group_pct.sum())

	# # Mask to remove small check deposits (<= $100)
	# remove_small_deposits_mask = (df.Activity == 'Check deposit') & (df.Amount <= 200.0)
	# print('mask:\n', remove_small_deposits_mask)
	# print('mask True count:\n', np.sum(remove_small_deposits_mask))
	# remove_small_deposits_mask = ~remove_small_deposits_mask # Invert it to keep everything but the small deposits
	# df = df[ remove_small_deposits_mask ]
	# print(df)
	# print(df.columns)

	
	df.set_index('Date', inplace=True)
	df = df[['Amount','Activity','Category folder','Category']]
	df['Balance'] = df.loc[::-1, 'Amount'].cumsum()[::-1]
	df['Duration'] = pd.to_timedelta(df.index - df.index[-1]).astype('timedelta64[s]')
	df.Duration = df.Duration/3600/24/12 # Convert to months

	print('df:\n', df.to_string())
	df_income = df[ df['Category folder'] == 'Income' ]
	df_outgoing = df[ df['Category folder'] != 'Income' ]
	print('df_income:\n', df_income.to_string())
	print('df_outgoing:\n', df_outgoing.to_string())
	total_income = df_income.Amount.sum()
	total_outgoing = df_outgoing.Amount.sum()
	print('total income:', total_income)
	print('total outgoing:', total_outgoing)
	print('income - outgoing:', total_income+total_outgoing)


	# Create transaction numbers to distinguish transcations on a particular day
	# for each date, get length, make a list from 0 to len(date), append to some transaction_num_list
	transaction_count = df.Duration.groupby('Date').agg('count')[::-1]
	print(transaction_count)

	transaction_num_list = [list(range(x))[::-1] for x in transaction_count]
	transaction_num_list = [x for sub_list in transaction_num_list for x in sub_list]

	df['TransactionNumber'] = transaction_num_list
	print('df:\n', df.to_string())

	category_spending = df.groupby(['Category']).Amount.agg('sum')
	print('Category spending:\n', category_spending)

	# df.plot(y='Balance', use_index=True)

	print(df['2018-09-06']) # Checking that the small deposits that were made on this day have been removed.

	# print('August 1, 2018:\n', df.loc['2018-08-02':'2018-07-30'])

	x1 = pd.DataFrame({'Duration':np.linspace(df.Duration.min(), df.Duration.max(), 10)})
	# lin = smf.ols.from_formula(formula='Balance ~ Duration', data=df).fit()
	lin = smf.ols(formula='Balance ~ Duration', data=df).fit()

	print('OLS Results:\n', lin.summary())

	df.reset_index(inplace=True)
	df.set_index(['Date', 'TransactionNumber'], inplace=True)
	print('df after reindexing:\n', df.to_string())

	# Q: Which direction does cumsum operate? It operates from row 0 down to the final row.
	# 
	df['Spend'] = df.loc[ df.Amount < 0, 'Amount']*-1 # *-1 makes positive
	df['Income'] = df.loc[ df.Amount > 0, 'Amount']
	df['CumulativeSpend'] = df.loc[ df.Amount < 0, 'Amount'][::-1].cumsum()[::-1]*-1 # *-1 makes positive
	df['CumulativeIncome'] = df.loc[ df.Amount > 0, 'Amount'][::-1].cumsum()[::-1]
	df_income_spend = df.loc[:, ['Spend','Income','CumulativeSpend','CumulativeIncome','Balance']]
	df_income_spend.index = df_income_spend.index.droplevel(1)
	# Q: Incorrect: 	df_income_spend = df.loc[ 'Amount' > 0, 'Amount'][::-1].cumsum()[::-1]
	# The daily resampling has cumulative values and balance for calculating % income saved,
	# but I'm not sure if I should be taking the mean or last:
	df_income_spend_rs_d = df_income_spend.loc[:,['CumulativeSpend','CumulativeIncome','Balance']].resample('D').last().interpolate(method='linear') # Multiply by -1 to make positive
	df_income_spend_rs_d['PercentIncomeSaved'] = 100 - ((df_income_spend_rs_d['CumulativeSpend'] / df_income_spend_rs_d['CumulativeIncome']) * 100)
	print('df_income_spend_rs_d:\n', df_income_spend_rs_d.to_string())

	# The monthly resampling has spend, income and balance
	df_income_spend_rs_m = df_income_spend.loc[:,['Spend','Income']].resample('M').sum()#.interpolate(method='linear')
	df_income_spend_rs_m['Saved'] = (df_income_spend_rs_m.loc[:,'Income'] - df_income_spend_rs_m.loc[:,'Spend'])
	df_income_spend_rs_m['Balance'] = round( df_income_spend_rs_m['Saved'].cumsum(), 2)
	print('df_income_spend_rs_m:\n', df_income_spend_rs_m.to_string())

	# Amount to save to reach 80% income saved:
	latest_date = df_income_spend_rs_d.index.values[-1] # 2019-12-10
	print('latest_date:\n', latest_date)
	spend_start = 20000
	spend_interval = 5000
	spend_stop = 65000
	spend_reshape_const = int((spend_stop - spend_start)/spend_interval+1)
	print('spend_reshape_const:\n', spend_reshape_const)

	earn_start = 90000
	earn_interval = 10000
	earn_stop = 180000
	earn_reshape_const = int((earn_stop - earn_start)/earn_interval+1)

	to_spend = np.arange(spend_start,spend_stop+1,spend_interval).reshape(spend_reshape_const,1).repeat(spend_reshape_const, axis=1)
	print('to_spend (broadcast to {} columns):\n{}'.format(spend_reshape_const, to_spend))

	to_earn = np.arange(earn_start, earn_stop+1, earn_interval).reshape(1,earn_reshape_const).repeat(earn_reshape_const, axis=0)
	print('to_earn (broadcast to {} rows):\n{}'.format(earn_reshape_const, to_earn))

	max_cumulative_spend = df_income_spend_rs_d.loc[latest_date, 'CumulativeSpend']
	max_cumulative_income = df_income_spend_rs_d.loc[latest_date, 'CumulativeIncome']
	pcnt_saved = 100 - (( (max_cumulative_spend + to_spend) / (max_cumulative_income + to_earn) ) * 100)
	print('Future percent saved:', pcnt_saved)

	ax = sns.heatmap(pcnt_saved, cmap="YlGnBu")
	plt.show()

	df_income_spend_rs_d.dropna(axis=0, inplace=True) # Q: First nine days are NaNs, why? Has to do with resampling and/or interpolation?
	print('Interpolated daily income and spending:\n', df_income_spend_rs_d)
	df_income_spend_rs_m.dropna(axis=0, inplace=True)

	# Q: How to create a boolean column in a dataframe and assign two different colors to True and False values
	df_income_spend_rs_d.loc[ df_income_spend_rs_d['PercentIncomeSaved']>=0, 'Colors'] = 'black'
	df_income_spend_rs_d.loc[ df_income_spend_rs_d['PercentIncomeSaved']<0, 'Colors'] = 'red'
	print('df_income_spend.Colors:\n', df_income_spend_rs_d.Colors.to_string())

	# Getting a 30-day rolling savings rate. Requires dividing CumulativeSpend max-min from CumulativeIncome max-min for the last 30 days.
	# This would be a little easier if I had a column in df_income_spend_rs_d 'Cumulative Savings' that is the difference between 
	# 'CumulativeIncome' and 'CumulativeSpend'
	df_running_pcnt_saved = 100 - ( df_income_spend_rs_d['CumulativeSpend'].rolling(30).apply(lambda x: np.max(x)-np.min(x)) / df_income_spend_rs_d['CumulativeIncome'].rolling(30).apply(lambda x: np.max(x)-np.min(x)) * 100 )
	df_running_pcnt_saved.dropna(axis=0, inplace=True)
	print('df_running_pcnt_saved:\n', df_running_pcnt_saved.to_string())

	# Plotting daily income, spend and percent saved, and monthly balance.
	fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(12,3))
	df_income_spend_rs_d.plot(use_index=True, y=['CumulativeIncome','CumulativeSpend','Balance'], ax=ax1, color=['blue','green','k'], alpha=0.5)
	df_income_spend_rs_m.plot(use_index=True, y=['Balance'], ax=ax1, color='magenta')
	df_income_spend_rs_d.plot(use_index=True, y=['PercentIncomeSaved'], ax=ax2, color='k', linewidth=1, alpha=0.5)
	ax2.scatter(x=df_income_spend_rs_d.index.values, y=df_income_spend_rs_d['PercentIncomeSaved'], c=df_income_spend_rs_d.Colors, s=1, alpha=0.5)
	df_running_pcnt_saved.plot(use_index=True, ax=ax3, color='k', linewidth=1.5, alpha=0.2)
	ax3.axhline(df_running_pcnt_saved[-1], color='green', linewidth=2, linestyle='--', alpha=0.5)
	ax3.axhline(0, color='k', linewidth=2)
	ax1.set_ylabel('Cumulative, $')
	ax2.set_ylabel('% Income Saved (All time)')
	ax3.set_ylabel('% Income Saved (30-day running)')
	ax3.set_ylim(-105,105)
	plt.subplots_adjust(wspace=0.3)
	plt.savefig('spending_habits_cumulative_income_spend_pct_saved.png', bbox_inches='tight')
	plt.show()

	fig, ax = plt.subplots(1,1, figsize=(4,3))
	df_income_spend_rs_m.plot(use_index=True, y=['Spend'], ax=ax, marker='o', alpha=0.3)
	ax.set_xlabel('Month')
	ax.set_ylabel('$')
	plt.title('Monthly Spending')
	plt.savefig('spending_habits_monthly_spending.png', bbox_inches='tight')
	plt.show()


	# Preparing to fit each spend duration between check deposits:

	plt.figure()
	plt.plot(df.Duration, df.Balance, 'o-', markersize=3, alpha=0.3)
	# plt.plot(x1.Duration, lin.predict(x1), 'r-')
	# plt.savefig('spending.png', bbox_inches='tight')
	# plt.show()

	df_income_idx = df.Amount > 0
	df_dur_start_idx = df_income_idx.shift(-1).fillna(False)
	df_dur_end_idx = df_income_idx.shift(1).fillna(False)
	# Concatenating all three indices:
	df_dur_start_end_idx = pd.concat((df_income_idx, df_dur_start_idx, df_dur_end_idx), axis=1)
	df_dur_start_end_idx.columns = ['Income','Start','End']
	print('df_dur_start_end_idx:\n', df_dur_start_end_idx.to_string())

	df['Row'] = list(range(len(df)))
	print('df[Row]:\n', df.Row)
	start_rows = df.Row[ df_dur_start_idx ].tolist()
	end_rows = df.Row[ df_dur_end_idx ].tolist()
	print('start_rows:\n', start_rows)
	print('end_rows:\n', end_rows)
	s_rows = list()
	e_rows = list()
	# Sometimes there is a start to a spending period (just after a check deposit) but no end to it
	# because another check has not been deposited. In this case, the fit will fail. To avoid this,
	# either
	# 1) the first value of start_rows can removed so that the end of a spend period occurs more recently
	#    than the start of a spend period.
	# 2) a '0' can be inserted as the first value in end_rows, such that the spend period is artificially
	#    ended but allows the end of the spend period to be more recent than the start of a spend period,
	#	 and this can be fit.
	# Option 1) results in a more accurate reflection of burn rate because 2) will fit any transient
	# spending that was just done, such as paying off a credit card balance, creating an artificially
	# high burn rate. The same problem occurs if no spending was done in the short time since the spend
	# period started, resulting in an underestimation of the future burn rate.
	if start_rows[0] < end_rows[0]: # if the most recent spend period starts but hasn't ended.
		# Option 1)
		start_rows = start_rows[1:]
		# Option 2)
		# end_rows = [0] + end_rows

	for s, e in zip(start_rows, end_rows):
		if s > e+1: # Will fit only regions that have two values (transactions) between start and end indices
		# if e > s+2: # Will fit only regions that have two values (transactions) between start and end indices
			s_rows.append(s)
			e_rows.append(e)

	print('s_rows:\n', s_rows)
	print('e_rows:\n', e_rows)

	df_pred = pd.DataFrame(columns=['x1','linpred'])
	df_x = pd.DataFrame()
	df_y = pd.DataFrame()
	burn_rate_list = list()
	time_point_list = list()
	for s, e in zip(s_rows, e_rows):
		df_dur = df.iloc[ e:s+2, : ] # For some reason I need to add 2 in order to get it to fit the first spend of the duration.
		# print('df_dur:\n', df_dur)

		d1 = df_dur.Duration[0]
		d2 = df_dur.Duration[-1]
		# print('d1:', d1)
		# print('d2:', d2)

		x1 = pd.DataFrame({'Duration': np.linspace(d1, d2, 2)})
		lin = smf.ols('Balance ~ Duration', data=df_dur).fit()
		linpred = pd.DataFrame({'Fit': lin.predict(x1)})

		# print('x1:\n', x1)
		# print('linpred:\n', linpred)

		burn_rate = (linpred.Fit[1] - linpred.Fit[0]) / (x1.Duration[1] - x1.Duration[0])
		burn_rate_list.append(burn_rate)
		time_point = x1.Duration[1]
		time_point_list.append(time_point)

		# Add linear fits of spending periods to the plot:
		plt.plot(x1.Duration, lin.predict(x1), 'r-', alpha=0.9)

		df_x = pd.concat((df_x, x1), axis=0)
		df_y = pd.concat((df_y, linpred), axis=0)

		# plt.plot(df_dur.Duration, df_dur.Balance, 'o-', markersize=2, alpha=0.2)
		# plt.plot(x1.Duration, lin.predict(x1), 'r-')
		# plt.show()

	plt.xlabel('Month')
	plt.ylabel('Checking Balance')
	plt.savefig('spending_habits_ols_fits.png', bbox_inches='tight')
	plt.show()
	# print('df_x:\n', df_x)
	# print('df_y:\n', df_y)
	df_pred = pd.concat((df_x, df_y), axis=1)
	# print('df_pred:\n', df_pred)

	# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,12))
	# df.plot(x='Duration', y='Balance', ax=ax1)
	# df_pred.plot(x='Duration', y='Fit', color='r', ax=ax1)
	# plt.show()

	burn_rate_list = [-x for x in burn_rate_list if x < 0] # Removing spurious positive values from burn rate
	print('burn_rate_list:\n', burn_rate_list)
	avg_burn_rate = round(np.mean(burn_rate_list), 2)
	max_burn_rate = round(np.max(burn_rate_list), 2)
	min_burn_rate = round(np.min(burn_rate_list), 2)
	med_burn_rate = round(np.median(burn_rate_list), 2)
	print('Average burn rate: ${}'.format(avg_burn_rate))
	print('Median burn rate: ${}'.format(med_burn_rate))
	print('Max burn rate: ${}'.format(max_burn_rate))
	print('Min burn rate: ${}'.format(min_burn_rate))
	total_income = df.Amount[ df.Amount > 0 ].sum()
	print('Total Income:', total_income)
	total_spent = df.Amount[ df.Amount < 0 ].sum()
	print('Total Spent:', total_spent)
	fraction_of_income_spent = round(total_spent / total_income * 100 * -1, 2)
	print('Fraction of income spent: {}%'.format(fraction_of_income_spent))

	# time_point_list = time_point_list[0:-1]
	# print(len(time_point_list))
	# print(len(burn_rate_list))
	print('time_point_list:\n', time_point_list)

	# Plot burn rate vs time
	plt.scatter(time_point_list, burn_rate_list, marker='o', color='green', alpha=0.5)
	plt.xlabel('Month')
	plt.ylabel('Burn Rate, $/month')
	plt.savefig('spending_habits_burn_rate_vs_time.png', bbox_inches='tight')
	plt.show()

	# Plot histogram of burn rates (linear fits between paychecks)
	plt.hist(burn_rate_list, bins=20, linewidth=1, edgecolor='green', color='green', alpha=0.5)
	plt.xlabel('Burn rate, $/month')
	plt.ylabel('Count')
	plt.savefig('spending_habits_burn_rate_histogram.png', bbox_inches='tight')
	plt.show()

	df_income_spend_rs_d.to_csv('/home/dp/Downloads/df_income_spend_rs_d.csv')
	df_income_spend_rs_m.to_csv('/home/dp/Downloads/df_income_spend_rs_m.csv')

	return

def net_savings_estimator(start_date, end_date, current_age, retirement_age, goal_net_savings, m, P_init, r, n, t):
	df = pd.read_csv('/home/dp/Downloads/df_income_spend_rs_d.csv')
	df_m = pd.read_csv('/home/dp/Downloads/df_income_spend_rs_m.csv')

	df.Date = pd.to_datetime(df.Date)
	df_m.Date = pd.to_datetime(df_m.Date)

	print('df_m:\n', df_m.to_string())

	df.set_index('Date', inplace=True)
	df_fit = df.loc[start_date:end_date, :]
	df.reset_index(inplace=True)
	df_fit.reset_index(inplace=True)

	print('np.timedelta64(1,"D"):\n', np.timedelta64(1,'D'))
	df_fit['DateDelta'] = (df_fit.Date - df_fit.Date.min()) / np.timedelta64(1,'D')
	print('df_fit:\n', df_fit)

	fit_start_day = 0
	fit_start_date = df_fit.Date.iloc[0]

	yrs_to_go = retirement_age - current_age
	days_to_go = yrs_to_go*365
	months_to_go = yrs_to_go*12
	days_to_go_td = pd.to_timedelta(days_to_go, unit='d')
	print('days to go timedelta:\n', days_to_go_td)
	current_date = df_fit.Date.iloc[-1]
	print('current_date:\n', current_date)
	retirement_date = current_date + days_to_go_td
	print('retirement date:\n', retirement_date)

	# Calculating linear fit to current savings rate:
	current_day = df_fit.DateDelta.iloc[-1]
	retirement_day = current_day + days_to_go

	lin = smf.ols(formula='Balance ~ DateDelta', data=df_fit).fit()
	x = pd.DataFrame({'DateDelta':[fit_start_day, retirement_day]})
	linpred = pd.DataFrame({'Fit':lin.predict(x)})
	linpred['Date'] = [fit_start_date, retirement_date]
	print('linpred:\n', linpred)

	retirement_savings = int(linpred.Fit.iloc[-1])
	print('\nRetiring at age {} on {}, net savings will be ${}'.format(retirement_age, retirement_date, retirement_savings))

	# Calculating compound interest on $1000/month into the bank:
	# avg_savings_between_paychecks = 
	print('df_m:\n', df_m)
	df_m.set_index('Date', inplace=True)
	median_monthly_savings = df_m.Saved.median()
	df_m.plot(use_index=True)
	plt.show()

	# Plotting linear fit followed by compound interest:
	fig, (ax1) = plt.subplots(1,1, figsize=(8,6))
	df.plot(x='Date', y='Balance', ax=ax1, color='k', alpha=0.5)
	linpred.plot(x='Date', y='Fit', ax=ax1, color='g', alpha=0.5)

	# Iterative: Compound interest calculation:
	color_list = ['g','b','r','k','purple','orange','violet']
	savings_list = list()
	for monthly_contribution, color in zip(range(1000,4001,1000), color_list): # Various savings rates
		balance_list = list()
		P = P_init # Initialize P
		savings_list = [monthly_contribution]*months_to_go
		for t_step in range(months_to_go-1): # Iterate through each month until retirement (0 to 240)
			# t_step is just creating a loop to sum balance and monthly contribution
			# to update the principal P. t is a constant and is 1/12, accounting
			# for compounding every month.
			balance = P*(1 + r/n)**(n*t) + ( ((1 + r/n)**(n*t) - 1) / (r/n) )
			P = balance+monthly_contribution # Update the principal with monthly contribution
			balance_list.append(balance)

		balance_list = [P_init] + balance_list
		print('Balance list:\n', balance_list)

		# Creating a dataframe of the balance and covering the time period from the linear fit start date to 240 months out
		df_compound = pd.DataFrame({'Date':pd.date_range(fit_start_date, periods=240, freq='M'), 'Balance':balance_list, 'Savings':savings_list})
		# Get the cumulative sum of the savings to plot against the compounded interest savings in the 'Balance' column
		df_compound.Savings = df_compound.Savings.cumsum()

		# Plot compound interest
		compound_label_str = '$'+str(monthly_contribution)+'/month invested'
		savings_label_str = '$'+str(monthly_contribution)+'/month saved'
		df_compound.plot(x='Date', y='Balance', ax=ax1, color=color, alpha=0.5, label=compound_label_str)
		df_compound.plot(x='Date', y='Savings', ax=ax1, color=color, alpha=0.2, label=savings_label_str)

	plt.show()

	# Matrix: Compound interest calculation:
	k1 = (1 + r/n)**(n*t)
	k2 = ( ((1 + r/n)**(n*t) - 1) / (r/n) )
	print('k1:\n', k1)
	print('k2:\n', k2)

	# Reshaping a 2x2 array. k1 and k2 are constants, to deal with this,
	# broadcast_based_reshape is modified to cover a range of orders
	# and the exponentiation is done differently

	# arr = np.array([[1,2],[3,4]])
	# print('arr:\n', arr)
	# def broadcast_based_reshape(arr, order):
	#     # Create a 3D exponent array for a 2D input array to force broadcasting
	#     powers = np.arange(order + 1)[:, None]
	#     # Generate values (3-rd axis contains array at various powers)
	#     exponentiated = arr[:, None] ** powers
	#     # reshape and return array
	#     return exponentiated.reshape(arr.shape[0], -1)  # <== using reshape function
	# new_arr = broadcast_based_reshape(arr, 3)
	# print('new_arr:\n', new_arr)

	def broadcast_based_reshape(k, order_low, order):
	    # Create a 3D exponent array for a 2D input array to force broadcasting
	    powers = np.arange(order_low, order+1)[:, None]
	    # print('powers:\n', powers)
	    # Generate values (3-rd axis contains array at various powers)
	    exponentiated = k ** powers
	    # reshape and return array
	    return exponentiated  # <== using reshape function

	k1_A_arr = broadcast_based_reshape(k1, 1, months_to_go)
	k1_B_arr = broadcast_based_reshape(k1, 0, months_to_go-1)
	print('k1_A_arr:\n', k1_A_arr)
	print('k1_B_arr:\n', k1_B_arr)

	m_arr = np.array(np.ones(months_to_go)*monthly_contribution) #m)
	# print('m_arr:\n', m_arr)
	k2_arr = np.array(np.ones(months_to_go)*k2)
	# print('k2_arr:\n', k2_arr)

	A = np.dot(m_arr, k1_A_arr)
	# print('A:', A)
	B = np.dot(k2_arr,k1_B_arr)
	# print('B:', B)
	C = P_init*((1+(r/months_to_go))**(months_to_go*t)) * (k1**(months_to_go))# + P*(k1-1)
	# print('C:', C)

	balance = A + B + C
	
	print('Retirement balance - matrix solution: ${}'.format(balance[0]))
	print('Retirement balance - iterative solution: ${}'.format(balance_list[-1]))

	diff = balance_list[-1] - balance[0]
	print('Iterative - Matrix solution: ${}'.format(diff))

	return


def mc(start_date, end_date, num_years, starting_balance, monthly_income, pcnt_income_invested_avg, pcnt_income_invested_std, income_invested_dist, rate_avg, rate_std, rate_dist, n, t, num_sims):
	
	# RETURNS ASSUMING CONSTANT RATE OF RETURN:

	# Renaming:
	r_avg = rate_avg
	r_std = rate_std
	r_dist = rate_dist

	n = 1 # number of times money is compounded per month
	t = 1/12 # compounding frequency per year

	num_months = num_years*12
	months = list(range(0,num_months,1))
	df = pd.DataFrame()

	# for i in range(num_sims):
	# 	if income_invested_dist == 'normal':
	# 		pcnt_income_invested = np.random.normal(pcnt_income_invested_avg, pcnt_income_invested_std, num_months).clip(0)

	# 	income_invested = pcnt_income_invested*monthly_income
	# 	# print('income_invested:\n', income_invested)
	# 	income_invested_cumsum = income_invested.cumsum()

	# 	if r_dist == 'normal':
	# 		rate_of_return = np.random.normal(r_avg, r_std, num_months)
	# 	# print('rate_of_return:\n', rate_of_return)

	# 	returns = []
	# 	balance_list = []
	# 	P = starting_balance
	# 	for m,r in zip(months, rate_of_return): # Iterate through each month until retirement (0 to 240)
	# 		# j is just creating a loop to sum balance and monthly contribution
	# 		# to update the principal P. t is a constant and is 1/12, accounting
	# 		# for compounding every month.
	# 		balance = P*(1 + r/n)**(n*t) + ( ((1 + r/n)**(n*t) - 1) / (r/n) )
	# 		P = balance+income_invested[m] # Update the principal with monthly contribution
	# 		balance_list.append(round(balance,2))

	# 	months_new = [-1] + months
	# 	returns = [starting_balance] + balance_list

	# 	# print('months_new:\n', months_new)
	# 	# print('returns:\n', returns)
		
	# 	# print('len(months):', len(months_new))
	# 	# print('len(returns):', len(returns))

	# 	plt.plot(returns, linewidth=1)
	# 	# data_dict = {'Month':months, 'Investment':income_invested, 'CS Investment':income_invested_cumsum, 'Investment Returns':returns}
		
	# # df = pd.DataFrame(df, )
	# # print('df:\n', df)

	# plt.show()


	df = pd.read_csv('/home/dp/Documents/Equities/DJI1985.csv')
	print('df:\n', df.head().to_string())
	df.Date = pd.to_datetime(df.Date)
	df.set_index('Date', inplace=True)
	df = df.loc[start_date:, ['Close']]
	print('df:\n', df)

	df['Pct Change'] = df['Close'].pct_change()
	df.dropna(inplace=True)
	print('df["Pct Change"]:\n', df['Pct Change'])
	df_pct_change = df['Pct Change']

	# plt.plot(df_pct_change)
	# plt.xlabel('Date')
	# plt.ylabel('Daily Percent Change')
	# plt.show()

	num_days = df_pct_change.shape[0]
	print('Number of days since 1985-01-30:', num_days)
	day = list(range(num_days))
	date = list(df.index.values)
	rate_of_return_0 = df_pct_change.values
	rate_of_return_1 = df_pct_change.shift(1, fill_value=0).values
	rate_of_return_2 = df_pct_change.shift(2, fill_value=0).values
	rate_of_return_3 = df_pct_change.shift(3, fill_value=0).values
	rate_of_return_4 = df_pct_change.shift(4, fill_value=0).values
	rate_of_return_5 = df_pct_change.shift(5, fill_value=0).values
	rate_of_return_6 = df_pct_change.shift(6, fill_value=0).values

	print('Daily rate of return:\n', rate_of_return_0)
	print('Daily rate of return shifted one day:\n', rate_of_return_1)

	buy_pct_change_rule_cutoff = -1.0 # -1 means the dji must lose at least 1% of its value in a day before income is invested according to the income investment rule
	buy_pct_change_rule_accel  = [-0.1, -0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]

	# income_invested_rule_random = np.random.normal(pcnt_income_invested_avg, pcnt_income_invested_std, num_days).clip(0)
	# income_invested_rule_fixed = 0.5
	# income_invested_rule_lin   = [0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0]
	# income_invested_rule_accel = [0.1,  0.2,  0.32,  0.46,  0.58,  0.76,  0.96,  1.0,  1.0,  1.0]
	# income_invested_rule_decel = [0.1,  0.3,  0.48,  0.64,  0.78,  0.9,  1.0,  1.0,  1.0,  1.0]

	dji_r_avg = df_pct_change.mean()
	dji_r_std = df_pct_change.std()
	print('Dow Jones rate of return avg = {} and std = {} since 1985:'.format(r_avg, r_std))
	
	balance_list = []
	P = starting_balance
	daily_income = monthly_income / 30 / 2 # Dividing by 2 to cut in half, then investing 100% in all strategies
	print('monthly_income:', monthly_income)
	print('daily_income:', daily_income)

	for d, r in zip(day, rate_of_return_0): # Iterate through each day until present day of the Dow Jones
		balance = P + P*r
		P = balance # Update the principal with daily contribution
		balance_list.append(balance)
	df['$6000 Growth'] = balance_list
	print('$6000 Growth:\n', df['$6000 Growth'].head())

	# Investing 50% of daily income on all days
	balance_list = []
	bankroll_list = []
	invested_list = []
	P = starting_balance
	cash_on_hand = 1000
	pcnt_income_invested = 1
	for d, r in zip(day, rate_of_return_0):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		
		balance = P + P*r + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
		P = balance
		cash_on_hand = (1-pcnt_income_invested) * (daily_income + cash_on_hand) # Update cash_on_hand by calculating the income and cash on hand that's not invested
		invested = amount_income_invested

		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
	df['DCA100 Balance'] = balance_list
	df['DCA100 Bankroll'] = bankroll_list
	df['DCA100 Invested'] = invested_list

	return df
	'''
	# Investing 50% of daily income when the rate of return for the day is < 0. Starting with $1000 cash on hand.
	balance_list = []
	bankroll_list = []
	invested_list = []
	P = starting_balance
	cash_on_hand = 0
	pcnt_income_invested = 1
	for d, r in zip(day, rate_of_return_0):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		if r < 0:
			balance = P + P*r + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			P = balance
			cash_on_hand = (1-pcnt_income_invested) * (daily_income + cash_on_hand) # Update cash_on_hand by calculating the income and cash on hand that's not invested
		else:
			balance = P + P*r
			P = balance
			cash_on_hand += pcnt_income_invested * daily_income # Add the daily income if it's not being invested
		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
	df['DCA100 Balance Down'] = balance_list
	df['DCA100 Bankroll Down'] = bankroll_list
	df['DCA100 Invested Down'] = invested_list

	# Investing 50% of daily income when the rate of return for the day is > 0. Starting with $1000 cash on hand.
	balance_list = []
	bankroll_list = []
	invested_list = []
	P = starting_balance
	cash_on_hand = 0
	pcnt_income_invested = 1
	for d, r in zip(day, rate_of_return_0):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		if r > 0:
			balance = P + P*r + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			P = balance
			cash_on_hand = (1-pcnt_income_invested) * (daily_income + cash_on_hand) # Update cash_on_hand by calculating the income and cash on hand that's not invested
			invested = amount_income_invested
		else:
			balance = P + P*r
			P = balance
			cash_on_hand += pcnt_income_invested * daily_income # Add the daily income if it's not being invested
			invested = 0
		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
	df['DCA100 Balance Up'] = balance_list
	df['DCA100 Bankroll Up'] = bankroll_list
	df['DCA100 Invested Up'] = invested_list

	# print('adjusted close:\n', df['Pct Change'].values[0:10])
	# print('balance list:\n', balance_list[0:10])
	# print('bankroll list:\n', bankroll_list[0:10])

	# Investing all cash on hand when the rate of return for the day is < -0.4 (a strong drop).
	balance_list = []
	bankroll_list = []
	invested_list = []
	P = starting_balance
	cash_on_hand = 0
	pcnt_income_invested = 1
	for d, r in zip(day, rate_of_return_0):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		if r < -0.04:
			print('<====== 4% Decline ======>')
			balance = P + P*r + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			P = balance
			invested = amount_income_invested
			cash_on_hand = (1-pcnt_income_invested) * (daily_income + cash_on_hand) # Update cash_on_hand by calculating the income and cash on hand that's not invested
		else:
			balance = P + P*r
			P = balance
			invested = 0
			cash_on_hand += pcnt_income_invested * daily_income # Add the daily income due to it not being invested
			# print('cash_on_hand:', cash_on_hand)
		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
	df['BearBuy Balance Down'] = balance_list
	df['BearBuy Bankroll Down'] = bankroll_list
	df['BearBuy Invested Down'] = invested_list
	df['BearBuy Invested Down CS'] = df['BearBuy Invested Down'].cumsum()


	# Investing all cash on hand when the rate of return for the day is < -0.04 (a strong drop) and sell 10% of
	# the principle after the day's return is > 0.04 (a strong gain). Starting with $0 cash on hand.
	# MR is mean reversion trading
	balance_list = []
	bankroll_list = []
	invested_list = []
	P = starting_balance
	cash_on_hand = 0 #1000
	pcnt_income_invested = 1
	pcnt_principle_sold = 0.2 # 0.1 # 0.1 wins on some occasions
	for d, r_0 in zip(day, rate_of_return_0):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		if r_0 < -0.04:
			balance = P + P*r_0 + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			print('<====== 4% Decline over one day ======>: Buying ${}'.format(amount_income_invested))
			P = balance
			invested = amount_income_invested
			cash_on_hand = 0 # Update cash_on_hand by calculating the income and cash on hand that's not invested
		elif r_0 > 0.05:
			principle_sold = pcnt_principle_sold*P
			print('<====== 4% Increase over one day ======>: Selling ${}'.format(principle_sold))
			balance = P + P*r_0 - principle_sold # Selling part of the principle
			P = balance
			invested = -principle_sold
			cash_on_hand += pcnt_income_invested * daily_income + principle_sold # Add the daily income due to it not being invested, add the sold principle as well
			#print('cash_on_hand:', cash_on_hand)
		else:
			invested = (pcnt_income_invested * daily_income)*1 		# Part of income invested and...
			cash_on_hand += (pcnt_income_invested * daily_income)*0  # ...part is saved to cash on hand.
			# print('<====== Normal Day ======>: Buying ${}'.format(invested))
			balance = P + P*r_0 + invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			P = balance
		# print('balance:', balance)
		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
	df['MR Balance'] = balance_list
	df['MR Bankroll'] = bankroll_list
	df['MR Invested'] = invested_list
	df['MR Invested CS'] = df['MR Invested'].cumsum()


	# Investing 50% of cash on hand when the rate of return over two days is < 8% (a strong drop) and sell 20%
	# of the principle after the last two days return > 10% (a strong gain). Starting with $0 cash on hand.
	# MR is mean reversion trading
	balance_list = []
	bankroll_list = []
	invested_list = []
	P = starting_balance
	cash_on_hand = 0 #1000
	pcnt_income_invested = 1
	pcnt_principle_sold = 0.2 # 0.1 beats the other strategies on some occasions
	for d, r_0, r_1 in zip(day, rate_of_return_0, rate_of_return_1):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		if r_0 + r_1 < -0.08:
			balance = P + P*r_1 + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			print('<====== 8% Decline over two days ======>: Buying ${}'.format(amount_income_invested))
			P = balance
			invested = amount_income_invested
			cash_on_hand = 0 # All money on hand is invested, update to $0
		elif r_0 + r_1 > 0.08:
			principle_sold = pcnt_principle_sold*P
			print('<====== 8% Increase over two days ======>: Selling ${}'.format(principle_sold))
			balance = P + P*r_1 - principle_sold # Selling part of the principle
			P = balance
			invested = -principle_sold
			cash_on_hand += pcnt_income_invested * daily_income + principle_sold # Add the daily income due to it not being invested, add the sold principle as well
			#print('cash_on_hand:', cash_on_hand)
		else:
			invested = (pcnt_income_invested * daily_income)*0.9 		# Part of income invested and...
			cash_on_hand += (pcnt_income_invested * daily_income)*0.1  # ...part is saved to cash on hand.
			# print('<====== Normal Day ======>: Buying ${}'.format(invested))
			balance = P + P*r_1 + invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			P = balance
		# print('balance:', balance)
		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
	df['MR2 Balance'] = balance_list
	df['MR2 Bankroll'] = bankroll_list
	df['MR2 Invested'] = invested_list
	df['MR2 Invested CS'] = df['MR2 Invested'].cumsum()


	# Investing 50% of cash on hand when the rate of return over two days is < 12% (a strong drop) and sell 20%
	# of the principle after the last two days return > 12% (a strong gain). Starting with $0 cash on hand.
	# MR is mean reversion trading
	balance_list = []
	bankroll_list = []
	invested_list = []
	P = starting_balance
	cash_on_hand = 0 #1000
	pcnt_income_invested = 1
	pcnt_principle_sold = 0.2 # 0.1 beats the other strategies on some occasions
	for d, r_0, r_1, r_2 in zip(day, rate_of_return_0, rate_of_return_1, rate_of_return_2):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		if r_0 + r_1 + r_2 < -0.1:# A 12% decline in three days is too strict
			balance = P + P*r_2 + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			print('<====== 10% Decline over three days ======>: Buying ${}'.format(amount_income_invested))
			P = balance
			invested = amount_income_invested
			cash_on_hand = 0 # All money on hand is invested, update to $0
		elif r_0 + r_1 + r_2 > 0.115:# A 12% increase in three days is too strict, no sell-off will occur
			principle_sold = pcnt_principle_sold*P
			print('<====== 10% Increase over three days ======>: Selling ${}'.format(principle_sold))
			balance = P + P*r_2 - principle_sold # Selling part of the principle
			P = balance
			invested = -principle_sold
			cash_on_hand += pcnt_income_invested * daily_income + principle_sold # Add the daily income due to it not being invested, add the sold principle as well
			#print('cash_on_hand:', cash_on_hand)
		else:
			invested = (pcnt_income_invested * daily_income)*0.9 		# Part of income invested and...
			cash_on_hand += (pcnt_income_invested * daily_income)*0.1  # ...part is saved to cash on hand.
			# print('<====== Normal Day ======>: Buying ${}'.format(invested))
			balance = P + P*r_2 + invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			P = balance
		# print('balance:', balance)
		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
	df['MR3 Balance'] = balance_list
	df['MR3 Bankroll'] = bankroll_list
	df['MR3 Invested'] = invested_list
	df['MR3 Invested CS'] = df['MR3 Invested'].cumsum()


	# Investing 50% of cash on hand when the rate of return over two days is < 12% (a strong drop) and sell 20%
	# of the principle after the last two days return > 12% (a strong gain). Starting with $0 cash on hand.
	# MR is mean reversion trading
	balance_list = []
	bankroll_list = []
	invested_list = []
	rr_list = []
	P = starting_balance
	cash_on_hand = 0 #1000
	pcnt_income_invested = 1
	pcnt_principle_sold = 0.4 # 0.1 beats the other strategies on some occasions
	print('MR6 Strategy:\n')
	for d, dt, adj_close, r_0, r_1, r_2, r_3, r_4, r_5 in zip(day, date, df['Close'].values, rate_of_return_0, rate_of_return_1, rate_of_return_2, rate_of_return_3, rate_of_return_4, rate_of_return_5):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		rr = r_0 + r_1 + r_2 + r_3 + r_4 + r_5
		if rr < -0.13:
			balance = P + P*r_5 + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			print('<====== 13% Decline over six days ======>: Buying ${} on {} at {}'.format(amount_income_invested, dt, adj_close))
			P = balance
			invested = amount_income_invested
			cash_on_hand = 0 # All money on hand is invested, update to $0
		elif rr > 0.15:
			principle_sold = pcnt_principle_sold*P
			print('<====== 15% Increase over six days ======>: Selling ${} on {} at {}'.format(principle_sold, dt, adj_close))
			balance = P + P*r_5 - principle_sold # Selling part of the principle
			P = balance
			invested = -principle_sold
			cash_on_hand += pcnt_income_invested * daily_income + principle_sold # Add the daily income due to it not being invested, add the sold principle as well
			#print('cash_on_hand:', cash_on_hand)
		else:
			invested = (pcnt_income_invested * daily_income)*1 		# Part of income invested and...
			cash_on_hand += (pcnt_income_invested * daily_income)*0  # ...part is saved to cash on hand.
			# print('<====== Normal Day ======>: Buying ${}'.format(invested))
			balance = P + P*r_5 + invested # Note the rate of return is applied to the previous balance, then the day's investment is added
			P = balance
		# print('balance:', balance)
		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
		rr_list.append(rr)
	df['MR6 Balance'] = balance_list
	df['MR6 Bankroll'] = bankroll_list
	df['MR6 Invested'] = invested_list
	df['MR6 Invested CS'] = df['MR6 Invested'].cumsum()
	df['MR6 RR'] = rr_list



	# Total invested amount:
	ser_invested = df.loc[:, ['DCA100 Invested', 'DCA100 Invested Up', 'DCA100 Invested Down', 'MR Invested', 'MR2 Invested', 'MR3 Invested', 'MR6 Invested']]
	ser_invested = ser_invested.sum()
	print('Total invested =====>:\n', ser_invested)

	last_date = df.index[-1]
	# Final balance = the last date's balance (e.g. your portfolio at retirement)
	ser_final_balance = df.loc[last_date, ['DCA100 Balance', 'DCA100 Balance Up', 'DCA100 Balance Down', 'MR Balance', 'MR2 Balance', 'MR3 Balance', 'MR6 Balance']]
	print('Final balance =====>:\n', ser_final_balance)
	# Final bankroll = the last date's bankroll (e.g. how much cash is in your pocket at retirement - doesn't include portfolio)
	ser_final_bankroll = df.loc[last_date, ['DCA100 Bankroll', 'DCA100 Bankroll Up', 'DCA100 Bankroll Down', 'MR Bankroll', 'MR2 Bankroll', 'MR3 Bankroll', 'MR6 Bankroll']]
	print('Final bankroll =====>:\n', ser_final_bankroll)
	# Final revenue = all the money you have at retirement (e.g. portfolio + cash bankroll)
	final_revenue = ser_final_balance.values + ser_final_bankroll.values
	ser_final_revenue = pd.Series(data=final_revenue, index=['DCA100 Rev', 'DCA100 Rev Up', 'DCA100 Rev Down', 'MR Rev', 'MR2 Rev', 'MR3 Rev', 'MR6 Rev'])
	print('Final revenue =====>\n', ser_final_revenue)

	# Gained = Final balance - total amount invested over the years
	ser_gained = ser_final_balance.subtract(ser_invested.values)
	# print('ser_gained:\n', ser_gained)

	# GIR is gain-to-invest ratio
	# GIR = (balance - invested) / invested. Gives a sense for how much growth you got out of a particular investment strategy.
	# GIR doesn't account for how much money is left over in your bankroll.
	# It's bad to have lots of money left over because it could have been invested.
	gain_invest_ratio = ser_gained.values / ser_invested.values
	ser_gain_invest_ratio = pd.Series(data=gain_invest_ratio, index=['DCA100 GIR', 'DCA100 GIR Up', 'DCA100 GIR Down', 'MR GIR', 'MR2 GIR', 'MR3 GIR', 'MR6 GIR'])
	print('Gain-to-invest ratio =====>\n', ser_gain_invest_ratio)


	fig, ax = plt.subplots(1,1, figsize=(10,10))
	df.plot(y='$6000 Growth', use_index=True, linewidth=1, ax=ax)
	ax.set_ylabel('Portfolio Value, $')
	plt.show()
	

	df.plot(y=['DCA100 Balance', 'DCA100 Balance Down', 'DCA100 Balance Up', 'MR Balance', 'MR2 Balance', 'MR3 Balance', 'MR6 Balance'], use_index=True, linewidth=1, marker='+', markersize=2)
	df.plot(y=['DCA100 Bankroll', 'DCA100 Bankroll Down', 'DCA100 Bankroll Up', 'MR Bankroll', 'MR2 Bankroll', 'MR3 Bankroll', 'MR6 Bankroll'], use_index=True, linewidth=1)
	#ser_invested.plot.bar(use_index=True, title='Amount Invested')
	#ser_gained.plot.bar(use_index=True, title='Amount Gained')
	ser_gain_invest_ratio.plot.bar(use_index=True, title='Gain-to-Invest Ratio')
	ser_final_revenue.plot.bar(use_index=True, title='Final Revenue')
	df.plot(y=['Close', 'MR Invested CS', 'MR2 Invested CS', 'MR Bankroll', 'MR2 Bankroll', 'MR3 Bankroll', 'MR6 Bankroll'], use_index=True, linewidth=1, title='MR vs MR2')
	df.plot(y=['MR6 RR'], use_index=True, linewidth=0.5)
	# df.plot(y='Pct Change', use_index=True)
	plt.show()
	'''



	# Notes:
	'''
	50 pcnt investment every single day vastly outperforms 50 pcnt investment after either up or down days. The reason
	for this is the total invested dollars. 50pcnt investment invests 50pcnt of income and cash on hand every day, which
	rapidly drives the cash on hand to zero, and then every day half of the daily income is invested. This is
	contrary to the up and down investment strategies which often leave $50-600 cash on hand at any time. The result
	is that, after many decades, the full time investment strategy invests $498533, while the up and down strategies
	invest $381889 and $365770. Thus, counterintuitively, investing only after days that gained results in a higher
	return than investing only after days that lost.

	MR and MR2 are both mean reversion strategies that attempt to buy low and sell high. MR uses single day declines and
	jumps of ~4% to sell ~20% of the portfolio and buy in with all of the cash on hand and that day's income. MR2 requires
	a total of two days worth of at least 4% declines or jumps to buy and sell, respectively. MR2's gain-to-invest ratio is
	3.2, and beats all other strategies. MR does the worst at 2.6, and the other strategies are around 2.84 to 2.88.

	For both MR and MR2, on days when the markets aren't making wild swings that invoke
	buys and sells, 90% of the day's income is invested, 10% is stored as cash on hand. A large amount is invested because
	the weakness of any market timing strategy such as mean reversion trading is in missing out on future gains because
	money was saved now to invest in market down turns later. Here, it's not enough to buy into a market downturn. You
	must have a market downturn that is severe enough to make up for lost gains due to saving money. So there's a clear
	opportunity cost to not investing today. There is probably an adage that says, "the best time to invest was yesterday."
	The only time this wouldn't be true is if (1) there were a way to know future stock market returns exactly, or (two) to have a
	generic strategy to follow that increases the probability of profiting on the market swings that have occurred historically,
	assuming that swings of a similar nature will occur in the future. We have no choice but to follow the second strategy.
	We also must assume that the stock market will increase over time. Massive exogenous shocks such as global warming, a
	meteor strike, a global plague, etc. could make this assumption false, but at that point we'll have more pressing concerns.

	One of the problems with mean reversion is selling a substantial slice of your portfolio on an upturn, then not buying back
	in any time soon, and thereby missing out on that cash gaining over a long growth period. The ideal mean-reversion trading
	strategy would give you money to work with by starting with a sell of say 10-20% of portfolio, then buying into the market
	soon afterwards to avoid having money sitting around for long periods of time. When would you do this? After a large jump
	in value (sell), then after the subsequent fall (buy). This is a sell-high buy-low strategy, and it rests
	on the assumption that a statistically rare jump will be followed by an equally rapid and rare fall. How true is this assumption?
	I don't know yet. One human factor worth looking at is irrational exuberance, and its opposite, irrational fatalism. The
	dot-com bubble of '01 and the real-estate bubble of '08 are examples of both. Contesting people on the profitibility of websites
	and real-estate ventures just before their respective bubbles was met with religiously fervent disagreement. No rational
	argument would be heard, no statistical figure would be heeded. Even if we knew how overly levered Lehman Brothers, Bear Stearns,
	Washington Mutual, and other investment banks were, people would have likely rationalized them as irrelevant.

	The reverse strategy is a buy-low sell-high. This is a more attractive strategy because it doesn't require identifying a peak.
	So it lets you wait for a downturn and just starting buying into it. Buying as you go down is a nice thought but it's
	problematic because it requires money to work with from the get-go, and if you've been smart, you've been investing
	regularly no matter what the market does. To raise the funds necessary for this strategy in a way that lets you make up 
	for the lost opportunity cost of not having that money in the market, a small percentage of money should be set aside on normal
	trading days. I'm experimenting with different percentages of daily income to save. The more frequenct and severe market
	downturns there are (e.g. the more volatile the market), then the more effective BLSH will be. If market downturns are few
	and far between, saving money to invest on a rainy day is riskier, and less money should be saved.

	Next experiments:
	1) MR3, MR4, MR5, MR6, etc looking at mean reversion strategies with differing lag-times. To mitigate the opportunity cost of cash
	on hand, I need to put in place a rule that requires the first move is a sell and then limits the amount of time I can go before
	buying again.
	
	2) How well does a sell-high buy-low (SHBL) strategy work? How frequently do big falls follow big jumps?
	
	3) Evolutionary algorithm for mean-reversion trading that will find optimal lag-times and percentage income to save on a daily basis


	NOTE: A mean-reversion trading strategy whose buy and sell rules require more volatility than the market every provides, and whose
	 	  daily investment is 50% of daily income, will yield the same outcome as a DCA strategy that invests 50% of daily income. This
	 	  is a useful check on the math used to calculate the balance in the MR strategies.


	'''

	return


def dca(starting_balance, monthly_income):

	df = pd.read_csv('/home/dp/Documents/Equities/DJI1985.csv')
	print('df:\n', df.head().to_string())
	df.Date = pd.to_datetime(df.Date)
	df.set_index('Date', inplace=True)
	# df = df.loc[start_date:, ['Close']]
	# print('df:\n', df)

	df['Pct Change'] = df['Close'].pct_change()
	df.dropna(inplace=True)
	print('df["Pct Change"]:\n', df['Pct Change'])
	df_pct_change = df['Pct Change']

	num_days = df_pct_change.shape[0]
	print('Number of days since 1985-01-30:', num_days)
	day = list(range(num_days))
	date = list(df.index.values)
	rate_of_return = df_pct_change.values

	daily_income = monthly_income / 30 / 2 # Dividing by 2 to cut in half, then investing 100%
	# Investing 50% of daily income on all days
	balance_list = []
	bankroll_list = []
	invested_list = []
	P = starting_balance
	cash_on_hand = 1000
	pcnt_income_invested = 1 # Investing 100% of half of daily income
	for d, r in zip(day, rate_of_return):
		amount_income_invested = pcnt_income_invested * (daily_income + cash_on_hand)
		
		balance = P + P*r + amount_income_invested # Note the rate of return is applied to the previous balance, then the day's investment is added
		P = balance
		cash_on_hand = (1-pcnt_income_invested) * (daily_income + cash_on_hand) # Update cash_on_hand by calculating the income and cash on hand that's not invested
		invested = amount_income_invested

		balance_list.append(balance)
		bankroll_list.append(cash_on_hand)
		invested_list.append(invested)
	df['DCA100 Balance'] = balance_list
	df['DCA100 Bankroll'] = bankroll_list
	df['DCA100 Invested'] = invested_list

	df['DCA100 Balance'].plot(use_index=True)
	plt.ylabel('Portfolio Value, $')
	plt.show()
	return df.loc[:, ['Close', 'DCA100 Balance']]



# sell_up() invokes sales when the market has gone up and buys when it has gone down.
# Sell up: sell after the market has risen at a particular rate
# Buy down: buy after the market has declined at a particular rate
def sell_up(stock_index, year, strategy, total_sims_to_run):

	dji_file = stock_index + str(year) + '.csv'
	df_dji = pd.read_csv(dji_file, header=0, names=['Date','Close'])
	df_dji.Date = pd.to_datetime(df_dji.Date)
	df_dji.set_index('Date', inplace=True)
	# df_dji = df_dji.loc['1950-01-01':, :]
	print('df_dji:\n', df_dji)
	print('df_dji:\n', df_dji.head().to_string())

	start_date = df_dji.index[0]
	print('df_dji:\n', df_dji)

	# Evolutionary algroithm to maximize both the return on invested dollars and final portfolio balance
	# Variables: buy decline percentage, buy wait time, buy amount, increase percentage, sell wait time, sell amount
	# No saving of money (maximum is invested each month)
	# Weights for each
	# Equation to optimize: ??
	# Variables:


	buy_wait_days = np.random.randint(1, 30, size=1)[0]
	buy_wait_days = 3
	buy_pct = np.random.uniform(0.01,0.08) * -1
	buy_pct = 0.05 * -1
	buy_portfolio_pct = np.random.randint(1, 50, size=1)[0]
	buy_portfolio_pct = 0.1


	print('Buy wait time: {} days'.format(buy_wait_days))
	# decline_before_sell = np.random.random('uniform', buy_wait_days*-0.01, buy_wait_days*-0.05)
	# rate_of_return = df_pct_change.shift(buy_wait_days, fill_value=0).values
	today = df_dji['Close']
	future = df_dji['Close'].shift(-buy_wait_days)
	period_diff = future.sub(today).dropna()
	print('period_diff:\n', period_diff)
	running_period_sum = period_diff.cumsum().dropna()
	print('running_period_sum:\n', running_period_sum)
	running_period_pct_chg = df_dji['Close'].pct_change(buy_wait_days)
	print('running_period_pct_chg:\n', running_period_pct_chg)

	# running_period_buy_bool = running_period_pct_chg.where( running_period_pct_chg < buy_pct, 0)
	running_period_buy_bool = running_period_pct_chg <= buy_pct
	print('running_period_buy_bool:\n', running_period_buy_bool)
	running_period_buy = running_period_buy_bool * buy_portfolio_pct

	# cumulative_pct_change = df_dji / df_dji.loc[start_date]
	# # Converting to a Series:
	# cumulative_pct_change = cumulative_pct_change.squeeze()
	# print('cumulative_pct_change:\n', cumulative_pct_change)
	# # cumulative_pct_change.plot(use_index=True)
	# # plt.show()

	# investments = pd.Series(data=[6000]*len(cumulative_pct_change), index=cumulative_pct_change.index)
	# print('investments:\n', investments)
	# balance = cumulative_pct_change.multiply(investments)
	# print('balance:\n', balance)
	# # balance.plot(use_index=True)
	# # plt.show()


	cumulative_pct_change = df_dji / df_dji.loc[start_date]
	cumulative_pct_change = cumulative_pct_change.squeeze()# Converting to a Series. Also works: cumulative_pct_change['Close']
	# print('cumulative_pct_change:\n', cumulative_pct_change)
	# cumulative_pct_change.plot(use_index=True)
	# plt.show()

	investments = pd.Series(data=[6000]*len(cumulative_pct_change), index=cumulative_pct_change.index)
	# print('investments:\n', investments)
	original_balance = cumulative_pct_change.multiply(investments)
	original_return = original_balance.iloc[-1]

	# These dates need to exist. If a date is on the weekend it won't exist and it will add it to the index's end, causing problems.
	sell_dates = ['2000-01-18', '2007-10-01']
	buy_dates = ['2002-10-10', '2009-03-02']
	sell_amount = 0# 40000
	buy_amount = 0# 40000

	# buy_sell = pd.Series(data=[0]*len(cumulative_pct_change), index=cumulative_pct_change.index)
	# buy_sell_cumsum = buy_sell.cumsum()
	# # print('buy_sell_cumsum:\n', buy_sell_cumsum)
	# # buy_sell_cumsum.plot(use_index=True)
	# # plt.show()

	# investments_with_transactions = investments + buy_sell_cumsum
	# balance_with_transactions = cumulative_pct_change.multiply(investments_with_transactions)
	
	# # balance.plot(use_index=True)
	# # balance_with_transactions.plot(use_index=True)
	# # plt.show()

	if strategy == 'sell_up':
		max_sell_wait_days = 10000
		max_buy_wait_days = 6000
		m = 1 # Multiplier to control selling and buying on the increase or decrease
	elif strategy == 'sell_down':
		max_sell_wait_days = 6000
		max_buy_wait_days = 10000
		m = -1

	traj_list = []
	parameters_list = []
	sell_dates_list = []
	buy_dates_list = []
	print('Running {} simulations...'.format(total_sims_to_run))
	for n in range(total_sims_to_run):
		pct_complete = n/total_sims_to_run*100
		print('Percentage complete: {:.1f}%'.format(pct_complete))
	
		# Parameters:
		sell_wait_days = np.random.randint(1, max_sell_wait_days, size=1)[0]
		sell_pct = np.random.uniform(0.01,0.9) * m
		sell_portfolio_pct = np.random.uniform(0.01,1.0)
		buy_wait_days = np.random.randint(1, max_buy_wait_days, size=1)[0]
		buy_pct = np.random.uniform(0.01,0.9) * -1 * m
		# print('buy_wait_days:', buy_wait_days)
		# print('buy_pct:', buy_pct)
		# print('sell_wait_days:', sell_wait_days)
		# print('sell_pct:', sell_pct)
		# print('sell_portfolio_pct:', sell_portfolio_pct)

		# buy_portfolio_pct doesn't exist because I'm setting the buy_amount = sell_amount below
		# decline_before_sell = np.random.random('uniform', buy_wait_days*-0.01, buy_wait_days*-0.05)
		# rate_of_return = df_pct_change.shift(buy_wait_days, fill_value=0).values

		# Use sell_wait_days and buy_wait_days to shift the data and calculate the
		# percent change over the shifted period.
		today = df_dji['Close']
		
		future = df_dji['Close'].shift(-sell_wait_days)
		# buy_period_diff = future.sub(today).dropna()
		# print('buy_period_diff:\n', buy_period_diff)
		# buy_period_cumsum = buy_period_diff.cumsum().dropna()
		# print('buy_period_cumsum:\n', buy_period_cumsum)
		sell_period_pct_chg = df_dji['Close'].pct_change(sell_wait_days).rename('Sell % Change', inplace=True)
		# print('sell_period_pct_chg:\n', sell_period_pct_chg)
		if strategy == 'sell_up':
			sell_dates = sell_period_pct_chg.where( sell_period_pct_chg >= sell_pct ).dropna().index.tolist()
			# print('sell dates:\n', sell_dates)
		else:
			sell_dates = sell_period_pct_chg.where( sell_period_pct_chg <= sell_pct ).dropna().index.tolist()

		future = df_dji['Close'].shift(-buy_wait_days)
		# buy_period_diff = future.sub(today).dropna()
		# print('buy_period_diff:\n', buy_period_diff)
		# buy_period_cumsum = buy_period_diff.cumsum().dropna()
		# print('buy_period_cumsum:\n', buy_period_cumsum)
		buy_period_pct_chg = df_dji['Close'].pct_change(buy_wait_days).rename('Buy % Change', inplace=True)
		# print('buy_period_pct_chg:\n', buy_period_pct_chg)
		if strategy == 'sell_up':
			buy_dates = buy_period_pct_chg.where( buy_period_pct_chg <= buy_pct ).dropna().index.tolist()
			# print('buy dates:\n', buy_dates)
		else:
			buy_dates = buy_period_pct_chg.where( buy_period_pct_chg >= buy_pct ).dropna().index.tolist()

		# If the sell dates or buy dates don't exist because conditions weren't met to invoke a transaction,
		# then store the parameters that caused that along with a final return of NaN
		# Else, run the everything below including the for loop that calculates the trajectory.
		if len(sell_dates) == 0 or len(buy_dates) == 0:

			print('Outer if: sell_dates or buy_dates is empty')
			traj_list.append(original_balance.values)
			print('traj_list:\n', traj_list)
			parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, original_return])
			print('parameters_list:\n', parameters_list)
			# Store the final return as NaN
			# traj_list.append([np.nan]*len(traj_list[0]))
			# parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, np.nan])
			sell_dates_list.append([np.nan])
			buy_dates_list.append([np.nan])
			print('sell_dates_list length = ', len(sell_dates_list))
			print('buy_dates_list length = ', len(buy_dates_list))

		else:
			# Pick alternating sell-buy dates, remove extra dates. Take the first sell date, find the next buy date, find the next
			# sell date, find the next buy date, etc. This process will remove extra buys and sells that occur more than once in a
			# row, essentially buying when money doesn't exist or selling multiple times back to back (which might be a good strategy,
			# but this will need to be investigated with consideration for transaction costs).
			sell_dates_list_temp = []
			buy_dates_list_temp = []
			s = sell_dates[0]

			# If sell_dates is the longer list, set l to the length of buy_dates,
			# and vice versa.
			if len(sell_dates) >= len(buy_dates):
				l = len(buy_dates)
			else:
				l = len(sell_dates)

			# Finding alternating sell and buy dates such that each sell is followed by a buy, which
			# is followed by another sell-buy pair, etc.
			sell_dates_list_temp.append(s)
			for i in range(l):
				for b in buy_dates:
					if b > s:# and b != buy_dates_list_temp[-1]:
						buy_dates_list_temp.append(b)
						break
				for s in sell_dates:
					if s > b:# and s != sell_dates_list_temp[-1]:
						sell_dates_list_temp.append(s)
						break

			# Just as important as making each first transaction a sell,
			# every last transaction must be a buy.
			# The final result sometimes produces a sell_dates_list_temp with 
			# an extra sell with no accompanying buy. If the sell_dates_list_temp
			# is one element longer than buy_dates_list_temp, cut the last value
			# from the sell list:
			buy_len = len(buy_dates_list_temp)
			sell_len = len(sell_dates_list_temp)
			if sell_len > buy_len:
				sell_dates_list_temp = sell_dates_list_temp[0:buy_len]

			# After adjusting the length of sell_dates_list_temp, do one last check
			# to make sure it contains at least one date, if not skip the simulation.
			buy_len = len(buy_dates_list_temp)
			sell_len = len(sell_dates_list_temp)
			
			if sell_len == 0 or buy_len == 0:
				print('Inner if: sell_dates or buy_dates is empty')
				print('len(traj_list):\n', len(traj_list))
				traj_list.append(original_balance.values)
				print('traj_list:\n', traj_list)
				parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, original_return])
				print('parameters_list:\n', parameters_list)
				sell_dates_list.append([np.nan])
				buy_dates_list.append([np.nan])
				print('sell_dates_list length = ', len(sell_dates_list))
				print('buy_dates_list length = ', len(buy_dates_list))

			else:
				# Necessary to sort to maintain chronological order after the set operation.
				# Now that we have a master list of sell and buy dates, write them to sell_dates
				# and buy_dates. sell_dates and buy_dates originally started out with many dates,
				# some identical to each other. Then they were selected so every sell was followed
				# by a buy, then the paired down list is written over sell_dates and buy_dates below.
				sell_dates_list_temp = sorted(list(set(sell_dates_list_temp)))
				buy_dates_list_temp = sorted(list(set(buy_dates_list_temp)))
				sell_dates = sell_dates_list_temp
				buy_dates = buy_dates_list_temp
				print('sell_dates:\n', len(sell_dates))
				print('buy_dates:\n', len(buy_dates))
				print('************************************************ n = ', n)

				df_period_pct_chg = pd.concat((sell_period_pct_chg, buy_period_pct_chg), axis=1)
				# print('df_period_pct_chg:\n', df_period_pct_chg)

				# df_period_pct_chg.plot(use_index=True, title='Percent Change')
				# plt.show()

				# running_period_buy_bool = running_period_pct_chg.where( running_period_pct_chg < buy_pct, 0)
				# running_period_buy_bool = running_period_pct_chg <= buy_pct
				# print('running_period_buy_bool:\n', running_period_buy_bool)
				# running_period_buy = running_period_buy_bool * buy_portfolio_pct

				# Restore the balance for the next iteration:
				balance = original_balance

				# Go through the buy and sell dates, compute the original investment required to achieve the buys and sells balance, then
				# update the investments array to update the balance array. Do this for each sell-buy combination, then finally calculate
				# the final balance using the last computed original_investment, to get the final return on the portfolio.
				return_list = [original_return] # Initiate the list with the first final return
				final_return_list = []
				# start_date = '1985-01-29'
				# last_date = '2019-12-13'
				start_date = '1914-12-12'
				last_date = '2021-07-16'
				# print('sell_dates:\n', sell_dates)
				sell_date = sell_dates[0]
				# print('sell_date:\n', sell_date)
				adj_sell_date = sell_dates[0]-dt.timedelta(days=1)
				# print('Is adj_sell_date in balance index? ', balance.index.contains(adj_sell_date))		
				# print('sim # {}, start_date:adj_sell_date = {}:{}'.format(n, start_date, adj_sell_date))
				trajectory = pd.Series(data=balance.loc[start_date:adj_sell_date])
				# print('trajectory:\n', trajectory)
				for i in range(len(sell_dates)):
					# print('i****************************', i)
					# Separately calculating the sell information for creating a trajectory
					sell_amount = balance.loc[sell_dates[i]]*sell_portfolio_pct
					sell_investment = ((balance.loc[sell_dates[i]] - sell_amount) / cumulative_pct_change.loc[sell_dates[i]])
					sell_investments = pd.Series(data=[sell_investment]*len(cumulative_pct_change), index=cumulative_pct_change.index)
					sell_balance = cumulative_pct_change.multiply(sell_investments)
					# Update the traj with the first sell:
					trajectory = pd.concat((trajectory, sell_balance.loc[sell_dates[i]:buy_dates[i]-dt.timedelta(days=1)]), axis=0)
					# print('trajectory*:\n', trajectory)
					# For now let's assume that we use 100% of what's sold to buy back into the market. This is
					# a decent strategy because saving money for the future is almost never a good idea.
					buy_amount = sell_amount

					# Calculate the original investment required to achieve a certain balance
					original_investment = ((balance.loc[sell_dates[i]] - sell_amount) / cumulative_pct_change.loc[sell_dates[i]]) + (buy_amount / cumulative_pct_change.loc[buy_dates[i]])
					# print('original_investment:', original_investment)

					# Using original_investment, recalculate the entire balance array:
					investments = pd.Series(data=[original_investment]*len(cumulative_pct_change), index=cumulative_pct_change.index)
					# print('investments:\n', investments)
					balance = cumulative_pct_change.multiply(investments)
					# print('balance:\n', balance)
					return_list.append(balance.iloc[-1])
				# 	# Update balance with next sell_date. I don't think this is needed
				# 	balance = original_investment * cumulative_pct_change.loc[sell_dates[i+1]] # next sell date
					# If we're not on the last sell date, then create the trajectory that goes from the buy date
					# to the next sell date. If we are on the last sell date, 
					if i+1 < len(sell_dates):
						trajectory = pd.concat((trajectory, balance.loc[buy_dates[i]:sell_dates[i+1]-dt.timedelta(days=1)]), axis=0)
						# print('trajectory**:\n', trajectory)

					# Math behind the original_investment equation above:
					# new_balance = (original_investment * cumulative_pct_change.loc[buy_date]) + buy_amount
					# original_investment = new_balance / cumulative_pct_change.loc[buy_date]
					# Plugging new_balance into the original_investment equation:
					# original_investment += (buy_amount / cumulative_pct_change.loc[buy_date])
				trajectory = pd.concat((trajectory, balance.loc[buy_dates[i]:]), axis=0)
				# print('trajectory***:\n', trajectory)
				traj_list.append(trajectory.values)
			
				final_balance = balance
				print('final_balance:\n', final_balance)
				final_return = original_investment * cumulative_pct_change.iloc[-1]
				print('original_investment:\n', original_investment)
				print('cumulative_pct_change.iloc[-1]:\n', cumulative_pct_change.iloc[-1])

				print('final_return:\n', final_return)
				parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, final_return])
				print('parameters_list:\n', parameters_list)

				sell_dates_list.append(sell_dates)
				buy_dates_list.append(buy_dates)
				print('sell_dates_list length = ', len(sell_dates_list))
				print('buy_dates_list length = ', len(buy_dates_list))



				# Another way to calculate the final return: final_return = balance.iloc[-1]
				# print('final_return:', final_return)
				# print('return_list:\n', return_list)
				# print('trajectory:\n', trajectory)

			

	# print('parameters_list:\n', parameters_list)

	traj_arr = np.asarray(traj_list).T#.reshape((-1,3))
	df_traj = pd.DataFrame(data=traj_arr, index=cumulative_pct_change.index)
	# print('df_traj:\n', df_traj)
	original_balance.rename('Original Balance', inplace=True)
	df_traj = df_traj.join(original_balance, how='outer')
	# print('df_traj after joining original_balance:\n', df_traj)

	df_parameters = pd.DataFrame(data=parameters_list, columns=['Sell Wait Days', 'Sell Pct','Sell Port Pct','Buy Wait Days','Buy Pct','Final Return'])
	df_parameters.index.rename('Strategy', inplace=True)
	# print('df_parameters before sort:\n', df_parameters.to_string())
	df_parameters.sort_values(axis=0, by=['Final Return'], ascending=False, inplace=True)
	print('df_parameters after sort:\n', df_parameters.to_string())

	df_sell_dates = pd.DataFrame(data=sell_dates_list, index=list(range(len(sell_dates_list)))).T
	print('df_sell_dates:\n', df_sell_dates)

	df_buy_dates = pd.DataFrame(data=buy_dates_list, index=list(range(len(buy_dates_list)))).T
	print('df_buy_dates:\n', df_buy_dates)

	ser_final_returns = df_parameters['Final Return'].dropna()
	print('ser_final_returns:\n', ser_final_returns.head(20))
	pct_winning = ser_final_returns[ser_final_returns > original_return].count() / ser_final_returns.count() * 100
	print('Percent winning strategies: {}%'.format(pct_winning))

	# DISPLAY ALL INVESTMENT STRATEGIES:
	# df_traj.plot(use_index=True, linewidth=0.7)
	# original_balance.plot(use_index=True, linewidth=3)
	# plt.show()

	# DISPLAY WINNING INVESTMENT STRATEGIES:
	df_traj.sort_values(axis=1, by=[last_date], ascending=False, inplace=True)
	df_traj.dropna(inplace=True)
	print('df_traj sorted by highest final return:\n', df_traj.iloc[:,:30])#df_traj_sorted.loc[:,:'Original Balance'])
	df_traj.iloc[:,:30].plot(use_index=True, linewidth=1)
	original_balance.plot(use_index=True, linewidth=3)
	plt.xlabel('Date')
	plt.ylabel('Value, $')
	plt.savefig(strategy+'_'+str(total_sims_to_run)+'_sims.png', bbox_inches='tight')
	plt.show()

	# Just verifying one last time that the original_balance is somewhere in df_traj
	print('df_traj columns:\n', df_traj.columns.values.tolist()) # To verify original_balance is somewhere in here

	# Pickle out the parameters and trajectories:
	df_traj.to_pickle('trajectories_'+strategy+'_'+str(total_sims_to_run)+'_sims.pkl')
	df_parameters.to_pickle('parameters_'+strategy+'_'+str(total_sims_to_run)+'_sims.pkl')
	df_sell_dates.to_pickle('sell_dates_'+strategy+'_'+str(total_sims_to_run)+'_sims.pkl')
	df_buy_dates.to_pickle('buy_dates_'+strategy+'_'+str(total_sims_to_run)+'_sims.pkl')


	# df_all_trajs = pd.concat((original_balance, final_balance, trajectory), axis=1)
	# df_all_trajs.rename(columns={0:'Original Balance', 1:'Final Balance', 2:'Trajectory'}, inplace=True)
	# original_balance.plot(use_index=True)
	# final_balance.plot(use_index=True)
	# trajectory.plot(use_index=True, legend=True)
	# print('df_trajectories:\n', df_trajectories)
	# df_all_trajs.plot(use_index=True)
	# plt.show()




	# Difficult to plot the buy-sell balance because it's a mess finding each
	# buy and sell balance and extrapolating over the various periods between
	# buys and sells. I'm mostly just interested in the final balance for
	# comparing the success of various strategies.

	return


# sell_up() invokes sales when the market has gone up and buys when it has gone down.
# Sell up: sell after the market has risen at a particular rate
# Buy down: buy after the market has declined at a particular rate
def sell_up_2(stock_index, year):

	dji_file = stock_index + str(year) + '.csv'
	df_dji = pd.read_csv(dji_file, header=0, names=['Date','Close'])
	df_dji.Date = pd.to_datetime(df_dji.Date)
	df_dji.set_index('Date', inplace=True)
	start_date = df_dji.index[0]
	df_dji = df_dji.loc[:, ['Close']]
	print('df_dji:\n', df_dji)
	print('Data type of the index:\n', df_dji.index.dtype)

	# EA to maximize both the return on invested dollars and final portfolio balance
	# Variables: buy decline percentage, buy wait time, buy amount, increase percentage, sell wait time, sell amount
	# No saving of money (maximum is invested each month)
	# Weights for each
	# Equation to optimize: ??
	# Variables:

	buy_wait_days = np.random.randint(1, 30, size=1)[0]
	buy_wait_days = 3
	buy_pct = np.random.uniform(0.01,0.08) * -1
	buy_pct = 0.05 * -1
	buy_portfolio_pct = np.random.randint(1, 50, size=1)[0]
	buy_portfolio_pct = 0.1

	print('Buy wait time: {} days'.format(buy_wait_days))
	# decline_before_sell = np.random.random('uniform', buy_wait_days*-0.01, buy_wait_days*-0.05)
	# rate_of_return = df_pct_change.shift(buy_wait_days, fill_value=0).values
	today = df_dji['Close']
	future = df_dji['Close'].shift(-buy_wait_days)
	period_diff = future.sub(today).dropna()
	print('period_diff:\n', period_diff)
	running_period_sum = period_diff.cumsum().dropna()
	print('running_period_sum:\n', running_period_sum)
	running_period_pct_chg = df_dji['Close'].pct_change(buy_wait_days)
	print('running_period_pct_chg:\n', running_period_pct_chg)

	# running_period_buy_bool = running_period_pct_chg.where( running_period_pct_chg < buy_pct, 0)
	running_period_buy_bool = running_period_pct_chg <= buy_pct
	print('running_period_buy_bool:\n', running_period_buy_bool)
	running_period_buy = running_period_buy_bool * buy_portfolio_pct

	cumulative_pct_change = df_dji / df_dji.loc[start_date]
	# Convert to 
	cumulative_pct_change = cumulative_pct_change.squeeze()# Converting to a Series. Also works: cumulative_pct_change['Close']
	print('cumulative_pct_change:\n', cumulative_pct_change)
	# cumulative_pct_change.plot(use_index=True)
	# plt.show()

	investments = pd.Series(data=[6000]*len(cumulative_pct_change), index=cumulative_pct_change.index)
	print('investments:\n', investments)
	balance = cumulative_pct_change.multiply(investments)
	print('balance:\n', balance)
	# balance.plot(use_index=True)
	# plt.show()

	# These dates need to exist. If a date is on the weekend it won't exist and it will add it to the index's end, causing problems.
	sell_dates = ['2000-01-18', '2007-10-01']
	buy_dates = ['2002-10-10', '2009-03-02']
	sell_amount = 0# 40000
	buy_amount = 0# 40000

	# buy_sell = pd.Series(data=[0]*len(cumulative_pct_change), index=cumulative_pct_change.index)
	# buy_sell_cumsum = buy_sell.cumsum()
	# # print('buy_sell_cumsum:\n', buy_sell_cumsum)
	# # buy_sell_cumsum.plot(use_index=True)
	# # plt.show()

	# investments_with_transactions = investments + buy_sell_cumsum
	# balance_with_transactions = cumulative_pct_change.multiply(investments_with_transactions)
	
	# # balance.plot(use_index=True)
	# # balance_with_transactions.plot(use_index=True)
	# # plt.show()

	lookup_table = pd.read_pickle('DJI'+str(year)+'_pct_change_lookup_long.pkl')
	lookup_table['Pct Change'] = lookup_table['Pct Change'] / 100
	print('lookup_table:\n', lookup_table)
	i, j = lookup_table.shape
	total_sims_to_run = i**2
	print('Total simulations to run = ', total_sims_to_run)

	traj_list = []
	parameters_list = []
	sell_dates_list = []
	buy_dates_list = []
	print('Running {} simulations...'.format(total_sims_to_run))
	for n, row in enumerate(lookup_table.itertuples()):
		# print('Row:\n', row)
		pct_complete = n/total_sims_to_run*100
		print('Percentage SELL complete: {:.1f}%'.format(pct_complete))

		# Sell variables:
		sell_wait_days = row[1]
		sell_pct = row[2] # Sell on the increase. SHOULD BE POSITIVE!

		if sell_pct < 0:
			sell_pct *= -1

		# sell_wait_days = 10
		# sell_pct = 0.5
		
		sell_portfolio_pct = np.random.uniform(0.01,1.0)
		
		print('sell_wait_days:', sell_wait_days)
		print('sell_pct:', sell_pct)
		print('sell_portfolio_pct:', sell_portfolio_pct)

		# Buy variables:
		for m, row in enumerate(lookup_table.iloc[n:,:].itertuples()):
			pct_complete = m/total_sims_to_run*100
			print('Percentage BUY complete: {:.1f}%'.format(pct_complete))

			buy_wait_days = row[1]
			buy_pct = row[2] # Buy on the decline. SHOULD BE NEGATIVE

			if buy_pct > 0:
				buy_pct *= -1

			# buy_wait_days = 10
			# buy_pct = -0.5

			
			print('buy_wait_days:', buy_wait_days)
			print('buy_pct:', buy_pct)


			# # Sell variables:
			# sell_wait_days = np.random.randint(14, 500, size=1)[0]# 5
			# sell_pct = np.random.uniform(0.01,0.6) # Sell on the increase
			# sell_portfolio_pct = np.random.uniform(0.01,1.0)
			# # print('sell_wait_days:', sell_wait_days)
			# # print('sell_pct:', sell_pct)
			# # print('sell_portfolio_pct:', sell_portfolio_pct)

			# # Buy variables:
			# buy_wait_days = np.random.randint(14, 500, size=1)[0]
			# buy_pct = np.random.uniform(0.01,0.6) * -1 # Buy on the decline
			# # print('buy_wait_days:', buy_wait_days)
			# # print('buy_pct:', buy_pct)

			# buy_portfolio_pct doesn't exist because I'm setting the buy_amount = sell_amount below
			# decline_before_sell = np.random.random('uniform', buy_wait_days*-0.01, buy_wait_days*-0.05)
			# rate_of_return = df_pct_change.shift(buy_wait_days, fill_value=0).values

			# Use sell_wait_days and buy_wait_days to shift the data and calculate the
			# percent change over the shifted period.
			today = df_dji['Close']
			
			future = df_dji['Close'].shift(-sell_wait_days)
			# buy_period_diff = future.sub(today).dropna()
			# print('buy_period_diff:\n', buy_period_diff)
			# buy_period_cumsum = buy_period_diff.cumsum().dropna()
			# print('buy_period_cumsum:\n', buy_period_cumsum)
			sell_period_pct_chg = df_dji['Close'].pct_change(sell_wait_days).rename('Sell % Change', inplace=True)
			# print('sell_period_pct_chg:\n', sell_period_pct_chg)
			sell_dates = sell_period_pct_chg.where( sell_period_pct_chg >= sell_pct ).dropna().index.tolist()
			# print('sell dates:\n', sell_dates)
			# print('sell dates above')

			# print('Calculating future...')
			future = df_dji['Close'].shift(-buy_wait_days)
			# buy_period_diff = future.sub(today).dropna()
			# print('buy_period_diff:\n', buy_period_diff)
			# buy_period_cumsum = buy_period_diff.cumsum().dropna()
			# print('buy_period_cumsum:\n', buy_period_cumsum)
			buy_period_pct_chg = df_dji['Close'].pct_change(buy_wait_days).rename('Buy % Change', inplace=True)
			# print('buy_period_pct_chg:\n', buy_period_pct_chg)
			buy_dates = buy_period_pct_chg.where( buy_period_pct_chg <= buy_pct ).dropna().index.tolist()
			# print('buy dates:\n', buy_dates)
			# print('buy dates above')

			# If the sell dates or buy dates don't exist because conditions weren't met to invoke a transaction,
			# then store the parameters that caused that along with a final return of NaN
			# Else, run everything below including the for loop that calculates the trajectory.
			if len(sell_dates) == 0 or len(buy_dates) == 0:
				print('********************* WARNING: sell_dates or buy_dates is empty *********************')
				# Store the final return as NaN
				#print('Buy dates:\n', buy_dates)
				#traj_list.append([np.nan]*len(traj_list[0]))
				#parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, np.nan])
			else:
				# Pick alternating sell-buy dates, remove extra dates. Take the first sell date, find the next buy date, find the next
				# sell date, find the next buy date, etc. This process will remove extra buys and sells that occur more than once in a
				# row, essentially buying when money doesn't exist or selling multiple times back to back (which might be a good strategy,
				# but this will need to be investigated with consideration for transaction costs).
				# print('Picking alternating sell-buy dates...')
				sell_dates_list_temp = []
				buy_dates_list_temp = []
				s = sell_dates[0]

				# If sell_dates is the longer list, set l to the length of buy_dates,
				# and vice versa.
				if len(sell_dates) >= len(buy_dates):
					l = len(buy_dates)
				else:
					l = len(sell_dates)

				# Finding alternating sell and buy dates such that each sell is followed by a buy, which
				# is followed by another sell-buy pair, etc.
				sell_dates_list_temp.append(s)
				for i in range(l):
					for b in buy_dates:
						if b > s:# and b != buy_dates_list_temp[-1]:
							buy_dates_list_temp.append(b)
							break
					for s in sell_dates:
						if s > b:# and s != sell_dates_list_temp[-1]:
							sell_dates_list_temp.append(s)
							break

				# Just as important as making each first transaction a sell,
				# every last transaction must be a buy.
				# The final result sometimes produces a sell_dates_list_temp with 
				# an extra sell with no accompanying buy. If the sell_dates_list_temp
				# is one element longer than buy_dates_list_temp, cut the last value
				# from the sell list:
				# print('Removing trailing sell date (last transaction must be a buy)...')
				buy_len = len(buy_dates_list_temp)
				sell_len = len(sell_dates_list_temp)
				if sell_len > buy_len:
					sell_dates_list_temp = sell_dates_list_temp[0:buy_len]

				# After adjusting the length of sell_dates_list_temp, do one last check
				# to make sure it contains at least one date, if not skip the simulation.
				# print('Checking sell_dates_list_temp for at least one date present...')
				buy_len = len(buy_dates_list_temp)
				sell_len = len(sell_dates_list_temp)
				if sell_len != 0 and buy_len != 0:
					# Necessary to sort to maintain chronological order after the set operation:
					sell_dates_list_temp = sorted(list(set(sell_dates_list_temp)))
					buy_dates_list_temp = sorted(list(set(buy_dates_list_temp)))
					sell_dates = sell_dates_list_temp
					buy_dates = buy_dates_list_temp
					# print('sell_dates:\n', sell_dates)
					# print('buy_dates:\n', buy_dates)

					df_period_pct_chg = pd.concat((sell_period_pct_chg, buy_period_pct_chg), axis=1)
					# print('df_period_pct_chg:\n', df_period_pct_chg)

					# df_period_pct_chg.plot(use_index=True, title='Percent Change')
					# plt.show()

					# running_period_buy_bool = running_period_pct_chg.where( running_period_pct_chg < buy_pct, 0)
					# running_period_buy_bool = running_period_pct_chg <= buy_pct
					# print('running_period_buy_bool:\n', running_period_buy_bool)
					# running_period_buy = running_period_buy_bool * buy_portfolio_pct

					cumulative_pct_change = df_dji / df_dji.loc[start_date]
					cumulative_pct_change = cumulative_pct_change.squeeze()# Converting to a Series. Also works: cumulative_pct_change['Close']
					# print('cumulative_pct_change:\n', cumulative_pct_change)
					# cumulative_pct_change.plot(use_index=True)
					# plt.show()

					investments = pd.Series(data=[6000]*len(cumulative_pct_change), index=cumulative_pct_change.index)
					# print('investments:\n', investments)
					balance = cumulative_pct_change.multiply(investments)
					original_balance = balance
					original_return = balance.iloc[-1]

					# Go through the buy and sell dates, compute the original investment required to achieve the buys and sells balance, then
					# update the investments array to update the balance array. Do this for each sell-buy combination, then finally calculate
					# the final balance using the last computed original_investment, to get the final return on the portfolio.
					return_list = [original_return] # Initiate the list with the first final return
					final_return_list = []
					# start_date = '1985-01-29'
					# last_date = '2019-12-13'
					start_date = '1914-12-12'
					last_date = '2021-07-16'
					# print('sell_dates:\n', sell_dates)
					sell_date = sell_dates[0]
					# print('sell_date:\n', sell_date)
					adj_sell_date = sell_dates[0]-dt.timedelta(days=1)
					# print('Is adj_sell_date in balance index? ', balance.index.contains(adj_sell_date))		
					# print('sim # {}, start_date:adj_sell_date = {}:{}'.format(n, start_date, adj_sell_date))
					trajectory = pd.Series(data=balance.loc[start_date:adj_sell_date])
					# print('trajectory:\n', trajectory)
					for i in range(len(sell_dates)):
						# print('i****************************', i)
						# Separately calculating the sell information for creating a trajectory
						sell_amount = balance.loc[sell_dates[i]]*sell_portfolio_pct
						sell_investment = ((balance.loc[sell_dates[i]] - sell_amount) / cumulative_pct_change.loc[sell_dates[i]])
						sell_investments = pd.Series(data=[sell_investment]*len(cumulative_pct_change), index=cumulative_pct_change.index)
						sell_balance = cumulative_pct_change.multiply(sell_investments)
						# Update the traj with the first sell:
						trajectory = pd.concat((trajectory, sell_balance.loc[sell_dates[i]:buy_dates[i]-dt.timedelta(days=1)]), axis=0)
						# print('trajectory*:\n', trajectory)
						# For now let's assume that we use 100% of what's sold to buy back into the market. This is
						# a decent strategy because saving money for the future is almost never a good idea.
						buy_amount = sell_amount

						# Calculate the original investment required to achieve a certain balance
						original_investment = ((balance.loc[sell_dates[i]] - sell_amount) / cumulative_pct_change.loc[sell_dates[i]]) + (buy_amount / cumulative_pct_change.loc[buy_dates[i]])
						# print('original_investment:', original_investment)

						# Using original_investment, recalculate the entire balance array:
						investments = pd.Series(data=[original_investment]*len(cumulative_pct_change), index=cumulative_pct_change.index)
						# print('investments:\n', investments)
						balance = cumulative_pct_change.multiply(investments)
						# print('balance:\n', balance)
						return_list.append(balance.iloc[-1])
					# 	# Update balance with next sell_date. I don't think this is needed
					# 	balance = original_investment * cumulative_pct_change.loc[sell_dates[i+1]] # next sell date
						# If we're not on the last sell date, then create the trajectory that goes from the buy date
						# to the next sell date. If we are on the last sell date, 
						if i+1 < len(sell_dates):
							trajectory = pd.concat((trajectory, balance.loc[buy_dates[i]:sell_dates[i+1]-dt.timedelta(days=1)]), axis=0)
							# print('trajectory**:\n', trajectory)

						# Math behind the original_investment equation above:
						# new_balance = (original_investment * cumulative_pct_change.loc[buy_date]) + buy_amount
						# original_investment = new_balance / cumulative_pct_change.loc[buy_date]
						# Plugging new_balance into the original_investment equation:
						# original_investment += (buy_amount / cumulative_pct_change.loc[buy_date])
					trajectory = pd.concat((trajectory, balance.loc[buy_dates[i]:]), axis=0)
					# print('trajectory***:\n', trajectory)
					traj_list.append(trajectory.values)
				
					final_balance = balance

					final_return = original_investment * cumulative_pct_change.iloc[-1]
					parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, final_return])


					# Another way to calculate the final return: final_return = balance.iloc[-1]
					# print('final_return:', final_return)
					# print('return_list:\n', return_list)
					# print('trajectory:\n', trajectory)
				else:
					traj_list.append(original_balance)
					parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, np.nan])
					

	# print('parameters_list:\n', parameters_list)

	traj_arr = np.asarray(traj_list).T#.reshape((-1,3))
	# print('traj_arr:\n', traj_arr)
	# print('traj_arr.shape:\n', traj_arr.shape)

	df_traj = pd.DataFrame(data=traj_arr, index=cumulative_pct_change.index)
	# print('df_traj:\n', df_traj)
	original_balance.rename('Original Balance', inplace=True)
	df_traj = df_traj.join(original_balance, how='outer')
	# print('df_traj after joining original_balance:\n', df_traj)


	df_parameters = pd.DataFrame(data=parameters_list, columns=['Sell Wait Days','Sell Inc Pct','Sell Port Pct','Buy Wait Days','Buy Dec Pct','Final Return'])
	df_parameters.index.rename('Strategy', inplace=True)
	# print('df_parameters before sort:\n', df_parameters.to_string())
	df_parameters.sort_values(axis=0, by=['Final Return'], ascending=False, inplace=True)
	# print('df_parameters after sort:\n', df_parameters.to_string())

	ser_final_returns = df_parameters['Final Return'].dropna()
	print('ser_final_returns:\n', ser_final_returns.head(20))
	pct_winning = ser_final_returns[ser_final_returns > original_return].count() / ser_final_returns.count() * 100
	print('Percent winning strategies: {}%'.format(pct_winning))

	# DISPLAY ALL INVESTMENT STRATEGIES:
	# df_traj.plot(use_index=True, linewidth=0.7)
	# original_balance.plot(use_index=True, linewidth=3)
	# plt.show()

	# DISPLAY WINNING INVESTMENT STRATEGIES:
	df_traj.sort_values(axis=1, by=[last_date], ascending=False, inplace=True)
	df_traj.dropna(inplace=True)
	# print('df_traj sorted by highest final return:\n', df_traj.iloc[:,:30])#df_traj_sorted.loc[:,:'Original Balance'])
	df_traj.iloc[:,:30].plot(use_index=True, linewidth=1)
	original_balance.plot(use_index=True, linewidth=3)
	plt.xlabel('Date')
	plt.ylabel('Value, $')
	plt.savefig('sell_up_'+str(total_sims_to_run)+'_sims.png', bbox_inches='tight')
	plt.show()

	# Just verifying one last time that the original_balance is somewhere in df_traj
	print('df_traj columns:\n', df_traj.columns.values.tolist()) # To verify original_balance is somewhere in here

	# Pickle out the parameters and trajectories:
	df_parameters.to_pickle('parameters_sell_up_'+str(total_sims_to_run)+'_sims.pkl')
	df_traj.to_pickle('trajectories_sell_up_'+str(total_sims_to_run)+'_sims.pkl')




	# df_all_trajs = pd.concat((original_balance, final_balance, trajectory), axis=1)
	# df_all_trajs.rename(columns={0:'Original Balance', 1:'Final Balance', 2:'Trajectory'}, inplace=True)
	# original_balance.plot(use_index=True)
	# final_balance.plot(use_index=True)
	# trajectory.plot(use_index=True, legend=True)
	# print('df_trajectories:\n', df_trajectories)
	# df_all_trajs.plot(use_index=True)
	# plt.show()




	# Difficult to plot the buy-sell balance because it's a mess finding each
	# buy and sell balance and extrapolating over the various periods between
	# buys and sells. I'm mostly just interested in the final balance for
	# comparing the success of various strategies.



	return



# # sell_down() invokes sales when the market has gone down and buys when it has gone up.
# # Sell down: sell after the market has declined at a particular rate
# # Buy up: buy after the market has risen at a particular rate
# def sell_down(stock_index, year, total_sims_to_run):
	
# 	dji_file = stock_index + str(year) + '.csv'
# 	df_dji = pd.read_csv(dji_file, header=0, names=['Date','Close'])
# 	df_dji.Date = pd.to_datetime(df_dji.Date)
# 	df_dji.set_index('Date', inplace=True)
# 	print('df_dji:\n', df_dji)
# 	print('df_dji:\n', df_dji.head().to_string())

# 	start_date = df_dji.index[0]
# 	print('df_dji:\n', df_dji)


# 	# EA to maximize both the return on invested dollars and final portfolio balance
# 	# Variables: buy decline percentage, buy wait time, buy amount, increase percentage, sell wait time, sell amount
# 	# No saving of money (maximum is invested each month)
# 	# Weights for each
# 	# Equation to optimize: ??
# 	# Variables:


# 	buy_wait_days = np.random.randint(1, 30, size=1)[0]
# 	buy_wait_days = 3
# 	buy_pct = np.random.uniform(0.01,0.08) * -1
# 	buy_pct = 0.05 * -1
# 	buy_portfolio_pct = np.random.randint(1, 50, size=1)[0]
# 	buy_portfolio_pct = 0.1


# 	print('Buy wait time: {} days'.format(buy_wait_days))
# 	# decline_before_sell = np.random.random('uniform', buy_wait_days*-0.01, buy_wait_days*-0.05)
# 	# rate_of_return = df_pct_change.shift(buy_wait_days, fill_value=0).values
# 	today = df_dji['Close']
# 	future = df_dji['Close'].shift(-buy_wait_days)
# 	period_diff = future.sub(today).dropna()
# 	print('period_diff:\n', period_diff)
# 	running_period_sum = period_diff.cumsum().dropna()
# 	print('running_period_sum:\n', running_period_sum)
# 	running_period_pct_chg = df_dji['Close'].pct_change(buy_wait_days)
# 	print('running_period_pct_chg:\n', running_period_pct_chg)

# 	# running_period_buy_bool = running_period_pct_chg.where( running_period_pct_chg < buy_pct, 0)
# 	running_period_buy_bool = running_period_pct_chg <= buy_pct
# 	print('running_period_buy_bool:\n', running_period_buy_bool)
# 	running_period_buy = running_period_buy_bool * buy_portfolio_pct

# 	cumulative_pct_change = df_dji / df_dji.loc[start_date]
# 	# Convert to 
# 	cumulative_pct_change = cumulative_pct_change.squeeze()# Converting to a Series. Also works: cumulative_pct_change['Close']
# 	print('cumulative_pct_change:\n', cumulative_pct_change)
# 	# cumulative_pct_change.plot(use_index=True)
# 	# plt.show()

# 	investments = pd.Series(data=[6000]*len(cumulative_pct_change), index=cumulative_pct_change.index)
# 	print('investments:\n', investments)
# 	balance = cumulative_pct_change.multiply(investments)
# 	print('balance:\n', balance)
# 	# balance.plot(use_index=True)
# 	# plt.show()

# 	# These dates need to exist. If a date is on the weekend it won't exist and it will add it to the index end, causing problems.
# 	sell_dates = ['2000-01-18', '2007-10-01']
# 	buy_dates = ['2002-10-10', '2009-03-02']
# 	sell_amount = 0# 40000
# 	buy_amount = 0# 40000

# 	# buy_sell = pd.Series(data=[0]*len(cumulative_pct_change), index=cumulative_pct_change.index)
# 	# buy_sell_cumsum = buy_sell.cumsum()
# 	# # print('buy_sell_cumsum:\n', buy_sell_cumsum)
# 	# # buy_sell_cumsum.plot(use_index=True)
# 	# # plt.show()

# 	# investments_with_transactions = investments + buy_sell_cumsum
# 	# balance_with_transactions = cumulative_pct_change.multiply(investments_with_transactions)
	
# 	# # balance.plot(use_index=True)
# 	# # balance_with_transactions.plot(use_index=True)
# 	# # plt.show()

# 	traj_list = []
# 	parameters_list = []
# 	print('Running {} simulations...'.format(total_sims_to_run))
# 	for n in range(total_sims_to_run):
# 		pct_complete = n/total_sims_to_run*100
# 		print('Percentage complete: {:.1f}%'.format(pct_complete))

# 		# Sell variables:
# 		sell_wait_days = np.random.randint(1, 6000, size=1)[0]
# 		sell_pct = np.random.uniform(0.01,0.9) * -1
# 		sell_portfolio_pct = np.random.uniform(0.01,1.0)
# 		# print('sell_wait_days:', sell_wait_days)
# 		# print('sell_pct:', sell_pct)
# 		# print('sell_portfolio_pct:', sell_portfolio_pct)

# 		# Buy variables:
# 		buy_wait_days = np.random.randint(1, 10000, size=1)[0]#5
# 		buy_pct = np.random.uniform(0.01,0.9)#0.08
# 		# Note: buy_portfolio_pct = sell_portfolio_pct because this is all the money I have on hand.
# 		# print('buy_wait_days:', buy_wait_days)
# 		# print('buy_pct:', buy_pct)

# 		# Note: buy_portfolio_pct doesn't exist because I'm setting the buy_amount = sell_amount below
# 		# decline_before_sell = np.random.random('uniform', buy_wait_days*-0.01, buy_wait_days*-0.05)
# 		# rate_of_return = df_pct_change.shift(buy_wait_days, fill_value=0).values

# 		# Use sell_wait_days and buy_wait_days to shift the data and calculate the
# 		# percent change over the shifted period.
# 		today = df_dji['Close']
		
# 		future = df_dji['Close'].shift(-sell_wait_days)
# 		# buy_period_diff = future.sub(today).dropna()
# 		# print('buy_period_diff:\n', buy_period_diff)
# 		# buy_period_cumsum = buy_period_diff.cumsum().dropna()
# 		# print('buy_period_cumsum:\n', buy_period_cumsum)
# 		sell_period_pct_chg = df_dji['Close'].pct_change(sell_wait_days).rename('Sell % Change', inplace=True)
# 		# print('sell_period_pct_chg:\n', sell_period_pct_chg)
# 		sell_dates = sell_period_pct_chg.where( sell_period_pct_chg <= sell_pct ).dropna().index.tolist()
# 		# print('sell dates:\n', sell_dates)

# 		future = df_dji['Close'].shift(-buy_wait_days)
# 		# buy_period_diff = future.sub(today).dropna()
# 		# print('buy_period_diff:\n', buy_period_diff)
# 		# buy_period_cumsum = buy_period_diff.cumsum().dropna()
# 		# print('buy_period_cumsum:\n', buy_period_cumsum)
# 		buy_period_pct_chg = df_dji['Close'].pct_change(buy_wait_days).rename('Buy % Change', inplace=True)
# 		# print('buy_period_pct_chg:\n', buy_period_pct_chg)
# 		buy_dates = buy_period_pct_chg.where( buy_period_pct_chg >= buy_pct ).dropna().index.tolist()
# 		# print('buy dates:\n', buy_dates)

# 		# If the sell dates or buy dates don't exist because conditions weren't met to invoke a transaction,
# 		# then store the parameters that caused that along with a final return of NaN
# 		# Else, run everything below including the for loop that calculates the trajectory.
# 		if len(sell_dates) == 0 or len(buy_dates) == 0:
# 			# print('sell_dates or buy_dates is empty')
# 			# Store the final return as NaN
# 			parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, np.nan])
# 		else:
# 			# Pick alternating sell-buy dates, remove extra dates. Take the first sell date, find the next buy date, find the next
# 			# sell date, find the next buy date, etc. This process will remove extra buys and sells that occur more than once in a
# 			# row, essentially buying when money doesn't exist or selling multiple times back to back (which might be a good strategy,
# 			# but this will need to be investigated with consideration for transaction costs).
# 			sell_dates_list_temp = []
# 			buy_dates_list_temp = []
# 			s = sell_dates[0]

# 			# If sell_dates is the longer list, set l to the length of buy_dates,
# 			# and vice versa.
# 			if len(sell_dates) >= len(buy_dates):
# 				l = len(buy_dates)
# 			else:
# 				l = len(sell_dates)

# 			# Finding alternating sell and buy dates such that each sell is followed by a buy, which
# 			# is followed by another sell-buy pair, etc.
# 			sell_dates_list_temp.append(s)
# 			for i in range(l):
# 				for b in buy_dates:
# 					if b > s:# and b != buy_dates_list_temp[-1]:
# 						buy_dates_list_temp.append(b)
# 						break
# 				for s in sell_dates:
# 					if s > b:# and s != sell_dates_list_temp[-1]:
# 						sell_dates_list_temp.append(s)
# 						break

# 			# Just as important as making each first transaction a sell,
# 			# every last transaction must be a buy.
# 			# The final result sometimes produces a sell_dates_list_temp with 
# 			# an extra sell with no accompanying buy. If the sell_dates_list_temp
# 			# is one element longer than buy_dates_list_temp, cut the last value
# 			# from the sell list:
# 			buy_len = len(buy_dates_list_temp)
# 			sell_len = len(sell_dates_list_temp)
# 			if sell_len > buy_len:
# 				sell_dates_list_temp = sell_dates_list_temp[0:buy_len]

# 			# After adjusting the length of sell_dates_list_temp, do one last check
# 			# to make sure it contains at least one date, if not skip the simulation.
# 			buy_len = len(buy_dates_list_temp)
# 			sell_len = len(sell_dates_list_temp)
# 			if sell_len == 0 or buy_len == 0:
# 				parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, np.nan])
# 			else:
# 				# Necessary to sort to maintain chronological order after the set operation:
# 				sell_dates_list_temp = sorted(list(set(sell_dates_list_temp)))
# 				buy_dates_list_temp = sorted(list(set(buy_dates_list_temp)))
# 				sell_dates = sell_dates_list_temp
# 				buy_dates = buy_dates_list_temp
# 				# print('sell_dates:\n', sell_dates)
# 				# print('buy_dates:\n', buy_dates)

# 				df_period_pct_chg = pd.concat((sell_period_pct_chg, buy_period_pct_chg), axis=1)
# 				# print('df_period_pct_chg:\n', df_period_pct_chg)

# 				# df_period_pct_chg.plot(use_index=True, title='Percent Change')
# 				# plt.show()

# 				# running_period_buy_bool = running_period_pct_chg.where( running_period_pct_chg < buy_pct, 0)
# 				# running_period_buy_bool = running_period_pct_chg <= buy_pct
# 				# print('running_period_buy_bool:\n', running_period_buy_bool)
# 				# running_period_buy = running_period_buy_bool * buy_portfolio_pct

# 				cumulative_pct_change = df_dji / df_dji.loc[start_date]
# 				cumulative_pct_change = cumulative_pct_change.squeeze()# Converting to a Series. Also works: cumulative_pct_change['Close']
# 				# print('cumulative_pct_change:\n', cumulative_pct_change)
# 				# cumulative_pct_change.plot(use_index=True)
# 				# plt.show()

# 				investments = pd.Series(data=[6000]*len(cumulative_pct_change), index=cumulative_pct_change.index)
# 				# print('investments:\n', investments)
# 				balance = cumulative_pct_change.multiply(investments)
# 				original_balance = balance
# 				original_return = balance.iloc[-1]

# 				# Go through the buy and sell dates, compute the original investment required to achieve the buys and sells balance, then
# 				# update the investments array to update the balance array. Do this for each sell-buy combination, then finally calculate
# 				# the final balance using the last computed original_investment, to get the final return on the portfolio.
# 				return_list = [original_return] # Initiate the list with the first final return
# 				final_return_list = []
# 				# start_date = '1985-01-29'
# 				# last_date = '2019-12-13'
# 				start_date = '1914-12-12'
# 				last_date = '2021-07-16'
# 				# print('sell_dates:\n', sell_dates)
# 				sell_date = sell_dates[0]
# 				# print('sell_date:\n', sell_date)
# 				adj_sell_date = sell_dates[0]-dt.timedelta(days=1)
# 				# print('Is adj_sell_date in balance index? ', balance.index.contains(adj_sell_date))		
# 				# print('sim # {}, start_date:adj_sell_date = {}:{}'.format(n, start_date, adj_sell_date))
# 				trajectory = pd.Series(data=balance.loc[start_date:adj_sell_date])
# 				# print('trajectory:\n', trajectory)
# 				for i in range(len(sell_dates)):
# 					# print('i****************************', i)
# 					# Separately calculating the sell information for creating a trajectory
# 					sell_amount = balance.loc[sell_dates[i]]*sell_portfolio_pct
# 					sell_investment = ((balance.loc[sell_dates[i]] - sell_amount) / cumulative_pct_change.loc[sell_dates[i]])
# 					sell_investments = pd.Series(data=[sell_investment]*len(cumulative_pct_change), index=cumulative_pct_change.index)
# 					sell_balance = cumulative_pct_change.multiply(sell_investments)
# 					# Update the traj with the first sell:
# 					trajectory = pd.concat((trajectory, sell_balance.loc[sell_dates[i]:buy_dates[i]-dt.timedelta(days=1)]), axis=0)
# 					# print('trajectory*:\n', trajectory)
# 					# For now let's assume that we use 100% of what's sold to buy back into the market. This is
# 					# a decent strategy because saving money for the future is almost never a good idea.
# 					buy_amount = sell_amount

# 					# Calculate the original investment required to achieve a certain balance
# 					original_investment = ((balance.loc[sell_dates[i]] - sell_amount) / cumulative_pct_change.loc[sell_dates[i]]) + (buy_amount / cumulative_pct_change.loc[buy_dates[i]])
# 					# print('original_investment:', original_investment)

# 					# Using original_investment, recalculate the entire balance array:
# 					investments = pd.Series(data=[original_investment]*len(cumulative_pct_change), index=cumulative_pct_change.index)
# 					# print('investments:\n', investments)
# 					balance = cumulative_pct_change.multiply(investments)
# 					# print('balance:\n', balance)
# 					return_list.append(balance.iloc[-1])
# 				# 	# Update balance with next sell_date. I don't think this is needed
# 				# 	balance = original_investment * cumulative_pct_change.loc[sell_dates[i+1]] # next sell date
# 					# If we're not on the last sell date, then create the trajectory that goes from the buy date
# 					# to the next sell date. If we are on the last sell date, 
# 					if i+1 < len(sell_dates):
# 						trajectory = pd.concat((trajectory, balance.loc[buy_dates[i]:sell_dates[i+1]-dt.timedelta(days=1)]), axis=0)
# 						# print('trajectory**:\n', trajectory)

# 					# Math behind the original_investment equation above:
# 					# new_balance = (original_investment * cumulative_pct_change.loc[buy_date]) + buy_amount
# 					# original_investment = new_balance / cumulative_pct_change.loc[buy_date]
# 					# Plugging new_balance into the original_investment equation:
# 					# original_investment += (buy_amount / cumulative_pct_change.loc[buy_date])
# 				trajectory = pd.concat((trajectory, balance.loc[buy_dates[i]:]), axis=0)
# 				# print('trajectory***:\n', trajectory)
# 				traj_list.append(trajectory.values)
			
# 				final_balance = balance

# 				final_return = original_investment * cumulative_pct_change.iloc[-1]
# 				parameters_list.append([sell_wait_days, sell_pct, sell_portfolio_pct, buy_wait_days, buy_pct, final_return])


# 				# Another way to calculate the final return: final_return = balance.iloc[-1]
# 				# print('final_return:', final_return)
# 				# print('return_list:\n', return_list)
# 				# print('trajectory:\n', trajectory)

# 	# print('parameters_list:\n', parameters_list)

# 	traj_arr = np.asarray(traj_list).T#.reshape((-1,3))
# 	# print('traj_arr:\n', traj_arr)
# 	# print('traj_arr.shape:\n', traj_arr.shape)

# 	df_traj = pd.DataFrame(data=traj_arr, index=cumulative_pct_change.index)
# 	# print('df_traj:\n', df_traj)
# 	original_balance.rename('Original Balance', inplace=True)
# 	df_traj = df_traj.join(original_balance, how='outer')
# 	# print('df_traj after joining original_balance:\n', df_traj)


# 	df_parameters = pd.DataFrame(data=parameters_list, columns=['Sell Wait Days','Sell Dec Pct','Sell Port Pct','Buy Wait Days','Buy Inc Pct','Final Return'])
# 	df_parameters.index.rename('Strategy', inplace=True)
# 	# print('df_parameters before sort:\n', df_parameters.to_string())
# 	df_parameters.sort_values(axis=0, by=['Final Return'], ascending=False, inplace=True)
# 	# print('df_parameters after sort:\n', df_parameters.to_string())

# 	ser_final_returns = df_parameters['Final Return'].dropna()
# 	print('ser_final_returns:\n', ser_final_returns.head(20))
# 	pct_winning = ser_final_returns[ser_final_returns > original_return].count() / ser_final_returns.count() * 100
# 	print('Percent winning strategies: {}%'.format(pct_winning))

# 	# DISPLAY ALL INVESTMENT STRATEGIES:
# 	# df_traj.plot(use_index=True, linewidth=0.7)
# 	# original_balance.plot(use_index=True, linewidth=3)
# 	# plt.show()

# 	# DISPLAY WINNING INVESTMENT STRATEGIES:
# 	df_traj.sort_values(axis=1, by=[last_date], ascending=False, inplace=True)
# 	df_traj.dropna(inplace=True)
# 	print('df_traj sorted by highest final return:\n', df_traj.iloc[:,:10])#df_traj_sorted.loc[:,:'Original Balance'])
# 	df_traj.iloc[:,:10].plot(use_index=True, linewidth=1) # .iloc[:, :30] gets the 30 best trajectories
# 	original_balance.plot(use_index=True, linewidth=3)
# 	plt.xlabel('Date')
# 	plt.ylabel('Portfolio Value, $')
# 	plt.savefig('sell_down_'+str(total_sims_to_run)+'_sims.png', bbox_inches='tight')
# 	plt.show()

# 	# Just verifying one last time that the original_balance is somewhere in df_traj
# 	print('df_traj columns:\n', df_traj.columns.values.tolist())

# 	# Pickle out the parameters and trajectories:
# 	df_parameters.to_pickle('parameters_sell_down_'+str(total_sims_to_run)+'_sims.pkl')
# 	df_traj.to_pickle('trajectories_sell_down_'+str(total_sims_to_run)+'_sims.pkl')




# 	# df_all_trajs = pd.concat((original_balance, final_balance, trajectory), axis=1)
# 	# df_all_trajs.rename(columns={0:'Original Balance', 1:'Final Balance', 2:'Trajectory'}, inplace=True)
# 	# original_balance.plot(use_index=True)
# 	# final_balance.plot(use_index=True)
# 	# trajectory.plot(use_index=True, legend=True)
# 	# print('df_trajectories:\n', df_trajectories)
# 	# df_all_trajs.plot(use_index=True)
# 	# plt.show()




# 	# Difficult to plot the buy-sell balance because it's a mess finding each
# 	# buy and sell balance and extrapolating over the various periods between
# 	# buys and sells. I'm mostly just interested in the final balance for
# 	# comparing the success of various strategies.

# 	return


def import_trajs(sell_up_simulations, sell_down_simulations, rolling_window):
	df_traj_sell_up = pd.read_pickle('trajectories_sell_up_' + str(sell_up_simulations) + '_sims.pkl')
	df_parameters_sell_up = pd.read_pickle('parameters_sell_up_' + str(sell_up_simulations) + '_sims.pkl')
	df_sell_dates_sell_up = pd.read_pickle('sell_dates_sell_up_' + str(sell_up_simulations) + '_sims.pkl')
	df_buy_dates_sell_up = pd.read_pickle('buy_dates_sell_up_' + str(sell_up_simulations) + '_sims.pkl')

	# Renaming the columns because they weren't correctly named, problem problem needs to be fixed in sell_up():
	df_parameters_sell_up.rename(columns={'Sell Dec Pct':'Sell Inc Pct', 'Buy Inc Pct':'Buy Dec Pct'}, inplace=True)

	print('df_traj_sell_up:\n', df_traj_sell_up)
	print('df_parameters_sell_up:\n', df_parameters_sell_up)
	print('df_sell_dates_sell_up:\n', df_sell_dates_sell_up)
	print('df_buy_dates_sell_up:\n', df_buy_dates_sell_up)



	df_traj_sell_down = pd.read_pickle('trajectories_sell_down_' + str(sell_down_simulations) + '_sims.pkl')
	df_parameters_sell_down = pd.read_pickle('parameters_sell_down_' + str(sell_down_simulations) + '_sims.pkl')

	# Change Sell Dec/Inc Pct to Sell Pct and Buy Dec/Inc Pct to Buy Pct
	# THIS CAN BE DONE IN THE PREVIOUS FUNCTIONS BUT I'M DOING IT HERE FOR CONVENIENCE
	df_parameters_sell_up.rename(columns={'Sell Pct':'Sell Fraction Change', 'Sell Port Pct':'Sell Fraction Portfolio', 'Buy Pct':'Buy Fraction Change'}, inplace=True)
	df_parameters_sell_down.rename(columns={'Sell Pct':'Sell Fraction Change', 'Sell Port Pct':'Sell Fraction Portfolio', 'Buy Pct':'Buy Fraction Change'}, inplace=True)

	# Plotting a few trajs for fun:
	# df_traj_sell_up.plot.line(use_index=True, y=[2015, 3518, 22, 'Original Balance'], linewidth=0.5)
	df_traj_sell_up.plot.line(use_index=True, y=[5, 10, 23, 'Original Balance'], linewidth=0.5)
	plt.yscale('log')
	plt.show()

	# Plotting all sell up strategies with sell wait days 4348 to 4351:
	# mask = (df_parameters_sell_up['Sell Wait Days'] <= 4351) & (4349 <= df_parameters_sell_up['Sell Wait Days'])
	mask = (df_parameters_sell_up['Sell Wait Days'] <= 4500) & (5300 <= df_parameters_sell_up['Sell Wait Days'])
	print(df_parameters_sell_up.loc[ mask, : ])
	diff = len(df_parameters_sell_up.index) - len(df_traj_sell_up.columns)
	print(diff)
	missing_trajs = df_parameters_sell_up.index[~df_parameters_sell_up.index.isin(df_traj_sell_up.columns)]
	print('trajs not present:\n', missing_trajs)
	unique_missing_trajs = missing_trajs.drop_duplicates()
	print(len(unique_missing_trajs))

	print(df_traj_sell_up.columns)

	trajs_of_interest = df_parameters_sell_up.loc[ mask, : ].index.to_list()
	print('trajs_of_interest:\n', trajs_of_interest)

	fig, ax = plt.subplots()
	# df_traj_sell_up.plot.line(use_index=True, y=[3600, 2288, 3028, 5559], linewidth=0.5, ax=ax)
	df_traj_sell_up.plot.line(use_index=True, y=['Original Balance'], linewidth=1, ax=ax)
	plt.show()
	

	df_parameters_sell_up_strategies_failed = df_parameters_sell_up.loc[df_parameters_sell_up['Final Return'].isnull()]
	print('df_parameters_sell_up_strategies_failed:\n', df_parameters_sell_up_strategies_failed)
	# print('Failed strategies in parameters:\n', df_parameters_sell_up_strategies_failed.loc[2288, :])
	# print('Failed strategies in trajs:\n', df_traj_sell_up.loc[:, trajs_of_interest])
	cols = df_parameters_sell_up.index[df_parameters_sell_up.index.isin(trajs_of_interest)]
	print('Trajs of interest present in df_parameters_sell_up:\n', cols)
	# Trajs of interest: [3600, 2288, 7223, 3028, 5559, 9756, 9008]
	# Missing from df_traj_sell_up: [7223, 9756, 9008]
	# Missing from df_parameters_sell_up: All missing
	print('df_parameters_sell_up[trajs_of_interest]:\n', df_parameters_sell_up.loc[trajs_of_interest, :])
	print('Number of rows in df_parameters_sell_up with NaN Final Return values:\n', df_parameters_sell_up['Final Return'].isnull().sum())
	print('Number of columns in df_traj_sell_up with NaN Final Return values:\n', df_traj_sell_up.iloc[-1, :].isnull().sum())



	print('df_traj_sell_up:\n', df_traj_sell_up)
	print('df_parameters_sell_up:\n', df_parameters_sell_up.to_string())
	print('Sell up strategy 18:\n', df_parameters_sell_up.loc[18, :])

	print('df_traj_sell_down:\n', df_traj_sell_down)
	print('df_parameters_sell_down:\n', df_parameters_sell_down)

	df_sup_corr = df_parameters_sell_up.corr() # Sell up parameters
	print('df_sup_corr:\n', df_sup_corr.to_string())

	df_sdp_corr = df_parameters_sell_down.corr()
	print('df_sdp_corr:\n', df_sdp_corr.to_string())

	df_parameters_sell_up_best = df_parameters_sell_up.iloc[0:50, :]
	df_parameters_sell_down_best = df_parameters_sell_down.iloc[0:50, :]

	final_original_balance = df_traj_sell_down['Original Balance'].iloc[-1] # Same for both sell up and sell down

	# Inspect NaN rows:
	nan_rows_sell_up = df_parameters_sell_up#.loc[ df_parameters_sell_up['Final Return'] == np.nan ]
	print('nan_rows_sell_up:\n', nan_rows_sell_up.to_string())
	n_sell_up_strategies = df_parameters_sell_up.shape[0]
	n_sell_up_strategies_failed = df_parameters_sell_up.loc[df_parameters_sell_up['Final Return'].isnull()].shape[0]
	pct_sell_up_strategies_failed = n_sell_up_strategies_failed / n_sell_up_strategies * 100
	print('Number of Sell Up strategies attempted:', n_sell_up_strategies)
	print('Number of Sell Up strategies failed:', n_sell_up_strategies_failed)
	print('Percent Sell Up strategies that failed: {:.1f}%'.format(pct_sell_up_strategies_failed))
	n_sell_down_strategies = df_parameters_sell_down.shape[0]
	n_sell_down_strategies_failed = df_parameters_sell_down.loc[df_parameters_sell_down['Final Return'].isnull()].shape[0]
	pct_sell_down_strategies_failed = n_sell_down_strategies_failed / n_sell_down_strategies * 100
	print('Number of Sell Down strategies attempted:', n_sell_down_strategies)
	print('Number of Sell Down strategies failed:', n_sell_down_strategies_failed)
	print('Percent Sell Down strategies that failed: {:.1f}%'.format(pct_sell_down_strategies_failed))

	# Make a column of W's and L's to indicate if a strategy beat or lost to the market
	# df_parameters_sell_up['Result'] = np.where(df_parameters_sell_up['Final Return'] > final_original_balance, 'Won', 'Lost')
	# df_parameters_sell_down['Result'] = np.where(df_parameters_sell_down['Final Return'] > final_original_balance, 'Won', 'Lost')

	df_parameters_sell_up.dropna(inplace=True)
	df_parameters_sell_down.dropna(inplace=True)

	# Make a column of W's (1's) and L's (0's) to indicate if a strategy beat or lost to the market
	df_parameters_sell_up['Result'] = np.where(df_parameters_sell_up['Final Return'] > final_original_balance, 1, 0)
	df_parameters_sell_down['Result'] = np.where(df_parameters_sell_down['Final Return'] > final_original_balance, 1, 0)

	# Make a column of number of transactions for each strategy:
	# Performing analysis of transactions. Move this down to the parameters analysis:
	df_transactions = df_sell_dates_sell_up.count()


	# Number of sells, number of buys, date of first trade for largest winning and losing strategies, time winning, time losing:
	df_n_sells_sell_up = df_sell_dates_sell_up.count(axis=0)
	df_n_buys_sell_up = df_buy_dates_sell_up.count(axis=0)
	# df_n_sells_sell_up.index.rename('Strategy', inplace=True)
	# df_n_buys_sell_up.index.rename('Strategy', inplace=True)
	# print('df_n_sells_sell_up:\n', df_n_sells_sell_up)
	# print('df_n_buys_sell_up:\n', df_n_buys_sell_up)
	if df_n_sells_sell_up.equals(df_n_buys_sell_up):
		df_n_transactions_sell_up = pd.concat((df_n_sells_sell_up, df_n_buys_sell_up), axis=1, join='outer')
		df_n_transactions_sell_up.rename(columns={0:'Sell',1:'Buy'}, inplace=True)
		print('df_n_transactions_sell_up:\n', df_n_transactions_sell_up)
		df_parameters_sell_up = df_parameters_sell_up.merge(df_n_sells_sell_up.to_frame(name='Total Transactions'), left_index=True, right_index=True)
		print('df_parameters_sell_up:\n', df_parameters_sell_up.to_string())
	else:
		print('WARNING: Number of Sells and Buys is not equivalent')
	
	df_parameters_sell_up['Sell Velocity'] = df_parameters_sell_up['Sell Fraction Change'] / df_parameters_sell_up['Sell Wait Days']
	df_parameters_sell_up['Buy Velocity'] = df_parameters_sell_up['Buy Fraction Change'] / df_parameters_sell_up['Buy Wait Days']
	df_parameters_sell_up['Sell Buy Velocity Ratio'] = df_parameters_sell_up['Sell Velocity'] / df_parameters_sell_up['Buy Velocity']

	df_parameters_sell_up_corr = df_parameters_sell_up.corr()
	df_corr_temp = df_parameters_sell_up_corr.copy()
	df_corr_temp.rename(columns={
								'Sell Wait Days':'Sell Days',
								'Sell Fraction Change':'Sell % Change',
								'Sell Fraction Portfolio':'Sell % Port',
								'Buy Wait Days':'Buy Days',
								'Buy Fraction Change':'Buy % Change',
								'Final Return':'Return',
								'Total Transactions':'n Trans',
								'Sell Velocity':'Sell V',
								'Buy Velocity':'Buy V',
								'Sell Buy Velocity Ratio':'SB V Ratio'
								},
								inplace=True)
	print('df_corr_temp:\n', df_corr_temp.to_string())
	# print('df_parameters_sell_up_corr:\n', df_parameters_sell_up_corr.to_string())
	return
	# traj = 18
	# print('df_transactions:\n', df_transactions)
	# print('df_sell_dates_sell_up.loc[{}]:\n{}'.format(traj, df_sell_dates_sell_up.loc[:, traj].dropna().to_string()))
	# print('df_buy_dates_sell_up.loc[{}]:\n{}'.format(traj, df_buy_dates_sell_up.loc[:, traj].dropna().to_string()))
	# print('df_sell_dates_sell_up.loc[{}].shape:\n{}'.format(traj, df_sell_dates_sell_up.loc[:, traj].dropna().shape))
	# print('df_buy_dates_sell_up.loc[{}].shape:\n{}'.format(traj, df_buy_dates_sell_up.loc[:, traj].dropna().shape))
	# print('df_traj_sell_up.loc[:, {}]:\n{}'.format(traj, df_traj_sell_up.loc[:, traj]))
	# print('df_parameters_sell_up.loc[{},:]:\n'.format(traj, df_parameters_sell_up.loc[traj,:]))
	# fig, ax = plt.subplots()
	# df_traj_sell_up.loc[:, [traj, 'Original Balance']].plot.line(use_index=True, ax=ax, color=['k','pink'], linewidth=0.5)
	# plt.show()



	# Fraction winning vs Parameter value
	var_list = df_parameters_sell_up.columns[0:5].to_list()
	print('var_list:\n', var_list)
	fig, ax = plt.subplots(2,5, figsize=(12,6))
	for n, var in enumerate(var_list):
		print('var:\n', var)

		df_sell_up_grouped_rolling = df_parameters_sell_up.groupby(var)['Result'].mean().rolling(rolling_window).mean()
		df_sell_down_grouped_rolling = df_parameters_sell_down.groupby(var)['Result'].mean().rolling(rolling_window).mean()

		df_sell_up_grouped_rolling.plot.line(use_index=True, color='k', ax=ax[0,n])
		df_sell_down_grouped_rolling.plot.line(use_index=True, color='k', ax=ax[1,n])

		ax[0,n].set_ylim([0, 1])
		ax[1,n].set_ylim([0, 1])

		ax[0,n].axhline(y=0.5, c='blue', linewidth=1, ls='--')
		ax[1,n].axhline(y=0.5, c='blue', linewidth=1, ls='--')

		# Formatting the plot so things look nice

		ax[0,0].set_ylabel('Fraction Winning')
		ax[0,0].set_title('Sell Up, Buy Down')
		if n > 0:
			ax[0,n].axes.yaxis.set_visible(False)

		ax[1,0].set_ylabel('Fraction Winning')
		ax[1,0].set_title('Sell Down, Buy Up')
		if n > 0:
			ax[1,n].axes.yaxis.set_visible(False)

	plt.subplots_adjust(hspace=0.5)
	plt.savefig('trajs_params_rolling_pct_winning.png', bbox_inches='tight')
	plt.show()

	# Final return vs Year of first trade:



	# Final return vs Year of second trade:



	# Find the winning strategies:
	df_params_sell_up_best = df_parameters_sell_up.loc[ df_parameters_sell_up['Final Return'] > final_original_balance, : ].copy()
	df_params_sell_up_worst = df_parameters_sell_up.loc[ df_parameters_sell_up['Final Return'] < final_original_balance, : ].copy()
	num_sell_up_best = df_params_sell_up_best.shape[0]
	num_sell_up = df_parameters_sell_up.shape[0]
	pcnt_sell_up_best = num_sell_up_best / num_sell_up * 100 # Percentage of all strategies that won
	print('Strategies: {}, Best strategies: {}, Percent of sell up strategies that won: {:.1f}%'.format(num_sell_up, num_sell_up_best, pcnt_sell_up_best))
	
	df_params_sell_down_best = df_parameters_sell_down.loc[ df_parameters_sell_down['Final Return'] > final_original_balance, : ].copy()
	df_params_sell_down_worst = df_parameters_sell_down.loc[ df_parameters_sell_down['Final Return'] < final_original_balance, : ].copy()
	num_sell_down_best = df_params_sell_down_best.shape[0]
	num_sell_down = df_parameters_sell_down.shape[0]
	pcnt_sell_down_best = num_sell_down_best / num_sell_down * 100 # Percentage of all strategies that won
	print('Strategies: {}, Best strategies: {}, Percent of sell down strategies that won: {:.1f}%'.format(num_sell_down, num_sell_down_best, pcnt_sell_down_best))


	
	# SCATTER PLOT OF FINAL RETURN VS EACH PARAMETER:
	fig, ax = plt.subplots(2,5, figsize=(12,6))
	df_params_sell_up_best.plot.scatter(x='Sell Wait Days', y='Final Return', ax=ax[0,0], s=1, c='k')
	df_params_sell_up_best.plot.scatter(x='Sell Fraction Change', y='Final Return', ax=ax[0,1], s=1, c='k')
	df_params_sell_up_best.plot.scatter(x='Sell Fraction Portfolio', y='Final Return', ax=ax[0,2], s=1, c='k')
	df_params_sell_up_best.plot.scatter(x='Buy Wait Days', y='Final Return', ax=ax[0,3], s=1, c='k')
	df_params_sell_up_best.plot.scatter(x='Buy Fraction Change', y='Final Return', ax=ax[0,4], s=1, c='k')
	df_params_sell_up_worst.plot.scatter(x='Sell Wait Days', y='Final Return', ax=ax[0,0], s=1, c='r')
	df_params_sell_up_worst.plot.scatter(x='Sell Fraction Change', y='Final Return', ax=ax[0,1], s=1, c='r')
	df_params_sell_up_worst.plot.scatter(x='Sell Fraction Portfolio', y='Final Return', ax=ax[0,2], s=1, c='r')
	df_params_sell_up_worst.plot.scatter(x='Buy Wait Days', y='Final Return', ax=ax[0,3], s=1, c='r')
	df_params_sell_up_worst.plot.scatter(x='Buy Fraction Change', y='Final Return', ax=ax[0,4], s=1, c='r')
	for n in range(5):
		ax[0,n].axhline(y=final_original_balance, c='gray', linewidth=1, ls='--')


	df_params_sell_down_best.plot.scatter(x='Sell Wait Days', y='Final Return', ax=ax[1,0], s=1, c='k')
	df_params_sell_down_best.plot.scatter(x='Sell Fraction Change', y='Final Return', ax=ax[1,1], s=1, c='k')
	df_params_sell_down_best.plot.scatter(x='Sell Fraction Portfolio', y='Final Return', ax=ax[1,2], s=1, c='k')
	df_params_sell_down_best.plot.scatter(x='Buy Wait Days', y='Final Return', ax=ax[1,3], s=1, c='k')
	df_params_sell_down_best.plot.scatter(x='Buy Fraction Change', y='Final Return', ax=ax[1,4], s=1, c='k')
	df_params_sell_down_worst.plot.scatter(x='Sell Wait Days', y='Final Return', ax=ax[1,0], s=1, c='r')
	df_params_sell_down_worst.plot.scatter(x='Sell Fraction Change', y='Final Return', ax=ax[1,1], s=1, c='r')
	df_params_sell_down_worst.plot.scatter(x='Sell Fraction Portfolio', y='Final Return', ax=ax[1,2], s=1, c='r')
	df_params_sell_down_worst.plot.scatter(x='Buy Wait Days', y='Final Return', ax=ax[1,3], s=1, c='r')
	df_params_sell_down_worst.plot.scatter(x='Buy Fraction Change', y='Final Return', ax=ax[1,4], s=1, c='r')
	for n in range(5):
		ax[1,n].axhline(y=final_original_balance, c='gray', linewidth=1, ls='--')

	ax[0,0].set_ylabel('Final Value, $')
	ax[0,0].set_title('Sell Up, Buy Down: {:.1f}% Winning'.format(pcnt_sell_up_best))
	for n in range(1,5):
		ax[0,n].axes.yaxis.set_visible(False)

	ax[1,0].set_ylabel('Final Value, $')
	ax[1,0].set_title('Sell Down, Buy Up: {:.1f}% Winning'.format(pcnt_sell_down_best))
	for n in range(1,5):
		ax[1,n].axes.yaxis.set_visible(False)

	plt.subplots_adjust(hspace=0.5)
	plt.savefig('trajs_params_final_return.png', bbox_inches='tight')
	plt.show()

	print('df_parameters_sell_down.columns:\n', df_parameters_sell_down.columns.tolist())
	

	
	
	# # Combine Wait Days and Inc/Dec Pct to Velocity so we can concatenate the two datasets after clustering
	# # THIS MAY NOT BE NECESSARY BUT WOULD BE INTERESTING TO INCLUDE A VELOCITY PLOT AT SOME POINT
	# df_parameters_sell_up['Sell Velocity'] = df_parameters_sell_up['Sell Inc Pct'] / df_parameters_sell_up['Sell Wait Days']
	# df_parameters_sell_up['Buy Velocity'] = df_parameters_sell_up['Buy Dec Pct'] / df_parameters_sell_up['Buy Wait Days']
	# df_parameters_sell_down['Sell Velocity'] = df_parameters_sell_down['Sell Dec Pct'] / df_parameters_sell_up['Sell Wait Days']
	# df_parameters_sell_down['Buy Velocity'] = df_parameters_sell_down['Buy Inc Pct'] / df_parameters_sell_up['Buy Wait Days']


	# See if the strategies cluster into two groups:
	from sklearn.cluster import KMeans

	df_parameters_sell_up_cluster = df_parameters_sell_up.dropna() # Dropping rows for which Final Return is NaN
	df_parameters_sell_down_cluster = df_parameters_sell_down.dropna() # Dropping rows for which Final Return is NaN. 


	# Elbow criterion to find optimal number of clusters:
	sse = {}
	data = df_parameters_sell_down_cluster.iloc[:, :-1]
	for k in range(1, 20):
	    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data)
	    data["clusters"] = kmeans.labels_
	    #print(data["clusters"])
	    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
	plt.figure()
	plt.plot(list(sse.keys()), list(sse.values()))
	plt.xlabel("Number of cluster")
	plt.ylabel("SSE")
	plt.show()


	n_clusters = 9

	# Sell up clustering:
	kmeans_model = KMeans(n_clusters=n_clusters, random_state=1)
	strategy_distances = kmeans_model.fit_transform(df_parameters_sell_up_cluster.iloc[:, :-1])
	labels = kmeans_model.labels_
	print('Crosstab of cluster groups and strategy results:\n', pd.crosstab(labels, df_parameters_sell_up_cluster['Result'], normalize=True))
	# Plot differently colored clusters:
	df_labels = pd.DataFrame(data=labels, columns=['Cluster'])
	df_labels.index.rename('Strategy', inplace=True)
	print('labels:\n', df_labels)
	df_parameters_sell_up_cluster = df_parameters_sell_up_cluster.join(df_labels, how='inner')
	print('df_parameters_sell_up_cluster:\n', df_parameters_sell_up_cluster)

	# Sell down clustering:
	kmeans_model = KMeans(n_clusters=n_clusters, random_state=1)
	strategy_distances = kmeans_model.fit_transform(df_parameters_sell_down_cluster.iloc[:, :-1])
	labels = kmeans_model.labels_
	print('Crosstab of cluster groups and strategy results:\n', pd.crosstab(labels, df_parameters_sell_down_cluster['Result'], normalize=True))
	
	# Plot differently colored clusters:
	df_labels = pd.DataFrame(data=labels, columns=['Cluster'])
	df_labels.index.rename('Strategy', inplace=True)
	print('labels:\n', df_labels)
	df_parameters_sell_down_cluster = df_parameters_sell_down_cluster.join(df_labels, how='inner')
	print('df_parameters_sell_down_cluster:\n', df_parameters_sell_down_cluster)
	print('cols:\n', df_parameters_sell_down_cluster.columns.tolist())

	# Concatenate then melt the sell up and sell down cluster dataframes:
	df_parameters_sell_up_cluster['Strategy'] = 'Sell Up Buy Down'
	df_parameters_sell_down_cluster['Strategy'] = 'Sell Down Buy Up'

	df_parameters_cluster = pd.concat((df_parameters_sell_up_cluster, df_parameters_sell_down_cluster), axis=0)
	df_parameters_cluster.reset_index(drop=True, inplace=True) # Drop Strategy index. Already have a 'Strategy' column.
	id_cols = ['Result', 'Cluster', 'Strategy', 'Final Return']
	value_cols = df_parameters_cluster.columns.tolist()[0:5]
	print('value_cols:\n', value_cols)
	# Melting to organize into long format for seaborn facet grid
	df_parameters_cluster_long = df_parameters_cluster.melt(id_vars=id_cols, value_vars=value_cols, var_name='Parameter')
	print('df_parameters_cluster_long:\n', df_parameters_cluster_long)


	# Clustering Plots: Scatter plot of final return vs each parameter

	# g = sns.FacetGrid(data=df_parameters_cluster_long, hue='Cluster', hue_order=list(range(0,5)), aspect=1.6)
	# g.map(sns.scatterplot, 'value', 'Final Return', s=2).add_legend()

	'''
	g = sns.PairGrid(df_parameters_sell_up_cluster, y_vars='Final Return', x_vars=value_cols, hue='Cluster', aspect=1.6)
	g.map(sns.scatterplot, s=7)
	ax1, ax2, ax3, ax4, ax5 = g.axes[0]
	ax1.axhline(final_original_balance, ls='--')
	ax2.axhline(final_original_balance, ls='--')
	ax3.axhline(final_original_balance, ls='--')
	ax4.axhline(final_original_balance, ls='--')
	ax5.axhline(final_original_balance, ls='--')
	ax1.set_ylabel('Final Return, $')
	# plt.ylabel('Final Return, $')
	plt.subplots_adjust(left=0.1)
	plt.savefig('Sell Up Cluster Plot.png', bbox_inches='tight')
	plt.show()
	# g.set(ylim=(-1, 11), yticks=[0, 5, 10])

	g = sns.PairGrid(df_parameters_sell_down_cluster, y_vars='Final Return', x_vars=value_cols, hue='Cluster', aspect=1.6)
	g.map(sns.scatterplot, s=7)
	ax1, ax2, ax3, ax4, ax5 = g.axes[0]
	ax1.axhline(final_original_balance, ls='--')
	ax2.axhline(final_original_balance, ls='--')
	ax3.axhline(final_original_balance, ls='--')
	ax4.axhline(final_original_balance, ls='--')
	ax5.axhline(final_original_balance, ls='--')
	ax1.set_ylabel('Final Return, $')
	plt.ylabel('Final Return, $')
	plt.subplots_adjust(left=0.1)
	plt.savefig('Sell Down Cluster Plot.png', bbox_inches='tight')
	plt.show()
	'''

	# SCATTER PLOT OF FINAL RETURN VS EACH PARAMETER:
	color_dict = {0:'pink', 1:'green', 2:'blue', 3:'orange', 4:'purple', 5:'red', 6:'gray', 7:'black', 8:'yellow'}
	df_parameters_sell_up_cluster['color'] = df_parameters_sell_up_cluster.Cluster.map(color_dict)
	df_parameters_sell_down_cluster['color'] = df_parameters_sell_down_cluster.Cluster.map(color_dict)
	print('Unique clusters:\n', df_parameters_sell_up_cluster.Cluster.unique())

	fig, ax = plt.subplots(2,5, figsize=(12,6))
	df_parameters_sell_up_cluster.plot.scatter(x='Sell Wait Days', y='Final Return', ax=ax[0,0], s=1, c=df_parameters_sell_up_cluster.color)
	df_parameters_sell_up_cluster.plot.scatter(x='Sell Fraction Change', y='Final Return', ax=ax[0,1], s=1, c=df_parameters_sell_up_cluster.color)
	df_parameters_sell_up_cluster.plot.scatter(x='Sell Fraction Portfolio', y='Final Return', ax=ax[0,2], s=1, c=df_parameters_sell_up_cluster.color)
	df_parameters_sell_up_cluster.plot.scatter(x='Buy Wait Days', y='Final Return', ax=ax[0,3], s=1, c=df_parameters_sell_up_cluster.color)
	df_parameters_sell_up_cluster.plot.scatter(x='Buy Fraction Change', y='Final Return', ax=ax[0,4], s=1, c=df_parameters_sell_up_cluster.color)
	for n in range(5):
		ax[0,n].axhline(y=final_original_balance, c='gray', linewidth=1, ls='--')

	df_parameters_sell_down_cluster.plot.scatter(x='Sell Wait Days', y='Final Return', ax=ax[1,0], s=1, c=df_parameters_sell_down_cluster.color)
	df_parameters_sell_down_cluster.plot.scatter(x='Sell Fraction Change', y='Final Return', ax=ax[1,1], s=1, c=df_parameters_sell_down_cluster.color)
	df_parameters_sell_down_cluster.plot.scatter(x='Sell Fraction Portfolio', y='Final Return', ax=ax[1,2], s=1, c=df_parameters_sell_down_cluster.color)
	df_parameters_sell_down_cluster.plot.scatter(x='Buy Wait Days', y='Final Return', ax=ax[1,3], s=1, c=df_parameters_sell_down_cluster.color)
	df_parameters_sell_down_cluster.plot.scatter(x='Buy Fraction Change', y='Final Return', ax=ax[1,4], s=1, c=df_parameters_sell_down_cluster.color)
	for n in range(5):
		ax[1,n].axhline(y=final_original_balance, c='gray', linewidth=1, ls='--')

	ax[0,0].set_ylabel('Final Value, $')
	ax[0,0].set_title('Sell Up, Buy Down: {:.1f}% Winning'.format(pcnt_sell_up_best))
	for n in range(1,5):
		ax[0,n].axes.yaxis.set_visible(False)

	ax[1,0].set_ylabel('Final Value, $')
	ax[1,0].set_title('Sell Down, Buy Up: {:.1f}% Winning'.format(pcnt_sell_down_best))
	for n in range(1,5):
		ax[1,n].axes.yaxis.set_visible(False)

	plt.subplots_adjust(hspace=0.5)
	plt.savefig('trajs_params_'+str(n_clusters)+'_clusters.png', bbox_inches='tight')
	plt.show()


	# Get the best trajectories:
	df_traj_sell_up_best = df_traj_sell_up.loc[:, (df_traj_sell_up.iloc[-1,:] > final_original_balance)]
	df_traj_sell_up_worst = df_traj_sell_up.loc[:, (df_traj_sell_up.iloc[-1,:] < final_original_balance)]
	print('df_traj_sell_up_best:\n', df_traj_sell_up_best)
	print('df_traj_sell_up_worst:\n', df_traj_sell_up_worst)

	df_traj_sell_down_best = df_traj_sell_down.loc[:, (df_traj_sell_down.iloc[-1,:] > final_original_balance)]
	df_traj_sell_down_worst = df_traj_sell_down.loc[:, (df_traj_sell_down.iloc[-1,:] < final_original_balance)]
	print('df_traj_sell_down_best:\n', df_traj_sell_down_best)
	print('df_traj_sell_down_worst:\n', df_traj_sell_down_worst)

	'''
	# kNN to find nearest neighbor strategies:
	from sklearn.neighbors import kNeighborsClassifier
	df_parameters_sell_up_kNN = df_parameters_sell_up.dropna()
	X = df_parameters_sell_up_kNN.drop(columns='Final Return')
	y = df_parameters_sell_up_kNN.loc[:, 'Final Return']
	cols_to_use = df_parameters_sell_up_kNN.columns.values.tolist()[:-1] # Dropping the Final Return column
	knn = kNeighborsClassifier(n_neighbors=5)
	knn.fit(X, y)
	'''




	# Note: Trajectory columns are already sorted in descending order of final balance, so 0:50 gives the 51 best strategies:
	# df_traj_sell_up.iloc[:,0:50].plot(use_index=True, linewidth=0.5)
	# Select the best trajs from the df_params_sell_up/down_best 'Strategy' index to then select the trajectory columns from
	# df_traj_sell_up/down dataframes.
	
	# Plotting both in subplots, isn't working:
	# fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))
	# df_traj_sell_up.iloc[0:10].plot(use_index=True, linewidth=0.5, ax=ax1)
	# df_traj_sell_up.loc[:,'Original Balance'].plot(use_index=True, ax=ax1)
	# ax1.set_title('Sell Up Best Trajectories')

	# df_traj_sell_down.iloc[0:10].plot(use_index=True, linewidth=0.5, ax=ax2)
	# df_traj_sell_down.loc[:,'Original Balance'].plot(use_index=True, ax=ax2)
	# ax2.set_title('Sell Down Best Trajectories')

	# plt.savefig('trajs_sell_up_and_down_best.png', bbox_inches='tight')
	# plt.show()

	# Plotting trajs on separate plots:
	# df_traj_sell_up.iloc[0:10].plot(use_index=True)
	# df_traj_sell_up.loc[:,'Original Balance'].plot(use_index=True)
	# plt.title('Sell Up Best Trajectories')
	# plt.savefig('trajs_sell_up_best.png', bbox_inches='tight')
	# plt.show()

	# df_traj_sell_down.iloc[0:10].plot(use_index=True, linewidth=0.5)
	# df_traj_sell_down.loc[:,'Original Balance'].plot(use_index=True)
	# plt.title('Sell Down Best Trajectories')
	# plt.savefig('trajs_sell_down_best.png', bbox_inches='tight')
	# plt.show()

	'''
	CONCLUSIONS FROM THE PLOTS:

	Sell Up Buy Down
	There is an ideal Sell Inc Pct ranging from 0.1 to 0.32 where all of the strategies that beat the original balance
	by a wide margin existed. Nonetheless, many strategies in this region also did poorly, but above 0.25 the ratio of
	winning to losing strategies was higher than below 0.25. Sell Wait Days showed only a slight preference for 50 to 90
	days, but many winning strategies were outside of this. Buy Wait Days showed no clear correlation. Buy Dec Pct was
	the same except for a slight preference for < -0.15, with many winning strategies outside of this.
	
	Generally, higher Sell Wait Days resulted in a denser clustering of strategies around the original balance's final
	return of $130k. I attribute this to the fact that longer wait days before selling provide a greater number of viable
	strategies, with fewer viable strategies existing for shorter wait days (e.g. it's easier for a 50% increase to occur
	in 120 days than 20 days).
	
	One notable is the dearth of strategies above a Sell Inc Pct of 0.25. The explanation for this is that the market has
	a tendency to move upward slowly over time and then decline rapidly during corrections. Thus, there are very few large
	increases in market value over any time period relative to the number of large declines.

	Percentage of portfolio sold amplifies losses and gains as it is increased.


	Sell Down Buy Up
	This strategy showed few winning strategies that exceeded the original balance's final return of $130k by $10k. This
	is counterintuitive because while the idea of this strategy is to get close to the 'sell high and buy low' strategy,
	it fails relative to the 'sell up buy down' strategy. On closer inspection, this may be the result of what the
	sell up buy down strategy ultimately does: keeps money in the stock market for longer. Alternatively, it may be the
	result of a few key transactions made in the 80's just before the drop in '87, but that begs the question of
	why the 'sell down buy up' strategy doesn't equally capitalize on that time period.

	'''


	return



def DJI_characterization(stock_index, year):
	dji_file = stock_index + str(year) + '.csv'
	df_dji = pd.read_csv(dji_file, header=0, names=['Date','Close'])
	df_dji.set_index('Date', inplace=True)
	# df_dji = df_dji.loc['1950-01-01':, :]
	print('df_dji:\n', df_dji)

	# Make Day column and Pct Change column
	n_days = df_dji.shape[0]
	print('Number of days =', n_days)
	# df_dji['Day'] = range(0, n_days)

	# step selection:
	# 1 day steps is highest resolution. Can be made larger to run code faster for test cases.
	# 3 day steps is the minimum my computer can process for the 1914 dataset before it runs out of RAM
	# trying to store the massive df_dji dataframe.

	'''
	df_dji_pct_change_partial = pd.DataFrame(data=[], index=df_dji.index)
	n_segments = 10
	interval = round(n_days*1/n_segments) # Split calculating into n_segments to lessen the RAM requirement
	start = 1
	stop = interval+1
	step = 1
	for i in range(1, n_segments+1):
		print('i = ', i)
		print('start = {}, stop = {}, step = {}'.format(start, stop, step))
		period_range = list(range(start,stop,step))
		for period_length in period_range: # Periods of 2 to 100 days
			# col = 'Pct Change '+str(n_periods)+' Days'
			df_dji_pct_change_partial[str(period_length)] = df_dji['Close'].pct_change(periods=period_length)*100 # Calculate percent change for all periods

		# Change type to save space:
		# print('**************************************************')
		# print(df_dji_pct_change_partial)
		# print(df_dji_pct_change_partial.dtypes)

		# print('df_dji_pct_change_partial:\n', df_dji_pct_change_partial)
		# print('df_dji_pct_change_partial memory usage:\n', df_dji_pct_change_partial.memory_usage())

		df_dji_pct_change_partial = df_dji_pct_change_partial.astype('float32')
		df_dji_pct_change_partial.columns = pd.to_numeric(df_dji_pct_change_partial.columns)
		df_dji_pct_change_partial.columns = df_dji_pct_change_partial.columns.astype('int16')

		print('df_dji_pct_change_partial info:\n', df_dji_pct_change_partial.info())
		return

		# print('df_dji_pct_change_partial:\n', df_dji_pct_change_partial)
		# print('df_dji_pct_change_partial memory usage:\n', df_dji_pct_change_partial.memory_usage())

		df_dji_pct_change_partial.to_pickle('df_dji_pct_change_partial_'+str(i)+'.pkl')
		df_dji_pct_change_partial = pd.DataFrame() # Drop all data from partial

		start += interval
		stop += interval
	
	# Reading in partial files:
	print('Reading in partial files...')
	df_1 = pd.read_pickle('df_dji_pct_change_partial_1.pkl')
	df_2 = pd.read_pickle('df_dji_pct_change_partial_2.pkl')
	df_3 = pd.read_pickle('df_dji_pct_change_partial_3.pkl')
	df_4 = pd.read_pickle('df_dji_pct_change_partial_4.pkl')
	df_5 = pd.read_pickle('df_dji_pct_change_partial_5.pkl')
	df_6 = pd.read_pickle('df_dji_pct_change_partial_6.pkl')
	df_7 = pd.read_pickle('df_dji_pct_change_partial_7.pkl')
	df_8 = pd.read_pickle('df_dji_pct_change_partial_8.pkl')
	df_9 = pd.read_pickle('df_dji_pct_change_partial_9.pkl')
	df_10 = pd.read_pickle('df_dji_pct_change_partial_10.pkl')
	print(df_10.dtypes)
	print(df_10.memory_usage())

	print('Concatenating partial files...')
	df = pd.concat((df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10), axis=1)
	# print('df:\n', df)
	df.to_pickle('DJI'+str(year)+'_pct_change.pkl')
	df_dji_pct_change = pd.read_pickle('DJI'+str(year)+'_pct_change.pkl')
	df_dji_pct_change.reset_index(inplace=True)
	print('df_dji_pct_change:\n', df_dji_pct_change)
	'''

	df_dji_pct_change = pd.DataFrame(data=[], index=df_dji.index)
	# step = 10
	periods_list = list(range(1,100,1)) + list(range(100,1000,2)) + list(range(1000,10000,10)) + list(range(10000,df_dji.shape[0],500))
	# for period_length in range(1, df_dji.shape[0], step):
	for period_length in periods_list:
		df_dji_pct_change[str(period_length)] = df_dji['Close'].pct_change(periods=period_length)*100 # Calculate percent change for all periods
	print('df_dji_pct_change:\n', df_dji_pct_change)
	print('df_dji_pct_change info:\n', df_dji_pct_change.info())

	value_cols = df_dji_pct_change.iloc[:, 1:-1].columns # Starting from the first column, pick every Xth column, no need to plot every column
	print('value_cols:\n', value_cols)
	df_dji_pct_change_long = df_dji_pct_change.melt(value_vars=value_cols, var_name='Period', value_name='Pct Change')
	df_dji_pct_change_long.dropna(inplace=True)
	df_dji_pct_change_long.Period = df_dji_pct_change_long.Period.astype('int')
	print('df_dji_pct_change_long:\n', df_dji_pct_change_long.dtypes)
	# print('df_dji_pct_change_long.loc[ df_dji_pct_change_long.Period == 22744, :]:\n', df_dji_pct_change_long.loc[ (df_dji_pct_change_long.Period == 22744), : ])

	df_dji_pct_change_long.to_pickle('DJI'+str(year)+'_pct_change_long.pkl') # Might not be useful, just backing it up

	# Getting min and max for each period so that when running simulations, the selection
	# of Sell Fraction Change and Buy Fraction Change are based on a randomized window that falls
	# between the min and max for each number of Wait Days. Right now I'm running simulations in which
	# I randomly select Wait Days and Fraction Change between two arbitrary values that may not make
	# sense. If Sell Wait Days is 1 day, it makes no sense to select a Sell Fraction Change of 60%
	# if that hasn't happened in the history of the market. Such a simulation will fail because it
	# will never trigger a trade, resulting in wasted computation time. This should allow me to avoid
	# all failed strategies.

	# Aggregate
	df_dji_pct_change_agg = df_dji_pct_change_long.groupby('Period')['Pct Change'].agg(Min='min', Max='max', Mean='mean')#, Mean='mean', Std='std')
	print('df_dji_pct_change_agg:\n', df_dji_pct_change_agg.sort_index())
	df_dji_pct_change_agg.to_pickle('DJI'+str(year)+'_pct_change_agg.pkl') # Might not be useful, just backing it up
	# Plotting the aggregated data:
	df_dji_pct_change_agg.plot()
	plt.xlabel('Period')
	plt.ylabel('Percent Change')
	plt.show()

	# Making a lookup table on the basis of the aggregated data in df_dji_pct_change_agg.
	# Period (e.g. Wait Days) is the index, Pct Change (e.g. Fraction Change) is column index.
	# In sell_up() and sell_down(), these values can be picked from the lookup table to use
	# as simulation rules for running complete simulations rather than generating percent change values on the
	# spot from distributions that are not within the min-max spread of percent changes of the DJI.
	# Using a lookup table should avoid having any simulations fail.
	# Create lookup table:
	df_dji_pct_change_lookup = df_dji_pct_change_agg.apply(lambda x: np.random.uniform(x.Min, x.Max, size=1), axis=1)
	print('df_dji_pct_change_lookup:\n', df_dji_pct_change_lookup)
	# print('Looking up the third value of the period 101:\n', df_dji_pct_change_lookup.loc[101][1])
	# df_dji_pct_change_lookup.to_pickle('DJI'+str(year)+'_pct_change_lookup.pkl') # This file I need for sell up and sell down

	# Putting it into a long format with Period and Pct Change columns for easy iteration in sell_up() and sell_down()
	# i is period (e.g. maximum wait days)
	# j is an arbitrary index value 1 to 10 (or whatever you choose in dji_characterization)
	# pct_change is the data in df_lookup
	# i*j is the number of simulations
	# I iterate through the df_lookup dataframe and run a simulation for each value.
	# This is in contrast to the previous method of generating random percent_change and
	# sell_wait_days values at the beginning of simulations. By using a lookup table I
	# might save a little bit of time but the main benefit is that I have a set of percent
	# change values that are realistic for each period.

	n_elements = len(df_dji_pct_change_lookup.to_list()[0])
	col_names = list(range(1,n_elements+1))
	df_dji_pct_change_lookup = pd.DataFrame(data=df_dji_pct_change_lookup.to_list(), index=df_dji_pct_change_lookup.index, columns=col_names)

	i, j = df_dji_pct_change_lookup.shape
	total_sims_to_run = i*j

	df_dji_pct_change_lookup_long = pd.melt(df_dji_pct_change_lookup, value_vars=col_names, value_name='Pct Change', ignore_index=False)
	df_dji_pct_change_lookup_long.reset_index(inplace=True)
	df_dji_pct_change_lookup_long.drop(columns='variable', inplace=True)
	df_dji_pct_change_lookup_long.sort_values(by='Period', ascending=True, inplace=True)
	df_dji_pct_change_lookup_long.reset_index(drop=True, inplace=True)
	print('df_dji_pct_change_lookup_long:\n', df_dji_pct_change_lookup_long)
	print('df_dji_pct_change_lookup_long.iloc[0:20]:\n', df_dji_pct_change_lookup_long.iloc[0:20])
	df_dji_pct_change_lookup_long.to_pickle('DJI'+str(year)+'_pct_change_lookup_long.pkl') # This file I need for sell up and sell down

	# Ending here because I have no need to plot. I just need the lookup table for sell up and sell down.
	return
	# Plot Percent Change vs Period only if the dataset is small. The full 1914 DJI has
	# too much data to plot for my 16 GB of RAM.
	if year != 1914:
		fig, ax = plt.subplots(figsize=(6,5.2))
		g = sns.lineplot(data=df_dji_pct_change_long, x='Period', y='Pct Change',  ci='sd', ax=ax)
		n_xticks_desired = 10
		xticks = range(0, round(10000/step), round(10000/step/n_xticks_desired))
		g.set_xticks(xticks)
		g.set_xticklabels(range(0,10000, round(10000/n_xticks_desired)))#round(8792/n_xticks)))
		g.set_ylabel('Percent Change')
		g.set_xlabel('Period, days')
		plt.savefig('DJI_Characterization.png', bbox_inches='tight')
		plt.show()


		# Histograms of all percent changes for each period in days:
		unique_periods = df_dji_pct_change_long.Period.unique()[::45]
		print('unique_periods:\n', unique_periods)
		df_dji_long_sample = df_dji_pct_change_long.loc[ df_dji_pct_change_long.Period.isin(unique_periods), : ]
		print('df_dji_long_sample:\n', df_dji_long_sample)
		sns.histplot(data=df_dji_long_sample, x='Pct Change', hue='Period', element='step', fill=False)
		plt.yscale('log')
		plt.savefig('DJI_Histo_Percent_Change.png', bbox_inches='tight')
		plt.show()

		# Histograms of all percent changes for each period in days:
		unique_periods = df_dji_pct_change_long.Period.unique()[::45]
		print('unique_periods:\n', unique_periods)
		df_dji_long_sample = df_dji_pct_change_long.loc[ df_dji_pct_change_long.Period.isin(unique_periods), : ]
		print('df_dji_long_sample:\n', df_dji_long_sample)
		sns.kdeplot(data=df_dji_long_sample, x='Pct Change', hue='Period', fill=False)
		plt.yscale('log')
		plt.ylim(10**-5, 10**-1)
		plt.savefig('DJI_KDE_Percent_Change.png', bbox_inches='tight')
		plt.show()

	return


# Use this to create new investment strategies:
# https://seekingalpha.com/article/4336155-sentiment-data-supports-bearish-case?mod=mw_quote_news




''' ------ Spending Habits Analysis ------ '''
# spending()
''' -------------------------------------- '''


''' ------ Net Savings at Retirement ----- '''
# start_date =  '2019-10-19'
# end_date = '2039-11-25'
# current_age = 35
# retirement_age = 55
# goal_net_savings = 1e6
# monthly_contribution = 1e3 # Monthly contribution
# P_init = 3e3  # principal
# r = 0.04 # annual rate of return
# n = 1 # number of times money is compounded per month
# t = 1/12 # compounding frequency per year
# net_savings_estimator(start_date, end_date, current_age, retirement_age, goal_net_savings, monthly_contribution, P_init, r, n, t)
# # Investing from '2018-05-05' to '2019-02-27', then stopping all new investments, produces a net savings of $151,548 by age 55
# # 
# # Saving $1000/month in equities from 35 to 55 produces a net savings of $240,000 assuming no inflation and a 4% annual rate of retun
''' -------------------------------------- '''


''' ------ Monte Carlo Simulation ------ '''
# start_date = '1985-01-29'
# end_date = '2019-12-31'
# num_years = 20
# starting_balance = 6000
# monthly_income = 2000
# pcnt_income_invested_avg = 0.5
# pcnt_income_invested_std = 0.01
# income_invested_dist = 'normal'
# rate_avg = 0.04 	# annual rate of return
# rate_std = 0.12
# rate_dist = 'normal'
# n = 1 				# number of times money is compounded per month
# t = 1/12 			# compounding frequency per year
# num_sims = 100
# mc(start_date, end_date, num_years, starting_balance, monthly_income, pcnt_income_invested_avg, pcnt_income_invested_std, income_invested_dist, rate_avg, rate_std, rate_dist, n, t, num_sims)
''' ------------------------------------ '''


''' ------ Dollar Cost Averaging Simulation ------ '''
# starting_balance = 6000
# monthly_income = 2000
# dca(starting_balance, monthly_income)
''' ------------------------------------ '''


''' ------ Sell Up - Monte Carlo for Trading Stocks ------- '''
# stock_index = 'DJI'
# year = 1914
# strategy = 'sell_down' # 'sell_up' or 'sell_down'
# total_sims_to_run = 50
# sell_up(stock_index, year, strategy, total_sims_to_run)
''' ------------------------------------------------------- '''


''' ------ Sell Up 2 - Monte Carlo for Trading Stocks ------- '''
# stock_index = 'DJI'
# year = 1914
# sell_up_2(stock_index, year)
''' --------------------------------------------------------- '''


# ''' ------ Sell Down - Monte Carlo for Trading Stocks ------ '''
# # stock_index = 'DJI'
# # year = 1914
# # total_sims_to_run = 50
# # sell_down(stock_index, year, total_sims_to_run)
# ''' -------------------------------------------------------- '''


''' ------ Import Simulations ------ '''
sell_up_simulations = 50
sell_down_simulations = 50
rolling_window = 10
import_trajs(sell_up_simulations, sell_down_simulations, rolling_window)
''' -------------------------------- '''


''' ------ DJI Characterization ------ '''
# stock_index = 'DJI'
# year = 1914
# DJI_characterization(stock_index, year)
''' ---------------------------------- '''