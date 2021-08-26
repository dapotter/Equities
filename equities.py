import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.tsa.stattools as smt
import seaborn as sns
import scipy.fftpack
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression


def dji_fft():
	df = pd.read_csv('/home/dp/Documents/Equities/DJI1914.csv')
	df = df.loc[:,['Date','Close']]
	df.Date = pd.to_datetime(df.Date)
	df.set_index('Date', inplace=True)
	print('df:\n', df)

	# Plot the entire Dow Jones 1914 - 2021
	fig, ax = plt.subplots(1,1, figsize=(6,6))
	df.plot(use_index=True, y='Close', ax=ax)
	ax.set_xlabel('Date')
	ax.set_ylabel('Closing Price')
	plt.show()

	# Plot the DJI over yearly, monthly and weekly timescales
	fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(18,4))
	df.plot(use_index=True, y='Close', ax=ax1)
	df.loc['1950-01-01':'2000-01-01', :].plot(use_index=True, y='Close', ax=ax2)
	df.loc['1970-01-01':'1980-01-01', :].plot(use_index=True, y='Close', ax=ax3)
	df.loc['1974-01-01':'1976-01-01', :].plot(use_index=True, y='Close', ax=ax4)
	df.loc['1975-01-01':'1975-02-01', :].plot(use_index=True, y='Close', ax=ax5)
	plt.savefig('DJI_Fractal.png', bbox_inches='tight')
	plt.show()

	close = df['Close']
	close_fft = scipy.fftpack.fft(close)
	close_psd = np.abs(close_fft)**2
	print('close_psd:\n', close_psd)
	fft_freq = scipy.fftpack.fftfreq(len(close_psd), 1/1)
	mask = fft_freq > 0
	x = fft_freq[mask]
	y = 10*np.log10(close_psd[mask])
	# Export power spectrum:
	ps = pd.DataFrame(data={'true_f':x, 'true_p': 10**y, 'p':y})
	ps.to_csv('/home/dp/Documents/Equities/DJIPowerSpectrum.csv')
	ps_every_100 = ps.iloc[::100, :] # Every 100th value since the full dataset slows excel
	ps_every_100.to_csv('/home/dp/Documents/Equities/DJIPowerSpectrum_100DayInterval.csv')

	# Fit linear regression to x and y data
	logx = np.log10(x)
	print('logx shape:', logx.shape)
	logx_reshaped = logx.reshape(-1,1)
	print('logx_reshaped shape:', logx_reshaped.shape)
	lr = LinearRegression()
	lr.fit(logx.reshape(-1,1), y)
	r_sq = lr.score(logx_reshaped, y)
	print('Intercept:', lr.intercept_)
	print('Slope:', lr.coef_)
	print('Coefficient of determination:', r_sq)
	y_pred = lr.predict(logx_reshaped)

	# Plot Power Spectrum with fit:
	fig, ax = plt.subplots(1,1, figsize=(6,6))
	ax.plot(logx, y)
	ax.plot(logx, y_pred, color='red')
	ax.set_ylabel('PSD, db')
	ax.set_xlabel('Frequency, 1/day')
	plt.show()


	def one_over_f(f, alpha, scale):
		return scale / f ** alpha
		# return 1 / f ** alpha

	time_step = 1 / 1
	ps1 = y# Don't use 10**y, doesn't work for some reason.
	freqs1 = x

	# remove the DC bin because 1/f cannot fit at 0 Hz
	ps1 = ps1[freqs1 != 0]
	freqs1 = freqs1[freqs1 != 0]

	# Reducing the fit size but this isn't necessary:
	# ps1 = ps1[100:-1]
	# freqs1 = freqs1[100:-1]

	params, _ = curve_fit(one_over_f, np.abs(freqs1), ps1)
	ps2 = one_over_f(freqs1, *params)
	print('params:\n', params)

	fig, ax = plt.subplots(1,1, figsize=(6,5))
	ax.plot(x, y, color='blue')
	ax.plot(freqs1, ps2, color='red')
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('Frequency, 1/day')
	ax.set_ylabel('Power')
	ax.annotate('alpha=0.08', xy=(0.001, 140), xycoords='data')
	plt.savefig('DJI_PS_with_fit.png', bbox_inches='tight')
	plt.show()

	df['Change'] = df.diff()
	df.sort_values(by='Change', ascending=False, inplace=True)
	print('Price Fluctuations:\n', df.head(20))
	df.sort_values(by='Change', ascending=True, inplace=True)
	print('Price Fluctuations:\n', df.head(20))

	df['Rel Change'] = df['Change'] / df['Close']
	df.sort_values(by='Rel Change', ascending=False, inplace=True)
	print('Relative Price Fluctuations:\n', df.head(20))
	df.sort_values(by='Rel Change', ascending=True, inplace=True)
	print('Relative Price Fluctuations:\n', df.head(20))

	price_fluc = df['Rel Change']
	print('price fluctuations:\n', price_fluc)
	fig, ax = plt.subplots(1,1, figsize=(6,6))
	price_fluc.hist(bins=50, ax=ax)
	ax.set_xlabel('abs(Daily Change), Points')
	ax.set_ylabel('Counts')
	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.show()

	fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=(18,4))
	price_fluc.plot(y='Close', ax=ax1)
	price_fluc.loc['1950-01-01':'2000-01-01'].plot(ax=ax2)
	price_fluc.loc['1970-01-01':'1980-01-01'].plot(ax=ax3)
	price_fluc.loc['1974-01-01':'1976-01-01'].plot(ax=ax4)
	price_fluc.loc['1975-01-01':'1975-02-01'].plot(ax=ax5)
	plt.savefig('DJI_Fluc.png', bbox_inches='tight')
	plt.show()

	# spacing = np.linspace(-5 * np.pi, 5 * np.pi, num=100)
	# s = pd.Series(0.7 * np.random.rand(100) + 0.3 * np.sin(spacing))
	# print('s:\n', s)
	# x = pd.plotting.autocorrelation_plot(s)
	# x.plot()
	# plt.show()

	s = price_fluc.sort_index(ascending=True)
	s.reset_index(drop=True, inplace=True)
	s = s.loc[1:]
	s.plot(use_index=True)
	plt.show()
	print('price_fluc sorted ascending by Date:\n', s)
	x = pd.plotting.autocorrelation_plot(s)
	x.plot()
	plt.savefig('DJI_Diff_ACF.png', bbox_inches='tight')
	plt.show()

	# Random signal generation:
	noise = np.random.normal(0,1,100)
	# Not sure if I want to do this.



	# Plotting change and change relative to the day's closing price.
	# This is needed because as the Dow climbs higher its fluctuations
	# also increase.
	fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))
	df.plot(y='Rel Change', use_index=True, ax=ax1, linewidth=0.2)
	df.plot(y='Change', use_index=True, ax=ax2, linewidth=0.2)
	ax1.set_xlabel('Date')
	ax1.set_ylabel('Relative Change')
	ax2.set_xlabel('Date')
	ax2.set_ylabel('Change')
	plt.show()

	rel_price_fluc = df['Rel Change']
	fig, ax = plt.subplots(1,1, figsize=(6,6))
	rel_price_fluc.hist(bins=200, ax=ax)
	ax.set_xlabel('abs(Daily Change / Close), Points')
	ax.set_ylabel('Counts')
	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.show()


# dji_fft()

# Playing with floating point precision
x = [62888.650808, 62888.650808, 62888.650808]
s = pd.Series(data=x)
y = s.astype('float')
print('y =\n', y)
y = s.astype('float32')
print('y =\n', y)
y = s.astype('float16')
print('y =\n', y)
y = s.round(2)
print('y =\n', y)

print(131.5*10*1E6/(1E9))

values = np.random.uniform(low=-10, high=15, size=1000)
plt.plot(values)
plt.show()
plt.hist(values)
plt.show()


