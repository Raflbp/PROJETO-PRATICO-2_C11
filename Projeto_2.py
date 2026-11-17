import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima

# Carregando o dataset do IPCA acumulado em 12 meses
# A coluna Date está no formato MM/YYYY, então informamos o formato para o pandas converter corretamente
df = pd.read_csv('IBGE_IPCA.csv', delimiter=',')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%Y')
df = df.set_index('Date').asfreq('MS')  # frequência mensal (início do mês)

# Renomeando a coluna para facilitar o uso
df.columns = ['IPCA Rate (last 12 months)']

# PLOTANDO A SÉRIE TEMPORAL COMPLETA
# Série completa (1994 a 2026)
df['IPCA Rate (last 12 months)'].plot(figsize=(10, 5),
                                       title='IPCA acumulado em 12 meses (1994-2026)',
                                       xlabel='Data', ylabel='IPCA (%)', x_compat=True)
plt.grid(True)
plt.show()


# Recorte do período de análise
inicio = '2000-01-01'
serie = df.loc[inicio:, 'IPCA Rate (last 12 months)']


serie.plot(figsize=(10, 5),
           title='IPCA acumulado em 12 meses (a partir de 2000)',
xlabel='Data', ylabel='IPCA (%)', x_compat=True)
plt.grid(True)
plt.show()

# DECOMPOSIÇÃO DA SÉRIE TEMPORAL
# Decomposição aditiva com período de 12 meses 
decomposicao = seasonal_decompose(serie, model='additive', period=12)
fig = decomposicao.plot()
fig.set_size_inches(10, 8)
plt.show()

from statsmodels.tsa.stattools import adfuller

# Teste ADF (Augmented Dickey-Fuller)
# H0: a série possui raiz unitária (NÃO é estacionária)
resultado = adfuller(serie)

# Inclinação de uma reta ajustada à série inteira (tendência linear global)
x = np.arange(len(serie))
coef_angular = np.polyfit(x, serie.values, 1)[0]

# Tendência extraída pela decomposição, em destaque
plt.figure(figsize=(10, 5))
serie.plot(label='IPCA 12 meses', alpha=0.5)
decomposicao.trend.plot(label='Tendência (média móvel de 12 meses)', color='red')
plt.title('Tendência da série')
plt.xlabel('Data')
plt.ylabel('IPCA (%)')
plt.legend()
plt.grid(True)
plt.show()


# Força e amplitude do componente sazonal
sazonal = decomposicao.seasonal
residuo = decomposicao.resid.dropna()

amplitude_sazonal = sazonal.max() - sazonal.min()
amplitude_serie = serie.max() - serie.min()

# Força da sazonalidade (0 = nenhuma, 1 = muito forte)
forca_sazonal = max(0, 1 - residuo.var() / (residuo + sazonal.loc[residuo.index]).var())

# Padrão sazonal médio por mês
sazonal.groupby(sazonal.index.month).mean().plot(kind='bar', figsize=(8, 4),
                                                 title='Componente sazonal médio por mês',
                                                 xlabel='Mês', ylabel='p.p.')
plt.show()


meses_teste = 36
conjunto_treinamento = serie.iloc[:-meses_teste]
conjunto_teste = serie.iloc[-meses_teste:]


# Versão 1: tendência e sazonalidade aditivas (como no notebook de aula)
model_hw1 = ExponentialSmoothing(endog=conjunto_treinamento, trend='add',
                                 seasonal='add', seasonal_periods=12).fit()
forecasting_hw1 = model_hw1.forecast(steps=len(conjunto_teste))

# Versão 2: tendência amortecida (damped) e sem sazonalidade
model_hw2 = ExponentialSmoothing(endog=conjunto_treinamento, trend='add',
                                 damped_trend=True, seasonal=None).fit()
forecasting_hw2 = model_hw2.forecast(steps=len(conjunto_teste))

# TREINAMENTO (últimos 36 meses separados para teste, sem gráfico)
meses_teste = 36
conjunto_treinamento = serie.iloc[:-meses_teste]
conjunto_teste = serie.iloc[-meses_teste:]

model_hw = ExponentialSmoothing(endog=conjunto_treinamento, trend='add',
                                damped_trend=True, seasonal=None).fit()
forecasting_hw = model_hw.forecast(steps=len(conjunto_teste))

model_arima = auto_arima(y=conjunto_treinamento, seasonal=True, m=12,
                         error_action='ignore', suppress_warnings=True, stepwise=True)
forecasting_arima = pd.Series(model_arima.predict(n_periods=len(conjunto_teste)),
                              index=conjunto_teste.index)

# PREVISÃO FUTURA (24 meses, treinando com todo o histórico)
passos = 24
datas_futuras = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1),
                              periods=passos, freq='MS')

modelo_final_hw = ExponentialSmoothing(endog=serie, trend='add',
                                       damped_trend=True, seasonal=None).fit()
prev_hw = modelo_final_hw.forecast(steps=passos)

modelo_final_arima = auto_arima(y=serie, seasonal=True, m=12,
                                error_action='ignore', suppress_warnings=True, stepwise=True)
prev_arima = pd.Series(np.array(modelo_final_arima.predict(n_periods=passos)),
                       index=datas_futuras)

# GRÁFICO 1: HOLT-WINTERS
plt.figure(figsize=(10, 5))
serie['2018-01-01':].plot(label='IPCA 12 meses (histórico)', color='black')
prev_hw.plot(label='Previsão Holt-Winters (24 meses)', color='red', linestyle='--')
plt.title('Previsão do IPCA - Próximos 24 meses (Holt-Winters)')
plt.xlabel('Data')
plt.ylabel('IPCA (%)')
plt.legend()
plt.grid(True)
plt.show()

# GRÁFICO 2: ARIMA
plt.figure(figsize=(10, 5))
serie['2018-01-01':].plot(label='IPCA 12 meses (histórico)', color='black')
prev_arima.plot(label='Previsão ARIMA (24 meses)', color='green')
plt.title('Previsão do IPCA - Próximos 24 meses (ARIMA)')
plt.xlabel('Data')
plt.ylabel('IPCA (%)')
plt.legend()
plt.grid(True)
plt.show()