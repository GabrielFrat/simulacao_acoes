import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from pandas_datareader import data as pdr
from numpy import linalg as la
import yfinance as yf



# Coleta de dados históricos do yfinance
listAcoes = ['ITUB4', 'KLBN11', 'TRPL4', 'PETR4', 'VALE3', 'TAEE11']
listAcoes = [acoes + '.SA' for acoes in listAcoes]
print(listAcoes)

data_f = datetime.datetime.now()
data_i = data_f - datetime.timedelta(days=8000)


precos = yf.download(listAcoes, start=data_i, end=data_f)['Adj Close']
print(precos)
retornos = precos.pct_change().dropna()

# Distribuição da carteira ajustada pela volatilidade de cada ativo
vol = retornos.std()
weights = 1 / vol
weights /= weights.sum()
print("Volatilidade")
print(weights) 

# Visualização do valor financeiro de cada ativo
alocacao = weights * 10000
print(f'Alocação total de: R$ {alocacao}')

media_r = retornos.mean()
matriz_covariancia = retornos.cov()
pesos_carteira = np.full(len(listAcoes), weights)
numero_acoes = len(listAcoes)

print(pesos_carteira)
print(media_r)

numero_simulacoes = 10000
dias_projetados = 252 * 3
capital_inicial = 10000


retorno_m = retornos.mean(axis=0).to_numpy()
matriz_r_m = retorno_m * np.ones(shape=(dias_projetados, numero_acoes))

L = la.cholesky(matriz_covariancia)
print(L)

retornos_carteira = np.zeros([dias_projetados, numero_simulacoes])
montante_final = np.zeros(numero_simulacoes)

for s in range(numero_simulacoes):
    Rdpf = np.random.normal(size=(dias_projetados, numero_acoes))
    retornos_sinteticos = matriz_r_m + np.inner(Rdpf, L)
    retornos_carteira[:, s] = np.cumprod(np.inner(pesos_carteira, retornos_sinteticos) + 1) * capital_inicial
    montante_final[s] = retornos_carteira[-1, s]

print(retornos_carteira)

plt.plot(retornos_carteira, linewidth=1)
plt.ylabel('Dinheiro')
plt.xlabel('Dias')
plt.savefig('gráfico_cenarios.png', format='png')
plt.show()

print(montante_final)

montante_99 = str(np.percentile(montante_final, 1))
montante_95 = str(np.percentile(montante_final, 5))
montante_mediano = str(np.percentile(montante_final, 50))
cenarios_com_lucro = str((len(montante_final[montante_final > capital_inicial]) / len(montante_final)) * 100) + "%"

print(f''''
      Ao investir R$ {capital_inicial}, podemos esperar esses resultados para
      os próximos 3 anos, utilizando o método de Monte Carlo com 10 mil simulações:

      Com 50% de probabilidade, o montante será maior que R$ {montante_mediano}.
      Com 95% de probabilidade, o montante será maior que R$ {montante_95}.
      Com 99% de probabilidade, o montante será maior que R$ {montante_99}.

      {cenarios_com_lucro} dos cenários, foi possível obter lucro nos próximos 3 anos.
      
      ''')