#Manipulação de dados
import pandas as pd

#Cálculos Vetorias
import numpy as np

#Visualização de Dados
import seaborn as sns
from matplotlib import pyplot as plt

# Leitura e acesso de Banco de dados
from google.cloud import bigquery
from google.colab import auth
from google.oauth2 import service_account

# Modelo de previsão
from prophet import Prophet


# Acesso e autenticação ao Banco de dados com credencial
auth.authenticate_user()
cred = service_account.Credentials.from_service_account_file('/content/drive/MyDrive/DMV/json/dmvadventures-bc4886b96019.json') #Precisa montar o drive antes no menu lateral
print('Authenticated')

re = pd.read_excel('/content/re - relatorio.xlsx', sheet_name='Mensal')
re_det = pd.read_excel('/content/re - relatorio.xlsx', sheet_name='Detalhamento')

re['Mês']=pd.to_datetime(re['Mês'], format='%m/%Y')

re.drop('Num_mes', axis=1, inplace=True)

re_pre=re[(re['Anos']==2024) & (re['Mês'].dt.month != 7)]

re_pre.describe()

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.barplot(data=re_pre, x='Mês',y='investimento')
ax.axhline(y=re_pre['investimento'].mean(), color='r', linestyle='--',label='Média de Investimento')
plt.title('Investimento por mes')
plt.xlabel('Mês')
plt.ylabel('Investimento em R$')
plt.legend()
#

plt.show()

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.barplot(data=re_pre, x='Mês',y='Impressões')
ax.axhline(y=re_pre['Impressões'].mean(), color='r', linestyle='--',label='Média de Impressões')
plt.title('Impressões por mes')
plt.xlabel('Mês')
plt.ylabel('Impressões')
plt.legend()
#

plt.show()

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.barplot(data=re_pre, x='Mês',y='Cliques')
ax.axhline(y=re_pre['Cliques'].mean(), color='r', linestyle='--',label='Cliques medio')
plt.title('Cliques por mes')
plt.xlabel('Mês')
plt.ylabel('Cliques')
plt.legend()
#

plt.show()

re_pre['ctr'] = (re_pre['Cliques']/re_pre['Impressões'])*100

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.barplot(data=re_pre, x='Mês',y='ctr')
ax.axhline(y=re_pre['ctr'].mean(), color='r', linestyle='--',label='ctr medio')
plt.title('CTR por mes')
plt.xlabel('Mês')
plt.ylabel('CTR ( % )')
plt.legend()
#

plt.show()

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.barplot(data=re_pre, x='Mês',y='Visitas')
ax.axhline(y=re_pre['Visitas'].mean(), color='r', linestyle='--',label='Média de Visitas')
plt.title('Visitas por mes')
plt.xlabel('Mês')
plt.ylabel('Visitas')
plt.legend()
#

plt.show()

re_pre['CR'] = (re_pre['Visitas']/re_pre['Cliques'])*100

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.barplot(data=re_pre, x='Mês',y='CR')
ax.axhline(y=re_pre['CR'].mean(), color='r', linestyle='--',label='CR medio')
plt.title('Connect Rate por mês')
plt.xlabel('Mês')
plt.ylabel('CR ( % )')
plt.legend()
#

plt.show()

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.barplot(data=re_pre, x='Mês',y='Leads')
ax.axhline(y=re_pre['Leads'].mean(), color='r', linestyle='--',label='Média de Leads')
plt.title('Leads por mes')
plt.xlabel('Mês')
plt.ylabel('Leads')
plt.legend()
#

plt.show()

client = bigquery.Client(project='dmvadventures', credentials=cred)

simple_count = 100_000
row_count = client.query('''
   SELECT
     COUNT(*) as total
     FROM  `dmvadventures.gold_reenergisa.vw_consolida_ads`''').to_dataframe().total[0]

# A partir daqui a linguagem segue lógica SQL

df_gads = client.query('''
   SELECT
   date_ads, campaign_name,SUM(cost) as cost,SUM(impressions) as impressions,SUM(clicks) as clicks,SUM(sessions) as sessions,SUM(conversion_lead_gd) as leads
   FROM
   `dmvadventures.gold_reenergisa.vw_consolida_ads`
   WHERE RAND() < %d%d
   AND
   data_source = 'google_ads'    #base exclusiva de GADS
   AND
   date_ads > '2023-12-31'
   AND
   date_ads < '2024-07-01'
   AND
   impressions > 0          #Evitando dados Nan de impressoes
   AND
   cost > 0        #Evitando dados Nan de Investimento
   AND
   frente = "GD"

   GROUP BY
   1,2
  #  ORDER BY
  #  impressions
   '''% (simple_count, row_count)).to_dataframe()



print(f'O dataset possui {row_count} linhas')

df_meta = client.query('''
   SELECT
   date_ads, campaign_name,SUM(cost) as cost,SUM(impressions) as impressions,SUM(clicks) as clicks,SUM(sessions) as sessions,SUM(conversion_lead_gd) as leads, SUM(on_facebook_leads) as lead_ads
   FROM
   `dmvadventures.gold_reenergisa.vw_consolida_ads`
   WHERE RAND() < %d%d
   AND
   data_source = 'meta_ads'    #base exclusiva de META
   AND
   date_ads > '2023-12-31'
   AND
   date_ads < '2024-07-01'
   AND
   impressions > 0          #Evitando dados Nan de impressoes
   AND
   cost > 0        #Evitando dados Nan de Investimento
   AND
   frente = "GD"

   GROUP BY
   1,2
  #  ORDER BY
  #  impressions
   '''% (simple_count, row_count)).to_dataframe()



print(f'O dataset possui {row_count} linhas')

df_gads.info()

df_gads.isnull().sum()

df_gads.describe()

#Alterando o formato data e criando uma coluna mês para facilitar analise
df_gads['date_ads'] = pd.to_datetime(df_gads['date_ads'])
df_gads['mes'] = df_gads['date_ads'].dt.month

#Criando a coluna calculada CTR
df_gads['ctr'] = df_gads['clicks']/df_gads['impressions']

#Criando um df agrupado por Campanha
df_gads_cmp = df_gads.groupby(['campaign_name'])[['impressions','clicks','cost','leads']].sum()

#Criando a coluna CTR direto no df de linha criativa para evitar trabalhar com a média do CTR das linhas
df_gads_cmp['ctr'] = df_gads_cmp['clicks'] / df_gads_cmp['impressions']

df_gads_cmp


ax = sns.FacetGrid(df_gads, col='campaign_name',col_wrap=2,aspect=5,sharex=False,)
ax.map(sns.scatterplot,'date_ads','leads')
ax.fig.tight_layout(w_pad=1)

df_gads['cpm'] = df_gads['cost']/df_gads['impressions']*1000
df_gads['cpc'] = df_gads['cost']/df_gads['clicks']
df_gads['cpl'] = df_gads['cost']/df_gads['leads']

ax = sns.FacetGrid(df_gads, col='campaign_name',col_wrap=2,aspect=5,sharex=False,)
ax.map(sns.scatterplot,'date_ads','cpm')
ax.fig.tight_layout(w_pad=1)

ax = sns.FacetGrid(df_gads, col='campaign_name',col_wrap=2,aspect=5,sharex=False,)
ax.map(sns.scatterplot,'date_ads','cpc')
ax.fig.tight_layout(w_pad=1)

ax = sns.FacetGrid(df_gads, col='campaign_name',col_wrap=2,aspect=5,sharex=False,)
ax.map(sns.lineplot,'date_ads','cpl')
ax.fig.tight_layout(w_pad=1)

df_meta

df_meta['date_ads'] = pd.to_datetime(df_meta['date_ads'])
df_meta['mes'] = df_meta['date_ads'].dt.month

df_meta['ctr'] = df_meta['clicks']/df_meta['impressions']
df_meta['lead_total'] = df_meta['leads'] + df_meta['lead_ads']

df_meta_camp = df_meta.groupby(['campaign_name'])[['impressions','clicks','cost','leads','lead_total']].sum()

df_meta_camp['ctr'] = df_meta_camp['clicks'] / df_meta_camp['impressions']

df_meta_camp

ax = sns.FacetGrid(df_meta, col='campaign_name',col_wrap=3,aspect=5,sharex=False,)
ax.map(sns.scatterplot,'date_ads','lead_total')
ax.fig.tight_layout(w_pad=1)

ax = sns.FacetGrid(df_meta, col='campaign_name',col_wrap=3,aspect=5,sharex=False,)
ax.map(sns.scatterplot,'date_ads','impressions')
ax.fig.tight_layout(w_pad=1)

ax = sns.FacetGrid(df_meta, col='campaign_name',col_wrap=3,aspect=5,sharex=False,)
ax.map(sns.scatterplot,'date_ads','clicks')
ax.fig.tight_layout(w_pad=1)

df_meta['cpm'] = df_meta['cost']/df_meta['impressions']*1000
df_meta['cpc'] = df_meta['cost']/df_meta['clicks']
df_meta['cpl'] = df_meta['cost']/df_meta['lead_total']

ax = sns.FacetGrid(df_meta, col='campaign_name',col_wrap=3,aspect=5,sharex=False,)
ax.map(sns.scatterplot,'date_ads','cpl')
ax.fig.tight_layout(w_pad=1)

re

re_2024 = re[re['Anos']==2024]

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.barplot(data=re_2024, x='Mês',y='Leads')
plt.title('Pós-otimizações: Leads por mes')
plt.xlabel('Mês')
plt.ylabel('Leads')
#

plt.show()

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.lineplot(data=re_det, x='Data',y='Lead_total')
plt.title('Leads por dia')
plt.xlabel('Data')
plt.ylabel('Leads')
plt.axvline(x= pd.to_datetime('2024-07-15'), color='black', linestyle="--", label='Mudança Estratégica')
plt.legend()


plt.show()

mean = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['Lead_google_total'].mean()
median = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['Lead_google_total'].median()
mode = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['Lead_google_total'].mode()[0]
max = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['Lead_google_total'].max()
min = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['Lead_google_total'].min()
std = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['Lead_google_total'].std()

#Após Mudança

mean2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['Lead_google_total'].mean()
median2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['Lead_google_total'].median()
mode2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['Lead_google_total'].mode()[0]
max2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['Lead_google_total'].max()
min2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['Lead_google_total'].min()
std2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['Lead_google_total'].std()

#Após 7 dias de aprendizado
mean3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['Lead_google_total'].mean()
median3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['Lead_google_total'].median()
mode3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['Lead_google_total'].mode()[0]
max3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['Lead_google_total'].max()
min3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['Lead_google_total'].min()
std3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['Lead_google_total'].std()


print(f'A média de leads google é : {mean}')
print(f'A mediana de leads google é: {median}')
print(f'A moda de leads google é: {mode}')
print(f'O valor máximo de leads google é: {max}')
print(f'O valor mínimo de leads google é: {min}')
print(f'O desvio padrão de leads google é: {std}')


print('________________________________________________')
print('')
print('APÓS MUDANÇA:')
print(f'A média de leads google é : {mean2}')
print(f'A mediana de leads google é: {median2}')
print(f'A moda de leads google é: {mode2}')
print(f'O valor máximo de leads google é: {max2}')
print(f'O valor mínimo de leads google é: {min2}')
print(f'O desvio padrão de leads google é: {std2}')


print('________________________________________________')
print('')
print('APÓS APRENDIZADO:')
print(f'A média de leads google é : {mean3}')
print(f'A mediana de leads google é: {median3}')
print(f'A moda de leads google é: {mode3}')
print(f'O valor máximo de leads google é: {max3}')
print(f'O valor mínimo de leads google é: {min3}')
print(f'O desvio padrão de leads google é: {std3}')

fig,ax = plt.subplots(figsize=(12,7))

ax = sns.lineplot(data=re_det, x='Data',y='CPL google')
plt.title('CPL por dia')
plt.xlabel('Data')
plt.ylabel('CPL (R$)')
plt.axvline(x= pd.to_datetime('2024-07-15'), color='black', linestyle="--", label='Mudança Estratégica')
plt.legend()


plt.show()

mean = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['CPL google'].mean()
median = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['CPL google'].median()
mode = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['CPL google'].mode()[0]
max = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['CPL google'].max()
min = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['CPL google'].min()
std = re_det[re_det['Data'] < pd.to_datetime('2024-07-15')]['CPL google'].std()

#Após Mudança

mean2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['CPL google'].mean()
median2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['CPL google'].median()
mode2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['CPL google'].mode()[0]
max2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['CPL google'].max()
min2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['CPL google'].min()
std2 = re_det[re_det['Data'] > pd.to_datetime('2024-07-15')]['CPL google'].std()

#Após 7 dias de aprendizado
mean3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['CPL google'].mean()
median3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['CPL google'].median()
mode3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['CPL google'].mode()[0]
max3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['CPL google'].max()
min3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['CPL google'].min()
std3 = re_det[re_det['Data'] > pd.to_datetime('2024-07-22')]['CPL google'].std()


print(f'A média de CPL google é : {mean:.2f}')
print(f'A mediana de CPL google é: {median:.2f}')
print(f'A moda de CPL google é: {mode:.2f}')
print(f'O valor máximo de CPL google é: {max:.2f}')
print(f'O valor mínimo de CPL google é: {min:.2f}')
print(f'O desvio padrão de CPL google é: {std:.2f}')


print('________________________________________________')
print('')
print('APÓS MUDANÇA:')
print(f'A média de CPL google é : {mean2:.2f}')
print(f'A mediana de CPL google é: {median2:.2f}')
print(f'A moda de CPL google é: {mode2:.2f}')
print(f'O valor máximo de CPL google é: {max2:.2f}')
print(f'O valor mínimo de CPL google é: {min2:.2f}')
print(f'O desvio padrão de CPL google é: {std2:.2f}')


print('________________________________________________')
print('')
print('APÓS APRENDIZADO:')
print(f'A média de CPL google é : {mean3:.2f}')
print(f'A mediana de CPL google é: {median3:.2f}')
print(f'A moda de CPL google é: {mode3:.2f}')
print(f'O valor máximo de CPL google é: {max3:.2f}')
print(f'O valor mínimo de CPL google é: {min3:.2f}')
print(f'O desvio padrão de CPL google é: {std3:.2f}')

re_det.dropna(inplace=True)

re_det

df = re_det[['Data','CPL google']]

df.rename(columns={'Data':'ds','CPL google':'y'},inplace=True)

df['y'] = np.log(df['y'])

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

forecast['yhat'] = np.exp(forecast['yhat'])
forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
df['y'] = np.exp(df['y'])

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

plt.figure(figsize=(10, 6))
plt.scatter(df['ds'], df['y'], label='Observado', color='black')
plt.plot(forecast['ds'], forecast['yhat'], label='Previsto')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='gray', alpha=0.2)
plt.xlabel('Data')
plt.ylabel('Valor (Reais)')
plt.title('Previsão e Valores Observados')
plt.axvline(pd.to_datetime('2024-07-15'), color='black', linestyle="--", label='Mudança Estratégica')
plt.legend()
plt.show()


