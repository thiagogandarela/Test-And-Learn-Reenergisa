Crie um novo notebook no https://colab.research.google.com/

Copie os codigos por blocos para rodar no notebook criado acima.

Exemplo:

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

esse código possui 4 blocos, rode copie o primeiro, cole no google collab, rode o codigo, depois copie o proximo, rode e assim por diante.
