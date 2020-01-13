#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto: Análise de dados sobre combustíveis no Brasil

Dados:
    A ANP libera reports semanais sobre os preços da gasolina, diesel e outros
    combustíveis utilizados no transporte pelo país. O dataset que será trabalhado
    neste projeto apresenta o preço médio por litro (R$/L), número de postos de
    gasolina analisados e outras informações agrupadas por estados e regiões 
    pelo país

O que pretendemos responder com estes dados:
    - Como os preços variaram entre as diferentes regiões e estados
    - Dentro das regiões, quais estados aumentaram mais os seus preços
    - Quais estados tem os preços mais baixos/altos para os diferentes tipos de
    combustível?
    - Depois da greve dos caminhoneiros, qual foi o real efeito sobre o Diesel?

"""

import util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import operator
sns.set_palette("colorblind") 
sns.set_style("darkgrid")

# Set Pandas display options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Import geospaitial libraries
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Read data

path = '2004-2019.tsv'
df = pd.read_csv(path, sep='\t',parse_dates=True, index_col=1)
df.index = pd.to_datetime(df.index, format='%Y-%m-%d')
df.index = df.index.to_period('M')
df.head()
df.info()

# Remvendo colunas que não serão utilizadas
df.drop(['Unnamed: 0', 'DATA FINAL', 'MÊS', 'ANO'], axis=1, inplace=True)
df.info()

# Ajustando os nomes das colunas
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("'", "")
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace("distribuição", "dist")
df.columns = df.columns.str.replace("revenda", "rev")
df.columns = df.columns.str.replace("variação", "var")
df.columns = df.columns.str.replace("médio", "med")
df.columns = df.columns.str.replace("média", "med")
df.columns = df.columns.str.replace("mínimo", "min")
df.columns = df.columns.str.replace("máximo", "max")
df.columns = df.columns.str.replace("padrão", "pad")
df.columns = df.columns.str.replace("região", "reg")
df.columns = df.columns.str.replace("número", "num")
df.columns = df.columns.str.replace("preço", "preco")


# Substituindo "-" por 0
for column in df.columns:
    if df[column].dtype == 'O' and \
    column not in ('reg','estado','produto','unidade_de_medida'):
        df[column] = df[column].str.replace("-", "0")


# Convertendo as colunas para float
for column in df.columns:
    if df[column].dtype == 'O' and \
    column not in ('reg','estado','produto','unidade_de_medida'):
        df[column].fillna(0).astype(float)
        df[column] = df[column].astype(float)
        
# Renomeando os combustíveis
produtos = {"ÓLEO DIESEL": "DIESEL", "GASOLINA COMUM": "GASOLINA", "GLP":"GLP", 
            "ETANOL HIDRATADO":"ETANOL", "GNV":"GNV", "ÓLEO DIESEL S10":"DIESEL S10"}

df["produto"] = df.produto.map(produtos)

# Renomeando as unidades de medida
unidades = {"R$/l":"liter", "R$/13Kg":"13kg", "R$/m3":"m3"}

df["unidade_de_medida"] = df["unidade_de_medida"].map(unidades)

# Tornando os objetos como categorias
object_cols = df.select_dtypes(include='object').columns
df[object_cols] = df[object_cols].astype('category', inplace=True)

# Categorias de combustíveis
df['categoria'] = df.unidade_de_medida.map({'liter':int(1), 'm3':int(1), '13kg':int(2)})

# Unificando os valores por mes
df['produtorio_rev'] = df.preco_med_rev * df.num_de_postos_pesquisados
df['produtorio_dist'] = df.preco_med_dist * df.num_de_postos_pesquisados

agregacoes = {
        'num_de_postos_pesquisados':'sum',
        'produtorio_rev': 'sum',
        'preco_min_rev': 'min',
        'preco_max_rev': 'max',
        'produtorio_dist': 'sum',
        'preco_min_dist': 'min',
        'preco_max_dist': 'max'
        }
valores = [
        'num_de_postos_pesquisados',
        'produtorio_rev',
        'preco_min_rev',
        'preco_max_rev',
        'produtorio_dist',
        'preco_min_dist',
        'preco_max_dist'
        ]



            
"""
Diferença entre groupby e pivot_table:
    - O groupby faz todas as combinações dos índices
    - A pivot_table apenas as combinações que existem na tabela
    
Exemplos:
# Group by
teste1 = df.groupby(['estado',"produto"]).agg({"preco_med_rev":[min,max,sum,"mean"]})
teste1.reset_index(level=['estado','produto'], inplace=True)

#Pivot Table
teste2 = df.pivot_table(values='preco_med_rev', 
                        index=['estado','produto', 'DATA INICIAL'], 
                        aggfunc=[min, max, sum, np.mean])


    
"""
# Unificando o preço mensal

df_novo = df.pivot_table(values=valores, 
                        index=['reg', 'estado','produto', 'DATA INICIAL','unidade_de_medida'], 
                        aggfunc=agregacoes)

df_novo.reset_index(level=['reg', 'estado', 'produto', 'DATA INICIAL', 'unidade_de_medida'], inplace=True)

df_novo['preco_med_rev'] = df_novo.produtorio_rev/df_novo.num_de_postos_pesquisados
df_novo['preco_med_dist'] = df_novo.produtorio_dist/df_novo.num_de_postos_pesquisados

df_novo.drop(['produtorio_rev','produtorio_rev'],axis=1, inplace=True)
df_novo.rename(columns={'DATA INICIAL':'data'}, inplace=True)
df_novo['ano'] = df_novo.data.dt.year
df_novo = util.filtro(df_novo,'ano', '>=', '2012')

# Normalizando sobre o preco de todos os combustíveis
df_novo['preco_med_norm_tot'] = 0
for i in range(len(df_novo.preco_med_rev)):
    if df_novo.produto.iloc[i] == 'GLP':
        df_novo['preco_med_norm_tot'].iloc[i] = (df_novo.preco_med_rev.iloc[i] - util.filtro(df_novo,'produto', '==', 'GLP').preco_med_rev.min())/(util.filtro(df_novo,'produto', '==', 'GLP').preco_med_rev.max() - util.filtro(df_novo,'produto', '==', 'GLP').preco_med_rev.min())
    else:
        df_novo['preco_med_norm_tot'].iloc[i] = (df_novo.preco_med_rev.iloc[i] - util.filtro(df_novo,'produto', '!=', 'GLP').preco_med_rev.min())/(util.filtro(df_novo,'produto', '!=', 'GLP').preco_med_rev.max() - util.filtro(df_novo,'produto', '!=', 'GLP').preco_med_rev.min())

df_novo['preco_med_norm_comb'] = 0
for i in range(len(df_novo.preco_med_rev)):
    df_novo['preco_med_norm_comb'].iloc[i] = (df_novo.preco_med_rev.iloc[i] - util.filtro(df_novo,'produto', '==', df_novo.produto.iloc[i]).preco_med_rev.min())/(util.filtro(df_novo,'produto', '==', df_novo.produto.iloc[i]).preco_med_rev.max() - util.filtro(df_novo,'produto', '==', df_novo.produto.iloc[i]).preco_med_rev.min())

# Checando os mínimos e máximos para ter certeza que foram devidamente normalizados
df_novo.pivot_table(values='preco_med_norm_tot', 
                        index=['produto'], 
                        aggfunc=[min, max])

df_novo.pivot_table(values='preco_med_norm_comb', 
                        index=['produto'], 
                        aggfunc=[min, max])

"""
Análise exploratória dos dados

Variáveis categóricas

"""
# Mostrar nomes e percentual de observações por Região
df_novo.reg.value_counts(normalize=True)

# Número de observações por Região
util.count_plot("reg", df_novo, title="Número de observações por região", xlabel='Região', ylabel='Contagem')


sigla_estado = {
        "ACRE": "AC",
        "ALAGOAS": "AL",
        "AMAPA": "AP",
        "AMAZONAS": "AM",
        "BAHIA": "BA",
        "CEARA": "CE",
        "DISTRITO FEDERAL": "DF",
        "ESPIRITO SANTO": "ES",
        "GOIAS": "GO",
        "MARANHAO": "MA",
        "MATO GROSSO": "MT",
        "MATRO GROSSO DO SUL": "MS",
        "MINAS GERAIS": "MG",
        "PARA": "PA",
        "PARAIBA": "PB",
        "PARANA": "PR",
        "PERNAMBUCO": "PE",
        "PIAUI": "PI",
        "RIO DE JANEIRO": "RJ",
        "RIO GRANDE DO NORTE": "RN",
        "RIO GRANDE DO SUL": "RS",
        "RONDONIA": "RO",
        "RORAIMA": "RR",
        "SANTA CATARINA": "SC",
        "SAO PAULO": "SP",
        "SERGIPE": "SE",
        "TOCANTINS": "TO"
        }

# Número de observações por Estado
df_novo["sigla"] = df_novo.estado.map(sigla_estado)
df_novo.sigla.value_counts()
util.count_plot("sigla", df_novo, title="Número de observações por estado", xlabel='Estado', ylabel='Contagem')

# Frequência relativa de observações por combustível
df_novo['produto'].value_counts(normalize=True)

# Número de observações por tipo de combustível
util.count_plot('produto', df_novo, title="Proporção de tipo de combustíveis", xlabel='Combustível', ylabel='Contagem', rot=12)


combustivel_anos = df_novo.pivot_table(values='ano', 
                        index=['produto'], 
                        aggfunc=[min, max])


# Qual é a proporção de observações de cada combustível por Estado
comb_por_estado = df_novo.groupby("estado")["produto"].value_counts().to_frame("contagem").reset_index()
comb_por_estado = comb_por_estado.pivot("estado", "produto")

fig, ax = plt.subplots(figsize=(15,10))

# Gráfico de barras emilhadas horizontal
comb_por_estado.plot(kind="barh", stacked=True, ax=ax)

plt.title("Combustíveis observados por Estado", fontsize=22)
plt.xlabel("Número de observações", fontsize=18)
plt.ylabel("Estado", fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.legend(sorted(df_novo.produto.unique().tolist()), loc='center left', bbox_to_anchor=(1, 0.5),prop={'size': 15})
plt.tight_layout()
plt.show()

"""     
Análise exploratória dos dados

Variáveis numéricas

"""
# Contagem por data e data inicial e final
count = len(df_novo.data)
start = df_novo.data.min().strftime('%Y-%m')
end = df_novo.data.max().strftime('%Y-%m')

print("The are {} observations beginning on {} and ending {}".format(count, start, end))


df_novo.num_de_postos_pesquisados.describe()

# Gráfico de quantidade de postos analizados por ano

postos_por_ano = df_novo.pivot_table(values='num_de_postos_pesquisados', 
                        index=[df_novo.ano, 'estado'], 
                        aggfunc=sum).reset_index()



fig, ax = plt.subplots(figsize=(20,14))
sns.scatterplot('ano','num_de_postos_pesquisados', data=postos_por_ano, hue='estado', ax=ax)
ax.legend(sorted(postos_por_ano.estado.unique().tolist()), loc='center left', 
          bbox_to_anchor=(1, 0.5), prop={'size': 18})
plt.title('Número de postos pesquisados anualmente por Estado', fontsize=22)
plt.show()

fig.savefig('imagem.png')  # eps, pdf, pgf, png, ps, raw, rgba, svg, svgz

# Preço médio
df_novo.preco_med_rev.describe()

util.ecdf(df_novo, 'preco_med_rev')

# 2 boxplots com escalas diferentes
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,6.5), gridspec_kw={"width_ratios":[5,1], "wspace":0})

# Eixo para produtos com preço médio similares
sns.boxplot(x="produto", y="preco_med_rev", data=df_novo[df_novo.produto!="GLP"], order=["ETANOL", "GASOLINA", 
                                                              "GNV", "DIESEL", "DIESEL S10"], ax=ax[0])

# Eixo para GLP
sns.boxplot(x="produto", y="preco_med_rev", data=df_novo[df_novo.produto=="GLP"],order=['GLP'], ax=ax[1], color='g')

plt.suptitle("Distribuição dos preços médios dos combustíveis no Brasil", fontsize=18)

# Formatando o gráfico de GLP
ax[1].yaxis.tick_right()
ax[1].yaxis.set_label_position("right")
ax[0].set_xlabel('Produto', fontsize=16)
ax[1].set_xlabel('GLP')
ax[0].set_ylabel('Preço Grupo 1: Combustíveis líquidos & GNV', fontsize=16)
ax[1].set_ylabel('Preço Grupo 2: GLP', fontsize=16)
ax[1].set(xticklabels=[])

# Adiciona subplot para GLP
plt.tight_layout(pad=5)

# ECDF Preço médio por produto
util.ecdf_category(df_novo, "preco_med_rev", "produto", title="ECDF Curves for Fuel Mean Price", xlabel="Mean Price")

# Preço médio anual
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12,9))

# Todos os combustíveis exceto GLP
sns.lineplot(x=df_novo[df_novo.produto!='GLP'].ano, y="preco_med_rev", data=df_novo[df_novo.produto!="GLP"], hue="produto", ax=ax[0], err_style=None)
ax[0].set_xlabel("Ano")
ax[0].set_ylabel("Preço Médio Nacional")
ax[0].set_title("Combustíveis líquidos + GNV", fontsize=14)

# GLP
sns.lineplot(x=df_novo[df_novo.produto=='GLP'].ano, y="preco_med_rev", data=df_novo[df_novo.produto == "GLP"], hue="produto", ax=ax[1], err_style=None)
ax[1].set_xlabel("Ano")
ax[1].set_ylabel("Preço Médio Nacional")
ax[1].set_title("GLP", fontsize=14)

plt.suptitle("Brasil: Média nacional dos preços dos combustíveis", fontsize=18)
plt.tight_layout(pad=5)
plt.show()

util.ecdf_category(df_novo, "preco_med_dist", "produto", title="ECDF Curves for Fuel Mean Distribution Price", 
             xlabel="Mean Distribution Price")


"""
Questions
How did the price change for the different regions of Brazil?
Within a region, which states increased their prices the most?
Which states are the cheapest (or most expensive) for different types of fuels?
To answer these questions we will 1) include geographical shape files of Brazil's states and 2) create a new dataframe containing the percent change in prices.
"""
# Dados geográficos

url = "ftp://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2016/Brasil/BR/br_unidades_da_federacao.zip"
brazil_geo = gpd.read_file(url)

state_dict = {"RONDÔNIA":"RONDONIA", "PARÁ":"PARA", "AMAPÁ":"AMAPA", "MARANHÃO":"MARANHAO",
             "PIAUÍ":"PIAUI", "CEARÁ":"CEARA", "PARAÍBA":"PARAIBA", "ESPÍRITO SANTO":"ESPIRITO SANTO",
             "SÃO PAULO":"SAO PAULO", "PARANÁ":"PARANA", "GOIÁS":"GOIAS", "ACRE":"ACRE",
             "AMAZONAS":"AMAZONAS", "RORAIMA":"RORAIMA", "TOCANTINS":"TOCANTINS", 
             "RIO GRANDE DO NORTE":"RIO GRANDE DO NORTE", "PERNAMBUCO":"PERNAMBUCO", 
             "ALAGOAS":"ALAGOAS", "SERGIPE":"SERGIPE", "BAHIA":"BAHIA", "MINAS GERAIS":"MINAS GERAIS",
             "RIO DE JANEIRO":"RIO DE JANEIRO", "SANTA CATARINA":"SANTA CATARINA", "MATO GROSSO DO SUL":"MATO GROSSO DO SUL", 
             "MATO GROSSO":"MATO GROSSO", "DISTRITO FEDERAL":"DISTRITO FEDERAL", "RIO GRANDE DO SUL":"RIO GRANDE DO SUL"}

brazil_geo["NM_ESTADO"] = brazil_geo.NM_ESTADO.map(state_dict)

brazil_geo.crs = {"init": "epsg:4326"}

brazil_geo.columns = ['estado', 'reg', 'CD_GEOCUF', 'geometry']
brazil_geo["reg"] = brazil_geo.reg.str.replace("-", " ")

# Extrai a geometria das regiões
brazil_geo_region = brazil_geo.dissolve(by='reg').reset_index()
brazil_geo_region = brazil_geo_region[['reg', 'geometry']]


# Percentual de mudança do preço
# DataFrame df_pct_change com a mudança dos preços

regioes = df_novo.reg.unique().tolist()

df_pct_change = pd.DataFrame()

count = 0

for i in range(len(regioes)):
    regiao = regioes[i]
    estados = df_novo[df_novo.reg==regiao]["estado"].unique()
    
    for i in range(len(estados)):
        estado = estados[i]
        produtos = df_novo[(df_novo.reg==regiao) & (df_novo.estado==estado)]["produto"].unique()
       
        for i in range(len(produtos)):
            produto = produtos[i]
            anos = df_novo[(df_novo.reg==regiao) & (df_novo.estado==estado) & (df_novo.produto==produto)]["ano"].unique()
            
            mean_price = df_novo[(df_novo.reg==regiao) & (df_novo.estado==estado) & 
                            (df_novo.produto==produto)]["preco_med_rev"].mean()
            
            # % de alteração para os preços brutos
            first_price = df_novo[(df_novo.reg==regiao) & (df_novo.estado==estado) & 
                            (df_novo.produto==produto) & (df_novo.ano==anos[0])]["preco_med_rev"].iloc[0]
            last_price = df_novo[(df_novo.reg==regiao) & (df_novo.estado==estado) & 
                            (df_novo.produto==produto) & (df_novo.ano==anos[-1])]["preco_med_rev"].iloc[-1]
            price_pct_change = (last_price - first_price) / np.abs(first_price)
        
        
            # % de alteração para os preços normalizados
            first_price_norm = df_novo[(df_novo.reg==regiao) & (df_novo.estado==estado) & 
                            (df_novo.produto==produto) & (df_novo.ano==anos[0])]["preco_med_norm_tot"].iloc[0]
            last_price_norm = df_novo[(df_novo.reg==regiao) & (df_novo.estado==estado) & 
                            (df_novo.produto==produto) & (df_novo.ano==anos[-1])]["preco_med_norm_tot"].iloc[-1]
            price_pct_change_norm = (last_price_norm - first_price_norm) / np.abs(first_price_norm)            
            
            # Adiciona ao DF
            df_temp = pd.DataFrame({"reg":regiao, "estado":estado, 
                                    "produto":produto, 
                                    "First_Year":anos[0], "Last_Year":anos[-1], "Fuel_Mean_Price":mean_price,
                                    "First_Price":first_price, "Last_Price":last_price, "Price_Pct_Change":price_pct_change, 
                                    "First_Price_Norm":first_price_norm, "Last_Price_Norm":last_price_norm, 
                                    "Price_Pct_Change_Norm":price_pct_change_norm
                                   }, 
                                   index=[count])
            
            df_pct_change = df_pct_change.append(df_temp)
            
            count += 1


# Agrupando os dados por região
df_pct_change_region = df_pct_change.groupby(['reg', 'produto']).mean().reset_index()

# Adicionando a informação geográfica ao DF
brazil_geo = brazil_geo.drop('reg', axis=1).merge(df_pct_change, on='estado')
brazil_geo_region = brazil_geo_region.merge(df_pct_change_region, on='reg')

# Corrigindo os tipos
for col in ['First_Year', 'Last_Year']:
    brazil_geo_region[col] = brazil_geo_region[col].astype(int)

# Save geographical data to disk
#brazil_geo.to_csv('../data/interim/brazil_geo.csv')
#brazil_geo_region.to_csv('../data/interim/brazil_geo_region.csv')
    
"""
Como o preço dos combustíveis se alterou entre as regiões brasileiras?
"""
# Gráfico com os preços absolutos

nrows = 2
ncols = 3
produtos = df_novo.produto.unique().tolist()

fig_raw, ax_raw = plt.subplots(figsize=(20,10), nrows=nrows, ncols=ncols)

n = 0

for row in range(nrows):
   
    for col in range(ncols):
        
        divider = make_axes_locatable(ax_raw[row,col])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        brazil_geo_region[brazil_geo_region.produto==produtos[n]].plot(column="Price_Pct_Change", 
                                                                       legend=True, ax=ax_raw[row,col], cmap="Reds", 
                                                                       edgecolor='black')
        
        # Plot titles
        year_range = brazil_geo_region[brazil_geo_region.produto==produtos[n]][["First_Year", "Last_Year"]]
        year_range = [year_range["First_Year"].min(), year_range["Last_Year"].max()]
        year_range = str(year_range[0]) + "-" + str(year_range[1])
        
        ax_raw[row,col].set_title(produtos[n] + ": " + year_range)
        ax_raw[row,col].set_xlabel("Longitude")
        ax_raw[row,col].set_ylabel("Latitude")
        n += 1

fig_raw.suptitle("Brasil: Alteração percentual por produto por região", fontsize=20)
plt.tight_layout(pad=5)
plt.show()

"""
2. Entre as regiões, qual estado teve a maior alteração percentual nos preços
dos combustíveis?
"""
# Gráfico com os preços absolutos por estado
nrows = 2
ncols = 3
produtos = df_novo.produto.unique().tolist()

fig_raw, ax_raw = plt.subplots(figsize=(20,12), nrows=nrows, ncols=ncols)

n = 0

for row in range(nrows):
   
    for col in range(ncols):
        
        divider = make_axes_locatable(ax_raw[row,col])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        brazil_geo[brazil_geo.produto==produtos[n]].plot(column="Price_Pct_Change", 
                                                                       legend=True, ax=ax_raw[row,col], cmap="Reds", 
                                                                       edgecolor='black')
        
        # Plot titles
        year_range = brazil_geo[brazil_geo.produto==produtos[n]][["First_Year", "Last_Year"]]
        year_range = [year_range["First_Year"].min(), year_range["Last_Year"].max()]
        year_range = str(year_range[0]) + "-" + str(year_range[1])
        
        ax_raw[row,col].set_title(produtos[n] + ": " + year_range, fontsize=20)
        ax_raw[row,col].set_xlabel("Longitude", fontsize=18)
        ax_raw[row,col].set_ylabel("Latitude", fontsize=18)
        n += 1

fig_raw.suptitle("Brasil: Alteração percentual por produto por Estado", fontsize=28)
plt.tight_layout(pad=5)
plt.show()

"""
Qual é o combustível mais barato/caro por Estado
"""
# Agrupando por estado e produto
brazil_fuel_state = brazil_geo.groupby(["estado", "produto"])["Fuel_Mean_Price"].mean().reset_index()

# Reagrupando com os dados espaciais
brazil_fuel_state_geo = brazil_geo.merge(brazil_fuel_state.drop(['produto','Fuel_Mean_Price'], axis=1) , on="estado")


nrows = 2
ncols = 3
produtos = df_novo.produto.unique()

fig, ax = plt.subplots(figsize=(20,12), nrows=nrows, ncols=ncols)

# set counter
n = 0

# Iterate through figure axes
for row in range(nrows): # iterate through rows
   
    for col in range(ncols): # iterate through each column while on one row
        
        # Adjust location and size of legend
        divider = make_axes_locatable(ax[row,col]) 
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        # Plot choropleth
        brazil_fuel_state_geo[brazil_fuel_state_geo.produto==produtos[n]].plot(column="Fuel_Mean_Price", 
                                                                       cmap="coolwarm", 
                                                                       legend=True, ax=ax[row,col], 
                                                                       edgecolor='black')
        
        # Set title and labels
        ax[row,col].set_title(produtos[n], fontsize=20)
        ax[row,col].set_xlabel("Longitude", fontsize=18)
        ax[row,col].set_ylabel("Latitude", fontsize=18)
        
        n += 1

fig.suptitle("Brasil: Preço médio dos combustíveis por Estado", fontsize=28)
plt.tight_layout(pad=5)
plt.show()


estados = brazil_fuel_state.estado.unique().tolist()

# Create dataframe of cheapest fuels by state
cheapest_fuels = pd.DataFrame()

# Iterate through group-by objects and extract minimum: extreme_of_group(data, column, extreme)
for i in range(len(estados)):

    state_group = brazil_fuel_state_geo.groupby("estado").get_group(estados[i])
    state_group = state_group[state_group.Fuel_Mean_Price == state_group.Fuel_Mean_Price.min()]
    cheapest_fuels = cheapest_fuels.append(state_group)
    cheapest_fuels.rename({"Fuel_Mean_Price":"Fuel_Min_Price"}, inplace=True)
    
# Create dataframe of cheapest fuels by state
expensive_fuels = pd.DataFrame()

# Remove LPG from list of products
brazil_fuel_state_geo_noLPG = brazil_fuel_state_geo[brazil_fuel_state_geo.produto!="GLP"]

# Iterate through group-by objects and extract minimum: extreme_of_group(data, column, extreme)
for i in range(len(estados)):

    state_group = brazil_fuel_state_geo_noLPG.groupby("estado").get_group(estados[i])
    state_group = state_group[state_group.Fuel_Mean_Price == state_group.Fuel_Mean_Price.max()]
    expensive_fuels = expensive_fuels.append(state_group)
    expensive_fuels.rename({"Fuel_Mean_Price":"Fuel_Max_Price"}, inplace=True)
    

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

# Choropleths of cheapest and most expensive fuels
expensive_fuels.plot(column="produto", cmap="Pastel2", legend=True, ax=ax[1], edgecolor='black')
cheapest_fuels.plot(column="produto", cmap="Pastel2", legend=True, ax=ax[0], edgecolor='black')

# Format figure and axes
ax[0].set_title("Combustível mais barato", fontsize=20)
ax[1].set_title("Combustível mais caro", fontsize=20)
ax[0].set_ylabel("Latitude", fontsize=18)
ax[0].set_xlabel("Longitude", fontsize=18)
ax[1].set_ylabel("Latitude", fontsize=18)
ax[1].set_xlabel("Longitude", fontsize=18)

plt.suptitle("Brasil: Comparação dos combustíveis por Estado", fontsize=28)
plt.tight_layout(pad=3)
plt.show()


"""
Qual estado tem a pior/melhor relação Gasolina/Álcool?
Qual estado tem a pior/melhor relação Gasolina/GNV?
GNV: 13 a 14km/m3
Gasolina: 10km/l
Alcool: 7km/l

"""

brazil_state_alc_gas = util.filtro(df_novo,'ano','==','2019').pivot_table(values='preco_med_rev',
                   index=['estado'], columns=['produto'])
brazil_state_alc_gas = pd.DataFrame(brazil_state_alc_gas.ETANOL/brazil_state_alc_gas.GASOLINA)
brazil_state_alc_gas.columns = ['relacao']
brazil_state_alc_gas = brazil_geo.filter(['estado','CD_GEOCUF','geometry']).merge(brazil_state_alc_gas, on="estado")

fig, ax = plt.subplots(figsize=(18,18))
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)

brazil_state_alc_gas.plot(column="relacao", cmap="coolwarm", legend=True, ax=ax, edgecolor='black', cax=cax)
ax.set_title("Relação Álcool/Gasolina", fontsize=20)
ax.set_ylabel("Latitude", fontsize=18)
ax.set_xlabel("Longitude", fontsize=18)
plt.tight_layout(pad=3)
plt.show()


# Preço médio anual
df_novo['anomes'] = df_novo.data.dt.strftime('%Y%m')
df_novo_1819 = df_novo[df_novo.anomes >= '201801']

fig, ax = plt.subplots(figsize=(10,10))
sns.lineplot(x=df_novo_1819[df_novo_1819.produto=='DIESEL'].anomes, y="preco_med_rev", data=df_novo_1819[df_novo_1819.produto=='DIESEL'], hue="estado", ax=ax, err_style=None)
handles, labels = ax.get_legend_handles_labels()
handles.remove(handles[0])
labels.remove(labels[0])


# Por estado
nrows = 2
ncols = 3
produtos = df_novo.produto.unique()

fig, ax = plt.subplots(figsize=(20,12), nrows=nrows, ncols=ncols)

# set counter
n = 0

# Iterate through figure axes
for row in range(nrows): # iterate through rows
   
    for col in range(ncols): # iterate through each column while on one row
        
        sns.lineplot(x=df_novo_1819[df_novo_1819.produto==produtos[n]].anomes, y="preco_med_rev", data=df_novo_1819[df_novo_1819.produto==produtos[n]], hue="estado", ax=ax[row,col], err_style=None, legend=False)
        
        # Set title and labels
        ax[row,col].set_title(produtos[n], fontsize=20)
        ax[row,col].set_xlabel("Ano mês", fontsize=18)
        ax[row,col].set_ylabel("Preço Médio", fontsize=18)
        ax[row,col].tick_params(labelrotation=45)


        n += 1

fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.1,0.75), ncol=1, borderpad=2.5, title='Estados')
plt.suptitle("Brasil: Evolução dos preços de cada combustível por estado", fontsize=18)

plt.tight_layout(pad=3.5, h_pad=3)
plt.show()


# Por região
fig, ax = plt.subplots(figsize=(10,10))
sns.lineplot(x=df_novo_1819[df_novo_1819.produto=='DIESEL'].anomes, y="preco_med_rev", data=df_novo_1819[df_novo_1819.produto=='DIESEL'], hue="reg", ax=ax, err_style=None)
handles, labels = ax.get_legend_handles_labels()
handles.remove(handles[0])
labels.remove(labels[0])

nrows = 2
ncols = 3
produtos = df_novo.produto.unique()

fig, ax = plt.subplots(figsize=(20,12), nrows=nrows, ncols=ncols)

# set counter
n = 0

# Iterate through figure axes
for row in range(nrows): # iterate through rows
   
    for col in range(ncols): # iterate through each column while on one row
        
        sns.lineplot(x=df_novo_1819[df_novo_1819.produto==produtos[n]].anomes, y="preco_med_rev", data=df_novo_1819[df_novo_1819.produto==produtos[n]], hue="reg", ax=ax[row,col], err_style=None, legend=False)
        
        # Set title and labels
        ax[row,col].set_title(produtos[n], fontsize=20)
        ax[row,col].set_xlabel("Ano mês", fontsize=18)
        ax[row,col].set_ylabel("Preço Médio", fontsize=18)
        ax[row,col].tick_params(labelrotation=45)


        n += 1

fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.09,0.55), ncol=1, borderpad=2.5, title="Região")
plt.suptitle("Brasil: Evolução dos preços de cada combustível por região", fontsize=18)
plt.tight_layout(pad=3.5, h_pad=3)
plt.show()


# Alteração percentual do preço dos combustíveis somente entre 2018 e 2019

preco_2018 = df_novo[df_novo.anomes == '201801'].filter(['estado', 'produto', 'preco_med_rev'])
preco_2018 = preco_2018[preco_2018.produto != 'GNV']

preco_2019 = df_novo[df_novo.anomes == '201905'].filter(['estado', 'produto', 'preco_med_rev'])
preco_2019 = preco_2019[preco_2019.produto != 'GNV']

brazil_state_alt_preco = preco_2018.merge(preco_2019, on=['estado','produto'])
brazil_state_alt_preco.rename(columns={'preco_med_rev_x':'preco_med_rev_2018','preco_med_rev_y':'preco_med_rev_2019'}, inplace=True)
brazil_state_alt_preco['alteracao_preco'] = round((brazil_state_alt_preco.preco_med_rev_2019/brazil_state_alt_preco.preco_med_rev_2018) - 1, 4)
brazil_state_alt_preco['alteracao_preco'] = brazil_state_alt_preco['alteracao_preco'].map("{0:.2%}".format)


menor_alteracao = brazil_state_alt_preco.sort_values(by=['estado','alteracao_preco'], 
                                                     ascending = [True, True]).drop_duplicates(subset=['estado'], keep='first')
maior_alteracao = brazil_state_alt_preco.sort_values(by=['estado','alteracao_preco'], 
                                                     ascending = [True, True]).drop_duplicates(subset=['estado'], keep='last')

brazil_state_alt_preco = brazil_geo.filter(['estado','CD_GEOCUF','geometry']).merge(brazil_state_alt_preco, on="estado")




nrows = 2
ncols = 3
produtos = brazil_state_alt_preco.produto.unique().tolist()

fig_raw, ax_raw = plt.subplots(figsize=(20,12), nrows=nrows, ncols=ncols)

n = 0

for row in range(nrows):
   
    for col in range(ncols):
        if row == 1 and col == 2:
            break
        else:
            divider = make_axes_locatable(ax_raw[row,col])
            cax = divider.append_axes("right", size="5%", pad=0.1)
            brazil_state_alt_preco[brazil_state_alt_preco.produto==produtos[n]].plot(column="alteracao_preco", 
                                                                           legend=True, ax=ax_raw[row,col], cmap="Reds", 
                                                                           edgecolor='black')
            
            # Plot titles
            
            ax_raw[row,col].set_title(produtos[n] + ": 2018-2019" , fontsize=20)
            ax_raw[row,col].set_xlabel("Longitude", fontsize=18)
            ax_raw[row,col].set_ylabel("Latitude", fontsize=18)
            n += 1


fig_raw.suptitle("Brasil: Alteração percentual por produto por Estado", fontsize=28)
plt.tight_layout(pad=5)
plt.show()


"""
Qual produto teve maior alteração percentual e qual teve a menor em cada estado?
"""

maior_alteracao = brazil_geo.filter(['estado','CD_GEOCUF','geometry']).merge(maior_alteracao, on="estado")
menor_alteracao = brazil_geo.filter(['estado','CD_GEOCUF','geometry']).merge(menor_alteracao, on="estado")


maior_alteracao['coords'] = maior_alteracao['geometry'].apply(lambda x: x.representative_point().coords[:])
maior_alteracao['coords'] = [coords[0] for coords in maior_alteracao['coords']]


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,10))

# Choropleths of cheapest and most expensive fuels
maior_alteracao.plot(column="produto", cmap="Pastel2", legend=True, ax=ax[1], edgecolor='black')
menor_alteracao.plot(column="produto", cmap="Pastel2", legend=True, ax=ax[0], edgecolor='black')

for idx, row in maior_alteracao.iterrows():
    plt.annotate(s=row['alteracao_preco'], xy=row['coords'],
                 horizontalalignment='center')


# Format figure and axes
ax[0].set_title("Combustível com menor aumento no preço entre 2018 e 2019", fontsize=20)
ax[1].set_title("Combustível com maior aumento no preço entre 2018 e 2019", fontsize=20)
ax[0].set_ylabel("Latitude", fontsize=18)
ax[0].set_xlabel("Longitude", fontsize=18)
ax[1].set_ylabel("Latitude", fontsize=18)
ax[1].set_xlabel("Longitude", fontsize=18)

plt.suptitle("Brasil: Comparação do aumento do preço dos combustíveis por Estado", fontsize=28)
plt.tight_layout(pad=3)
plt.show()



