#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:24:54 2019

@author: viniciussaurin

Funções úteis
"""
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def filtro(df, coluna, operacao, filtro): 
    """
    Função que retorna o DataFrame df com o filtro e operação aplicados na coluna
    
    Exemplo de múltiplos filtros:
        teste = util.filtro(
            util.filtro(df.sort_index()['201401'], 
                        'estado', '==', 'SERGIPE'), 
                            'produto', '==', 'GLP')

    """
    if filtro.isnumeric():
        return eval('df[df.' + coluna + ' '  + operacao + ' ' + filtro + ']')
    else:
        return eval('df[df.' + coluna + ' ' + operacao + " '" + filtro + "'" + ']')
    
def count_plot(column, df, title=None, xlabel=None, ylabel=None, rot=None):
    """
    Apresenta o countplot do pacote Seaborn, mas é mais fácil de escrever
    """

    fig, ax = plt.subplots(figsize=(14,6))
    ax = sns.countplot(x=column, data=df)
    plt.title(title, fontsize=28)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=24)
    plt.xticks(rotation=rot, fontsize=20)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    
    plt.show()

# Definindo a função estimadora da distrivuição cumulativa (ECDF)
def ecdf(data, column, title=None, color=None):
    n = len(data[column])
    x = np.sort(data[column])
    y = np.arange(1, n+1)/n
    
    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(x, y, marker='.', linestyle='none', color=color)
    plt.title(title)
    plt.xlabel("{}".format(column))
    plt.ylabel("ECDF")

def ecdf_category(df, column, group, title=None, xlabel=None):

    # Create figure with ECDF plots for each fuel Mean Price 
    products = df[group].unique().tolist()

    # Group dataframe by Product and select Mean Price
    product_group = df.groupby(group)[column]

    # Set rows and columns
    ncols = int(3)
    nrows = int(len(products) / ncols if len(products) % 2 == 0 else (len(products) + 1) / ncols)
    

    # List of colors
    color = ["b", "y", "g", "m", "r", "c"]

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))

    # Index counter
    n = 0

    # Create subplots
    for row in range(nrows):
        for col in range(ncols):
        
            df_product = product_group.get_group(products[n]).to_frame("Mean Price")
   
            x = np.sort(df_product["Mean Price"]) 
            y = np.arange(1, len(df_product["Mean Price"])+1) / len(df_product["Mean Price"])
        
            ax[row,col].step(x, y, color=color[n])
            ax[row,col].set_ylabel("ECDF")
            ax[row,col].set_xlabel(xlabel)
            ax[row,col].set_title(products[n])
        
            n += 1

    plt.tight_layout(pad=5)
    plt.suptitle(title, fontsize=18)
    
    return plt.show()


"""
Normalizacao
teste = list(range(len(df_novo.preco_med_rev)))
for i in range(len(df_novo.preco_med_rev)):
    if df_novo.produto.iloc[i] == 'GLP':
        teste[i] = df_novo.preco_med_rev.iloc[i] - util.filtro(df_novo,'produto', '==', 'GLP').preco_med_rev.min()/(util.filtro(df_novo,'produto', '==', 'GLP').preco_med_rev.max() - util.filtro(df_novo,'produto', '==', 'GLP').preco_med_rev.min())
    else:
        teste[i] = df_novo.preco_med_rev.iloc[i] - util.filtro(df_novo,'produto', '!=', 'GLP').preco_med_rev.min()/(util.filtro(df_novo,'produto', '!=', 'GLP').preco_med_rev.max() - util.filtro(df_novo,'produto', '!=', 'GLP').preco_med_rev.min())


normalizador = lambda x: (x - x.min()) / (x.max() - x.min())

# Normalizando total

# Normalizando preços para todos os combustíveis, exceto GLP
df["preco_med_norm_1"] = util.filtro(df_novo,'produto', '!=', 'GLP').groupby("produto")["preco_med_rev"].transform(normalizador)

# Normalizando preços para todos GLP
df["preco_med_norm_2"] = util.filtro(df_novo,'produto', '==', 'GLP').groupby("produto")["preco_med_rev"].transform(normalizador)

# Combinando as colunas
df["preco_med_norm_tot"] = df["preco_med_norm_1"].fillna(df["preco_med_norm_2"])
df.drop(["preco_med_norm_1", "preco_med_norm_2"], axis=1, inplace=True)



# Primeiros e ultimos anos de combustivel
produtos = df_novo.produto.unique().tolist()

combustivel_anos = pd.DataFrame()

## Extract first and last years of observation for each fuel product
for i, produto in enumerate(produtos):
    
    df_temp = pd.DataFrame({"Produto":produto, "First_Year":util.filtro(df_novo,'produto','==',produto).data.dt.year.min(), 
                            "Last_Year":util.filtro(df_novo,'produto','==',produto).data.dt.year.max()}, index=[i])
    
    combustivel_anos = combustivel_anos.append(df_temp)
    
combustivel_anos.set_index("Produto", inplace=True)




"""