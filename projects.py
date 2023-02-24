import pandas as pd
import awswrangler as wr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pandas.core.common import standardize_mapping
from sklearn.preprocessing import StandardScaler


#Extraccion de datos
df = wr.s3.read_csv(path="s3://proyecto-data/data2//2023-02-22-19-20-28/044c3fea-fb0c-49c6-bd6e-46b2c7a3c08b.csv",header=None)

#Limpieza de datos
df[5]=df[5].apply(lambda x: x.replace(" ",""))
correos=df[5].apply(lambda x: x.split('@')[1])
df['correo']=correos.apply(lambda x: 1 if x== 'gmail.com' else 0)

lista=df[1].apply(lambda x: x.split(','))
unicos=list(set([item.strip().lower() for sublist in lista.tolist() for item in sublist]))

cluster={'viaje':['viajar con amigos','pareja o familia'],'peliculas':['ir al cine con pareja o amigos','ver peliculas con tu pareja o familia o amigos'],'conciertos':['ir a conciertos'],'salidas':['hacer reuniones tranquilas en tu casa con familiares y amigos','salir a fiestas','tener una cita','comer con amigas o amigos','hacer una parrillada con familiares y amigos','salir a bailar','ir al casino','ir a comer a tu restaurante favoritos','albercada con amigos'],'estudio':['trabajar en un proyecto con colegas'],'deporte':['ir al estadio o evento deportivo','hacer deporte acompañado'],'videojuego':['jugar videojuegos']}

#Categorizacion de datos
for i in cluster:
  for cont,j in enumerate(cluster[i]):
    if cont ==0:
      value=lista.apply(lambda x: 1 if j in [ele.strip().lower() for ele in x] else 0)
    else:
      value_=lista.apply(lambda x: 1 if j in [ele.strip().lower() for ele in x] else 0)
      value += value_
  df[i]=value

lista=df[2].apply(lambda x: x.split(','))
unicos=list(set([item.strip().lower() for sublist in lista.tolist() for item in sublist]))
unicos

cluster={'extrovertido':['patinar sobre patineta o patines','correr','nadar','caminar','pasar la tarde en una alberca','pasear en bicicleta o hacer ciclismo','surfear','escalar o hacer rapel','entrenar en el gimnacio','ir en tu restaurante favorito','conducir','estar en la playa tomando el sol','cantar','hacer videos para internet','ir de compras','alcohol','viajar a lugares desconocidos y extravagantes'],'introvertido':['estudiar y aprender cualquier cosa','cocinar en tu casa','tomar fotografias','hacer tareas del hogar','escribir','leer','dormir','diseñar/crear cosas/inventar','arreglar tu casa','ver videos en internet','ver series','ver deportes en tv','hacer jardineria','jugar videojuegos']}



for i in cluster:
  for cont,j in enumerate(cluster[i]):
    if cont ==0:
      value=lista.apply(lambda x: 1 if j in [ele.strip().lower() for ele in x] else 0)
    else:
      value_=lista.apply(lambda x: 1 if j in [ele.strip().lower() for ele in x] else 0)
      value += value_
  df[i]=value

lista=df[6].apply(lambda x: x.split(','))
unicos=list(set([item.strip().lower() for sublist in lista.tolist() for item in sublist]))
unico=[item.strip().lower() for sublist in lista.tolist() for item in sublist]
dep=pd.Series(unico).value_counts().head(10).index.tolist()
for i in range(len(dep)):
  value=lista.apply(lambda x: 1 if dep[i] in [ele.strip().lower() for ele in x] else 0)
  df[dep[i]]=value

lista=df[9].apply(lambda x: x.split(','))
unicos=list(set([item.strip().lower() for sublist in lista.tolist() for item in sublist]))
generos={}
for gen in unicos:
  generos[gen]=lista.apply(lambda x:1 if gen in [ele.strip().lower() for ele in x] else 0)

musica=pd.DataFrame(generos)

musica.columns = ['mus_'+x for x in musica.columns]
df=pd.concat([df,musica],axis=1)

lista=df[11].apply(lambda x: x.split(','))
unicos=list(set([item.strip().lower() for sublist in lista.tolist() for item in sublist]))
genero={}
for gen in unicos:
  genero[gen]=lista.apply(lambda x:1 if gen in [ele.strip().lower() for ele in x] else 0)

etretenimiento=pd.DataFrame(genero)
df=pd.concat([df,etretenimiento],axis=1)

lista=df[17].apply(lambda x: x.split(','))
unicos=list(set([item.strip().lower() for sublist in lista.tolist() for item in sublist]))
genero={}
for gen in unicos:
  genero[gen]=lista.apply(lambda x:1 if gen in [ele.strip().lower() for ele in x] else 0)

entretener=pd.DataFrame(genero)
df=pd.concat([df,entretener],axis=1)

col=[x for x in df.columns if type(x) != type(1)]
x=df[col]

cor = x.corr()
filter=cor > 0.40
cor.where(filter, inplace = True)
lista=filter.columns
#for elem in lista:
  #cont=0
  #for num,i in enumerate(filter[elem]):
    #if i == True and filter.index[num]!= elem and elem == "introvertido":
      #if cont==0:
        #print("\n----------",elem,"----------\n")
      #print(num,"--",i,"---",filter.index[num],"---",cor[elem][num])
      #cont+=1

#introvertido - documentales e historia - noticias - basket
#extrovertido - conciertos - mus_reggae - noticias sobre el narco

X_2=x[['introvertido','documentales e historia','noticias','basket']]
X_3=x[['conciertos','mus_reggae','noticias sobre el narco','mus_techno']]

pca1= PCA(n_components=1)
pca1.fit(X_2)
X_2['scale']=(pca1.fit_transform(X_2).reshape(1,47)[0])
X_2
X_2.rename(columns={'scale':'intro-informado'},inplace=True)

pca1= PCA(n_components=1)
pca1.fit(X_3)
X_3['scale']=(pca1.fit_transform(X_3).reshape(1,47)[0])
X_3
X_3.rename(columns={'scale':'extro-concert'},inplace=True)
union=pd.concat([X_2,X_3], axis=1)[['intro-informado','extro-concert']]

#import seaborn as sns
#sns.scatterplot(data=union, x ='intro-informado', y='extro-concert')


scaler= StandardScaler()
standardize_data = scaler.fit_transform(union)

model=KMeans(n_clusters=4)
model.fit(standardize_data)

union['cluster']= model.labels_

#sns.scatterplot(data=union, x ='intro-informado', y='extro-concert',hue='cluster')

#pd.pivot_table(union,index='cluster')

#Grupo 0 : extrovertido
#Grupo 1 : conservadores
#Grupo 2 : Deportistas
#Grupo 3 : introvertido

id = df[range(5,17,11)]

id.rename(columns={5:'correo',16:'nombre'},inplace=True)
id['cluster'] = model.labels_
id['Categoria']=id['cluster'].apply(lambda x: 'extrovertido' if x==0 else ('conservadores' if x==1 else ('Deportistas' if x==2 else 'introvertido')))

id.to_csv("resultados.csv")


#aws s3 cp "resultados.csv" "s3://proyecto-data/data2//2023-02-22-19-20-28/"









