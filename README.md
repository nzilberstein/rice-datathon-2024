# Notas de Facu 01-20

- Primero tubo que va al fondo: torre de destilación
    - Fondo del tubo: bh
- Más stages en el lateral, mejor PR (stages de pipe)
- Más stages en la columna, peor PR (stages de destilación)

- bin lateral length: longitud total del tubo post bottom-hole

- buena idea clusterizar por surface X y Y

# Features we're using

## Cluster Analysis
We grouped the wells by analyzing their surface X and Y coordinates. We used the KMeans algorithm to cluster the wells into 20 groups. Afterwards we used the cluster number to compute the variation with respect to the mean OilPeakRate.
Under the `/model` directory we store the KMeans model and the cluster labels for each well are stored in `data/cluster_label_features.csv`.