# Utiliser une image de base pour Spark
FROM openjdk:8

# Installer PySpark
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install pyspark

# Copier le driver PostgreSQL dans le conteneur
ADD https://jdbc.postgresql.org/download/postgresql-42.2.23.jar /opt/spark/jars/

# Copier le script de votre application
COPY etl_pipeline_db.py /app/etl_pipeline_db.py

# Définir le point d'entrée du conteneur
CMD ["python3", "/app/etl_pipeline_db.py"]
