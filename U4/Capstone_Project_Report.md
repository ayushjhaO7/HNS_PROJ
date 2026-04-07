# Capstone Project 4 Report: Distributed Databases for AI
**Topic:** Crime Analysis in India Utilizing Apache Spark  
**Framework utilized:** PySpark (Apache Spark API), Spark MLlib, Spark SQL  

---

## 1. Introduction
This project fulfills the requirements for **Unit IV: Distributed Databases for AI**. The objective of this capstone is to introduce and apply open-source distributed frameworks to real-world datasets, integrating Artificial Intelligence (AI) and Machine Learning capabilities. 

This project serves as an extension of a baseline "Crime Analysis in India" project (originally built using R and Power BI). To demonstrate scalability, the data pipeline was successfully transposed into a distributed compute ecosystem using **Apache Spark**.

## 2. Technology Stack
- **Apache Spark (PySpark):** The core distributed computing framework used to partition data and execute operations in parallel.
- **Spark SQL:** Used to perform distributed analytical querying over massive datasets via temporary data views.
- **Spark MLlib:** The scalable machine learning library used to fulfill the AI requirement.
- **Jupyter Notebook environment:** Used to interactively visualize and sequence the data processing steps.

## 3. Methodology & Implementation

### 3.1 Distributed Data Ingestion
The dataset, `cleaned_crime_data.csv`, containing detailed logs of crimes grouped by State/UT, District, and Year, was injested into a **Spark DataFrame**. Unlike traditional pandas dataframes, Spark dataframes support resilient distributed datasets (RDDs), meaning the data was correctly partitioned across the simulated nodes of the driver environment, ensuring scalability for much larger, gigabyte-scale datasets.

### 3.2 Distributed Analytical SQL Queries
Using `createOrReplaceTempView`, the system exposed the distributed DataFrame to Spark's robust SQL engine. We executed large-scale aggregations to retrieve:
1. **Year-wise Crime Trends:** Aggregating total crimes year over year.
2. **Crime Categorization:** Sorting and grouping the dataset to identify the Top 10 most prominent crime types across India.
3. **State-level Density:** Filtering the states experiencing the highest volume of offenses natively.

### 3.3 Artificial Intelligence Component (K-Means Clustering)
To satisfy the "AI" metric of the syllabus, we implemented an automated pattern-recognition algorithm to classify States and Union Territories into "Risk Clusters" using PySpark's scalable Machine Learning library (`pyspark.ml`).

- **Feature Engineering:** We pivoting the state data so that the top 5 specific crime types became independent numerical columns. 
- **Vectorization & Scaling:** We utilized `VectorAssembler` to compile these columns into a unified feature vector, and utilized `StandardScaler` to normalize the data distributions (ensuring no single category overpowered the distance metrics).
- **K-Means Clustering:** A distributed `KMeans` model was configured with $k=3$ (representing hypothetical Low, Medium, and High similarity domains). The model was trained against the distributed state vectors, autonomously assigning a cluster label to every region based purely on the variance in its specific crime breakdown.

## 4. Results
The execution of the `Crime_Analysis_Spark.ipynb` pipeline confirmed the following:
- Spark successfully initialized, mapped the CSV schema types seamlessly, and removed invalid/empty structures.
- The SQL results structurally matched the baseline visual assumptions from the R project but computed via MapReduce concepts under the hood.
- The AI K-Means algorithm successfully deduced logical severity layers based strictly on cluster centroids, categorizing every state into three precise risk levels automatically:
  - **High Risk States:** Identified as states with massive total volumes in key crimes (e.g., Madhya Pradesh, Maharashtra, Andhra Pradesh).
  - **Medium Risk States:** States with significant but slightly lower categorical density (e.g., West Bengal, Rajasthan, Gujarat, Bihar, Uttar Pradesh).
  - **Low Risk States:** Lower overall frequency regions encompassing the majority of smaller states and UTs (e.g., Sikkim, Goa, Chandigarh).

## 5. Conclusion
This project successfully demonstrates the transition from traditional, localized data analytics (like R/Shiny) to a **Distributed Database** architecture. By utilizing Apache Spark, we showcased that large-scale demographic and crime analysis can run infinitely faster across clustered hardware. Furthermore, by seamlessly integrating Spark MLlib, the analysis transitioned from purely historical reporting into an automated Artificial Intelligence pipeline.
