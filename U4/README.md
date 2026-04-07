# 📄 Technical Documentation: Crime Analysis in India (PySpark)

This repository contains the Apache Spark integration for the "Crime Analysis in India" distributed database and AI capstone project. 

## 🏗️ Project Architecture
Traditional data analysis scripts run on a single thread constraint. This project utilizes the **Apache Spark** computing architecture, dividing the workload across a local Spark cluster.
1. **Data Lake / Storage**: Static `.csv` assets located in the parent directory (`../crime---analysis/data/`).
2. **Spark SQL Engine**: Interrogates the resilient distributed datasets (RDDs) converting raw records into grouped insights (Year, Crime Type, etc).
3. **PySpark MLlib**: Conducts automatic Artificial Intelligence operations. It extracts the top 5 crime features and performs an unsupervised K-Means mapping mapping states into similar categories.

## 📁 File Structure
```text
/U4/
│
├── Crime_Analysis_Spark.ipynb   # Main Jupyter Notebook containing the Spark execution cells
├── requirements.txt             # Python dependencies (NumPy, PySpark, Pandas)
├── Capstone_Project_Report.md   # The academic capstone report
└── README.md                    # Technical documentation (You are here)
```

## ⚙️ Prerequisites
- Python 3.8+
- Active Java installation (Required by Apache Spark's JVM under the hood). *Note: PySpark will attempt to use your default Java path.*

## 🚀 Installation & Setup
1. **Navigate to the Project Folder**:
   Ensure your terminal or command prompt is located inside the `U4` directory.
   ```bash
   cd ./U4
   ```

2. **Install Python Dependencies**:
   Install the required frameworks (PySpark, etc) utilizing pip.
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the Application
The project is built as an interactive Jupyter Notebook. There is no background sever required if you are using an IDE such as VS Code.

### Using VS Code or PyCharm:
1. Double-click to open `Crime_Analysis_Spark.ipynb` inside your IDE.
2. Select your local Python environment as the active Kernel.
3. Click **"Run All"** to execute the pipeline. 

### Output Expectations:
As the notebook executes, you will view terminal logs representing Spark spawning the execution threads: `Spark Session Initialized...`.
- The **SQL Cells** will output truncated dataframes representing crime counts.
- The **AI Cell** will dynamically compute cluster sizes and output a final mapped classification. It provides distinct printouts isolating:
  - `=== HIGH RISK STATES ===`
  - `=== MEDIUM RISK STATES ===`
  - `=== LOW RISK STATES ===`
This ensures the final output is immediately human-readable and ready for presentation.

## 🛠 Troubleshooting
**`ModuleNotFoundError: No module named 'pyspark'`**
Ensure your IDE environment is using the actual underlying Python installation where `requirements.txt` was loaded. In VS Code, click the Python version in the top-right corner of the notebook and switch to the correct `python.exe`.
