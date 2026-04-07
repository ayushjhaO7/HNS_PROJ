# Capstone Project 5 Report: DevOps for AI Deployment
**Topic:** Implementation of Containerized AI Services for Crime Analysis  
**Frameworks utilized:** PySpark (AI Engine), FastAPI (Serving API), Streamlit (Dashboard), Docker (Containerization)

---

## 1. Introduction
This project fulfills the requirements for **Unit V: DevOps for AI**. The objective is to demonstrate the principles of "AI-as-a-Service" (AIaaS) by taking a distributed machine learning model (trained in Unit IV) and deploying it into a production-grade infrastructure using DevOps methodologies.

## 2. Methodology & Implementation

### 2.1 Containerization (The Core of DevOps)
We utilized **Docker** to containerize the application. This ensures "Environment Consistency"—meaning the AI model will perform exactly the same whether it is being run on a local laptop, on-premise server, or in the cloud (AWS/Azure). 
The `Dockerfile` handles:
- **Baseline OS:** Debian-slim.
- **Dependencies:** Automated installation of OpenJDK 11 and Python 3.8.
- **Packaging:** Encapsulating the Spark Model and FastAPI code into a single immutable image.

### 2.2 AI Model Serving (FastAPI)
Instead of manually executing scripts, we built a **REST API** using FastAPI. This represents the "Deployment" phase of the AI lifecycle. 
- The model is loaded into memory on startup.
- The system exposes a POST/GET endpoint that allows external users or front-end applications to query the "Risk Level" of any specific Indian State in real-time.

### 2.3 Continuous Monitoring & Visualization (Streamlit Dashboard)
To complete the "Feedback Loop" of DevOps, we built an interactive **Spark Dashboard** using Streamlit. 
- The dashboard communicates with the API to pull AI results.
- It provides live status checks (Monitoring) of the AI service.
- It visualizes historical data alongside real-time AI predictions.

### 2.4 DevOps Simulation (The Build Pipeline)
A `setup_devops.bat` script was created to simulate a "CI/CD Pipeline." This script automates:
1. **Training:** Retraining the AI model on new crime data.
2. **Serving:** Starting the API server.
3. **Deployment:** Running the user-facing dashboard.

## 3. Results
The implementation of the `U5/` project folder successfully demonstrates:
- **Scalability:** The AI system can now be scaled horizontally using Docker containers.
- **Accessibility:** The Crime Analysis AI is no longer a static notebook; it is a live web service reachable via HTTP.
- **Monitoring:** The health of the distributed cluster is monitored through the dashboard UI.

## 4. Conclusion
By applying DevOps principles to the Spark Crime Analysis project, we have transitioned from "Experimental Machine Learning" (Notebooks) to "Production AI Deployment" (APIs and Containers). This fulfills the Unit V requirement of developing a project focused on the lifecycle management and deployment of open-source AI frameworks.
