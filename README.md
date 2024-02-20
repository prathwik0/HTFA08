# Team Code Commandos

- This repository contains our python and machine learning code as well as docker containers uploaded to GCP.
- [The front end repository : real-incognito](https://github.com/prathwik0/real-incognito)
- [The Documentation repository : real-incognito-docs](https://github.com/prathwik0/real-incognito)
- The main application is hosted at : https://real-incognito.vercel.app
- The documentation is hosted at : https://real-incognito-docs.vercel.app

## Members

- Kausthubh J Rao
- Prathwik Kushal Kumar
- Pratham Kadekar
- Shaun Dsouza

---

## Writing and Deploying the Docker containers

### 1. Writing the app

- The code to build, train, and save the model is in the `test` folder.
- Implement the app in `main.py`

### 2. Setup Google Cloud

- Create new project
- Activate Cloud Run API and Cloud Build API

### 3. Install and init Google Cloud SDK

- https://cloud.google.com/sdk/docs/install

### 4. Dockerfile

### 5. Cloud build & deploy

```
gcloud builds submit --tag gcr.io/<project_id>/<function_name>
gcloud run deploy --image gcr.io/<project_id>/<function_name> --platform managed
```
