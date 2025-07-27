# Split Deployment Guide

## Backend Deployment (Railway/Render)

### Option A: Railway (Recommended)
1. Push code to GitHub
2. Connect Railway to your repo
3. Railway auto-detects `railway.toml` and `Dockerfile`
4. Set environment variables in Railway dashboard
5. Deploy automatically

### Option B: Render
1. Push code to GitHub  
2. Create new Web Service on Render
3. Connect your repo
4. Render auto-detects `render.yaml`
5. Set environment variables in Render dashboard
6. Deploy

### Required Environment Variables:
```
OPENAI_API_KEY=your_openai_key
```

## Frontend Deployment (Streamlit Cloud)

1. Push code to GitHub (separate repo or same repo)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file path: `app.py`
5. Add secrets in Streamlit Cloud:
   ```
   API_URL = "https://your-backend-url.railway.app"
   ```
6. Deploy

## Post-Deployment

1. Update CORS origins in `api.py:line_30` with your Streamlit Cloud URL
2. Test the connection between frontend and backend
3. Monitor logs for any issues

## Local Testing
```bash
# Backend
uvicorn api:app --reload

# Frontend (new terminal)
streamlit run app.py
```