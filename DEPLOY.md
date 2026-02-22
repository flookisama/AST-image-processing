# Deploy Antibiogram Reader (use on iPhone)

Deploy the app to **Streamlit Community Cloud** so you can open it in Safari on your iPhone (or any device).

## 1. Push your code to GitHub

If the project is not yet on GitHub:

```bash
cd /Users/anouarakhssas/Documents/GitHub/AST-image-processing
git add .
git commit -m "Add Streamlit app and deployment config"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AST-image-processing.git
git push -u origin main
```

(Replace `YOUR_USERNAME` with your GitHub username. Create the repo on GitHub first if needed.)

## 2. Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your **GitHub** account
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/AST-image-processing`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

After a few minutes you’ll get a URL like:

**https://ast-image-processing-xxxxx.streamlit.app**

Open this URL on your iPhone (Safari or Chrome) to use the app. You can take a photo with the camera or upload an image.

## 3. (Optional) Use EUCAST breakpoints from Excel in the cloud

The app works without the Excel file (built-in breakpoints are used). To use the full EUCAST v16 table in the deployed app:

1. Create a `data` folder in the project
2. Copy your file into it:  
   `data/v_16.0__BreakpointTables.xlsx`
3. Commit and push:

   ```bash
   git add data/
   git commit -m "Add EUCAST breakpoints Excel"
   git push
   ```

Streamlit will redeploy automatically and the app will load breakpoints from this file.

## 4. Alternative: use on iPhone from your Mac (no cloud)

If you only want to use the app when your Mac is on the same Wi‑Fi as your iPhone:

1. On your Mac, run:
   ```bash
   cd /Users/anouarakhssas/Documents/GitHub/AST-image-processing
   source .venv/bin/activate
   streamlit run app.py --server.address 0.0.0.0
   ```
2. Note your Mac’s IP (e.g. System Settings → Network → Wi‑Fi → Details).
3. On your iPhone, open: **http://YOUR_MAC_IP:8501** (e.g. `http://192.168.1.10:8501`).

This only works on the same local network.
