# ──────────────────────────────────────────────────────────
# DEPLOY.md — Step-by-step cloud deployment guide
# ──────────────────────────────────────────────────────────

# Step 1: Get a free Groq API key
# → https://console.groq.com → Sign Up → API Keys → Create key
# Copy the key starting with "gsk_..."

# Step 2: Create free Supabase project
# → https://supabase.com → New Project
# Copy: Project URL + anon/service key from Settings → API
# Run the SQL in chatbot/db/migrations.sql in the Supabase SQL Editor

# Step 3: Push code to GitHub
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main

# Step 4: Deploy to Render.com (free)
# → https://render.com → New → Web Service → Connect GitHub repo
# Set environment variables in Render dashboard:
#   GROQ_API_KEY    = your Groq key
#   SUPABASE_URL    = your Supabase project URL
#   SUPABASE_KEY    = your Supabase service role key
#   ADMIN_SECRET    = a strong password for /admin
# Deploy → wait ~3 min for first build

# Step 5: Access your app
# Chat UI  : https://your-app.onrender.com/
# Admin UI : https://your-app.onrender.com/admin
# API Docs : https://your-app.onrender.com/docs

# Step 6: Onboard clients via Admin UI
# Go to /admin → Onboard Client tab
# Fill in Client ID, Business Name, Tone
# Add website URL and/or upload PDFs
# Click "Start Onboarding"
# The frontend dropdown updates automatically!

# Optional: Keep Render warm (prevents 30s cold starts)
# → https://uptimerobot.com → Free account
# → Add HTTP monitor for https://your-app.onrender.com/health
# → Set interval: 5 minutes
# This keeps the server alive 24/7 for free.
