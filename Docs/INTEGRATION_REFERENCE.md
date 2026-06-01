# SF360 AskAI Integration — Environment Variables & Deployment Reference

---

## Python Backend (.env on Render)

No new variables needed. Existing setup is sufficient.
The new POST /chat endpoint uses the same GROQ_API_KEY already set.

```
GROQ_API_KEY=<already set on Render>
```

Confirm CORS is open (allow_origins=["*"] — already the case in agent.py).

---

## Next.js App (.env.local for dev / Vercel env vars for prod)

### Python AI Service
```
PYTHON_AI_URL=https://askai-1flm.onrender.com
```
No trailing slash. The API route appends /chat directly.

### Firebase Admin SDK  (server-side — never prefix with NEXT_PUBLIC_)
Get these from: Firebase Console → Project Settings → Service Accounts → Generate new private key
```
FIREBASE_PROJECT_ID=<your-project-id>
FIREBASE_CLIENT_EMAIL=firebase-adminsdk-xxx@<your-project-id>.iam.gserviceaccount.com
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
```
⚠️  Paste the FIREBASE_PRIVATE_KEY value in double quotes.
    The \n characters must be literal backslash-n — Next.js converts them at runtime.

### Firebase Client SDK  (browser-safe — NEXT_PUBLIC_ prefix required)
Get these from: Firebase Console → Project Settings → General → Your apps → Config
```
NEXT_PUBLIC_FIREBASE_API_KEY=AIza...
NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=<project>.firebaseapp.com
NEXT_PUBLIC_FIREBASE_PROJECT_ID=<project>
NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=<project>.appspot.com
NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=<number>
NEXT_PUBLIC_FIREBASE_APP_ID=1:<number>:web:<hex>
```

### Auth (add when confirmed with team)
```
# NextAuth — if using NextAuth:
NEXTAUTH_URL=https://sportsfan-frontend.vercel.app
NEXTAUTH_SECRET=<generate with: openssl rand -base64 32>
```

---

## File Placement in Next.js Repo

```
<nextjs-root>/
├── app/
│   └── api/
│       └── ask-ai/
│           └── route.ts          ← from outputs/route.ts
├── components/
│   └── AskAI.tsx                 ← from outputs/AskAI.tsx
│                                    (keep AskSF360.css — same class names)
├── lib/
│   ├── firebase-admin.ts         ← from outputs/firebase-admin.ts
│   ├── firebase.ts               ← from outputs/firebase.ts
│   └── auth-helper.ts            ← from outputs/auth-helper.ts
└── .env.local                    ← add variables above
```

---

## Python Backend Deployment

1. Replace agent.py in the repo with outputs/agent.py
2. Commit and push:
   ```
   git config core.autocrlf false
   git add -f agent.py
   git commit -m "feat: add POST /chat endpoint for Next.js integration"
   git push
   ```
3. Render auto-deploys on push. Monitor the deploy log for startup errors.
4. Verify the new endpoint:
   ```
   curl -X POST https://askai-1flm.onrender.com/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "Most IPL runs", "conversation_history": [], "user_id": "test"}'
   ```
   Expected: { "answer": "...", "sources": [], "metadata": { "chart_title": "...", "chart_data": [...] } }

---

## Auth Placeholder (resolve before going live)

auth-helper.ts currently accepts an x-user-id request header for development.
This is fine locally — it means unauthenticated calls work during dev/testing.

Before deploying to production:
1. Confirm auth system with frontend team (NextAuth or Firebase Auth)
2. Uncomment the appropriate block in lib/auth-helper.ts
3. Remove the devUserId header fallback

---

## Firestore Setup

1. Create a Firestore database in the Firebase Console (production mode)
2. Deploy security rules:
   ```
   # firestore.rules
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /conversations/{userId}/{document=**} {
         allow read, write: if request.auth != null
                            && request.auth.uid == userId;
       }
       match /{document=**} {
         allow read, write: if false;
       }
     }
   }
   ```
   Deploy with: firebase deploy --only firestore:rules

3. Create a composite index (required for the message query):
   Collection: messages (inside sessions/sessionId)
   Field: timestamp — Ascending
   Firebase Console → Firestore → Indexes → Add index

---

## What's Not Yet Done (next sprint)

- Streaming (SSE) — non-streaming ships first, streaming follow-up
- User personalisation from Firebase profile (followed teams, favourite players)
- Persistent session history across page loads (store sessionId in localStorage or URL)
- Streaming conversation history to Groq (currently only last N turns from Firebase)
