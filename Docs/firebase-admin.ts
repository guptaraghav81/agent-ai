// lib/firebase-admin.ts
// Server-side only. Never import this from any client component or browser code.
// Initialises the Firebase Admin SDK once and exports adminDb (Firestore).

import { initializeApp, getApps, cert } from 'firebase-admin/app'
import { getFirestore } from 'firebase-admin/firestore'

if (!getApps().length) {
  initializeApp({
    credential: cert({
      projectId:   process.env.FIREBASE_PROJECT_ID!,
      // .env stores literal \n — this converts them to real newlines
      privateKey:  process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL!,
    }),
  })
}

export const adminDb = getFirestore()
