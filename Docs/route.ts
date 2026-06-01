// app/api/ask-ai/route.ts
// Server-side bridge: authenticates user → calls Python /chat → saves to Firestore → returns answer.
// Never runs in the browser. Never exposes PYTHON_AI_URL or Firebase Admin credentials.

import { NextRequest, NextResponse } from 'next/server'
import { adminDb } from '@/lib/firebase-admin'
import { FieldValue } from 'firebase-admin/firestore'
import { getUserId } from '@/lib/auth-helper'

export async function POST(req: NextRequest) {
  // ── 1. Authenticate ────────────────────────────────────────────────────────
  // getUserId() is a thin wrapper — swap the internals when you know the auth
  // system (NextAuth or Firebase Auth). See lib/auth-helper.ts.
  const userId = await getUserId(req)
  if (!userId) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  // ── 2. Parse request ───────────────────────────────────────────────────────
  let query: string, sessionId: string, history: { role: string; content: string }[]
  try {
    const body = await req.json()
    query     = (body.query     ?? '').trim()
    sessionId = body.sessionId  ?? crypto.randomUUID()
    history   = Array.isArray(body.history) ? body.history : []
  } catch {
    return NextResponse.json({ error: 'Invalid request body' }, { status: 400 })
  }

  if (!query) {
    return NextResponse.json({ error: 'Empty query' }, { status: 400 })
  }

  // ── 3. Call Python AI service ──────────────────────────────────────────────
  let answer   = ''
  let sources: string[]  = []
  let metadata: Record<string, unknown> = {}

  try {
    const aiRes = await fetch(`${process.env.PYTHON_AI_URL}/chat`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        query,
        conversation_history: history,
        user_id:    userId,
        session_id: sessionId,
      }),
    })

    if (!aiRes.ok) {
      throw new Error(`Python service returned ${aiRes.status}`)
    }

    const data = await aiRes.json()
    answer   = data.answer   ?? ''
    sources  = data.sources  ?? []
    metadata = data.metadata ?? {}

  } catch (err) {
    console.error('[ask-ai] Python call failed:', err)
    return NextResponse.json({ error: 'AI service unavailable' }, { status: 502 })
  }

  // ── 4. Save to Firestore ───────────────────────────────────────────────────
  // Errors here are non-fatal — the user still gets their answer.
  try {
    const sessionRef = adminDb
      .collection('conversations')
      .doc(userId)
      .collection('sessions')
      .doc(sessionId)

    // Upsert session document (creates on first message)
    await sessionRef.set(
      { updatedAt: FieldValue.serverTimestamp(), userId },
      { merge: true }
    )

    const msgCol = sessionRef.collection('messages')

    await msgCol.add({
      role:      'user',
      content:   query,
      timestamp: FieldValue.serverTimestamp(),
    })

    await msgCol.add({
      role:      'assistant',
      content:   answer,
      sources,
      metadata,
      timestamp: FieldValue.serverTimestamp(),
    })

  } catch (err) {
    // Log but do not fail the request — answer is more important than storage
    console.error('[ask-ai] Firestore write failed:', err)
  }

  // ── 5. Return to browser ───────────────────────────────────────────────────
  return NextResponse.json({ answer, sources, sessionId })
}
