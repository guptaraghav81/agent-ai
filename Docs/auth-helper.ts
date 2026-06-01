// lib/auth-helper.ts
// Thin abstraction over whatever auth system the app uses.
// TODAY: returns a placeholder — replace the body of getUserId() with your
//        actual auth call once you confirm the system with the team.
//
// NextAuth example:
//   import { getServerSession } from 'next-auth'
//   import { authOptions } from '@/lib/auth'
//   const session = await getServerSession(authOptions)
//   return session?.user?.id ?? null
//
// Firebase Auth (via session cookie) example:
//   import { adminAuth } from '@/lib/firebase-admin'
//   const cookie = req.cookies.get('__session')?.value
//   if (!cookie) return null
//   const decoded = await adminAuth.verifySessionCookie(cookie, true)
//   return decoded.uid

import { NextRequest } from 'next/server'

export async function getUserId(req: NextRequest): Promise<string | null> {
  // ── TEMPORARY: accept userId from request header for development ──────────
  // Replace this entire block with your real auth check before going to prod.
  const devUserId = req.headers.get('x-user-id')
  if (devUserId) return devUserId

  // ── PRODUCTION: uncomment ONE of the blocks below ─────────────────────────

  // Option A — NextAuth
  // import { getServerSession } from 'next-auth'
  // import { authOptions } from '@/lib/auth'
  // const session = await getServerSession(authOptions)
  // return session?.user?.id ?? null

  // Option B — Firebase Auth session cookie
  // import { adminAuth } from '@/lib/firebase-admin'
  // const cookie = req.cookies.get('__session')?.value
  // if (!cookie) return null
  // try {
  //   const decoded = await adminAuth.verifySessionCookie(cookie, true)
  //   return decoded.uid
  // } catch { return null }

  return null
}
