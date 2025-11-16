import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const apiKey = process.env.ANTHROPIC_API_KEY
    if (!apiKey) {
      return NextResponse.json({ error: 'Missing ANTHROPIC_API_KEY on server' }, { status: 500 })
    }

    const res = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Anthropic accepts either an Authorization Bearer token or x-api-key depending on setup.
        // Adjust header below if your account requires a different header.
        'x-api-key': apiKey,
      },
      body: JSON.stringify(body),
    })

    const data = await res.text()
    // Return the proxied response body (raw) and preserve status
    return new NextResponse(data, { status: res.status, headers: { 'Content-Type': res.headers.get('content-type') || 'application/json' } })
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 500 })
  }
}
