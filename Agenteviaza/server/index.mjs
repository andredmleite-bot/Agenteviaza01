import "dotenv/config"
import express from "express"

const app = express()
const PORT = process.env.PORT || 8081

app.use(express.json())

const sessions = new Map()
const sessionState = new Map()
const sessionLocks = new Map()

async function acquireLock(sessionId){ while (sessionLocks.has(sessionId)) { await new Promise(r=>setTimeout(r,50)) } sessionLocks.set(sessionId,true) }
function releaseLock(sessionId){ sessionLocks.delete(sessionId) }

function getSessionState(sessionId){ return sessionState.get(sessionId) || {} }
function setSessionState(sessionId, state){ sessionState.set(sessionId, state) }

app.post('/webhook/evo', async (req,res)=>{
  try{ const { text, number } = req.body || {}; console.log(`📱 [${number}] "${text}"`); return res.json({ ok:true }) } catch(err){ console.error('Erro webhook:', err?.message || String(err)); return res.json({ ok:true }) }
})

app.post('/api/chat', async (req,res)=>{
  try{ const { sessionId, message } = req.body || {}; console.log(` [${sessionId}] "${message}"`); return res.json({ reply: 'Bot em construção...' }) } catch(err){ console.error('Erro chat:', err?.message || String(err)); return res.status(500).json({ reply: 'Erro interno' }) }
})

app.get('/health', (req,res)=>{ res.json({ status:'ok', timestamp: new Date().toISOString(), 'sessões_ativas': sessions.size }) })

app.listen(PORT, ()=>{
  console.log(`\n Servidor iniciado em http://localhost:${PORT}`)
  console.log(' Rotas disponíveis:')
  console.log('    POST /webhook/evo')
  console.log('    POST /api/chat')
  console.log('    GET  /health\n')
})

export default app
