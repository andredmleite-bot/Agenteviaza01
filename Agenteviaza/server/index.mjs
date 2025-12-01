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

function stripDiacritics(s){ return String(s||"").normalize('NFD').replace(/[\u0300-\u036f]/g,'') }
function normalizeText(s){ return stripDiacritics(String(s||"").toLowerCase()).replace(/[^a-z0-9\s]/g,' ').replace(/\s+/g,' ').trim() }
function pick(a){ return a[Math.floor(Math.random()*a.length)] }
function parseIntSafe(v){ const n = parseInt(String(v),10); return Number.isNaN(n)?undefined:n }

const OFFICIAL_IATA = new Set(['CNF','PLU','SAO','GRU','CGH','RIO','GIG','SDU','BSB','SSA','REC','POA','VIX','FOR','MAO','BEL','VCP','CWB','FLN','NAT'])
const IATA_ALIASES = [
  { code:'SAO', aliases:['sao paulo','são paulo','sp'] },
  { code:'GRU', aliases:['gru','guarulhos','guarulho','aeroporto de guarulhos'] },
  { code:'CGH', aliases:['cgh','congonhas','aeroporto de congonhas'] },
  { code:'RIO', aliases:['rio de janeiro','rio','rj'] },
  { code:'GIG', aliases:['gig','galeao','galeão','tom jobim','antonio carlos jobim'] },
  { code:'SDU', aliases:['sdu','santos dumont','santo dumont'] },
  { code:'CNF', aliases:['cnf','belo horizonte','bh','confins'] },
  { code:'PLU', aliases:['plu','pampulha','belo horizonte pampulha'] },
  { code:'BSB', aliases:['bsb','brasilia','brasília'] },
  { code:'SSA', aliases:['ssa','salvador','bahia'] },
  { code:'REC', aliases:['rec','recife'] },
  { code:'POA', aliases:['poa','porto alegre'] },
  { code:'VIX', aliases:['vix','vitoria','vitória'] },
  { code:'FOR', aliases:['for','fortaleza'] },
  { code:'MAO', aliases:['mao','manaus'] },
  { code:'BEL', aliases:['bel','belem','belém'] },
  { code:'VCP', aliases:['vcp','campinas','viracopos'] },
  { code:'CWB', aliases:['cwb','curitiba'] },
  { code:'FLN', aliases:['fln','florianopolis','florianópolis','floripa'] },
  { code:'NAT', aliases:['nat','natal'] }
]

function resolveIATA(input){
  const raw = String(input||"").trim()
  if (/^[a-z]{3}$/i.test(raw)) { const c = raw.toUpperCase(); return OFFICIAL_IATA.has(c)?c:undefined }
  const s = normalizeText(raw)
  for (const e of IATA_ALIASES){
    for (const a of e.aliases){
      const w = normalizeText(a)
      const re = new RegExp(`(^|\b)${w.replace(/[-/\\^$*+?.()|[\]{}]/g,'\\$&')}(\b|$)`,`i`)
      if (re.test(s)) return e.code
    }
  }
  return undefined
}

function iataToViazaSlug(code){ const c = String(code||"").toUpperCase(); if (c==='CNF'||c==='PLU') return 'bhz'; if (c==='SAO') return 'sao'; if (c==='RIO') return 'rio'; return c.toLowerCase() }
function toISO(d){ const y=d.getFullYear(), m=String(d.getMonth()+1).padStart(2,'0'), da=String(d.getDate()).padStart(2,'0'); return `${y}-${m}-${da}` }
function formatISOToBR(iso){ const m=String(iso||"").match(/^(\d{4})-(\d{2})-(\d{2})$/); if(!m) return iso; return `${m[3]}/${m[2]}/${m[1]}` }
function within365(iso){ const now = new Date(); const base = new Date(now.getFullYear(), now.getMonth(), now.getDate()); const t = new Date(iso); const tgt = new Date(t.getFullYear(), t.getMonth(), t.getDate()); const diff = Math.floor((tgt-base)/(1000*60*60*24)); return diff>=0 && diff<=365 }

function parseDates(text){
  const s = normalizeText(text)
  const iso = s.match(/\b(\d{4})-(\d{2})-(\d{2})\b/g) || []
  if (iso.length>=1){ const dpt=iso[0]; let dst=null; if (iso.length>=2) dst=iso[1]; return { dpt, dst, ow: !dst } }
  const dm = Array.from(s.matchAll(/\b(\d{1,2})\/(\d{1,2})\b/g))
  if (dm.length>=1){ const d1=dm[0]; const dd=String(parseIntSafe(d1[1])).padStart(2,'0'); const mm=String(parseIntSafe(d1[2])).padStart(2,'0'); let yyyy = new Date().getFullYear(); const cand = new Date(yyyy, parseInt(mm,10)-1, parseInt(dd,10)); const today = new Date(); if (cand < today) yyyy += 1; const dpt = `${yyyy}-${mm}-${dd}`; let dst=null; if (dm.length>=2){ const d2=dm[1]; const dd2=String(parseIntSafe(d2[1])).padStart(2,'0'); const mm2=String(parseIntSafe(d2[2])).padStart(2,'0'); let y2=yyyy; if (parseInt(mm2,10) < parseInt(mm,10)) y2 += 1; dst = `${y2}-${mm2}-${dd2}` }
    return { dpt, dst, ow: !dst }
  }
  return null
}

function parsePassengers(text){
  const s = normalizeText(text)
  if (/\b(so eu|só eu|apenas eu|somente eu)\b/.test(s)) return { adt:1, chd:0, bby:0 }
  let adt=undefined, chd=0, bby=0
  const mAdt = s.match(/\b(\d+)\s*adult/)
  const mChd = s.match(/\b(\d+)\s*crian/)
  const mBby = s.match(/\b(\d+)\s*beb/)
  if (mAdt) adt = parseIntSafe(mAdt[1])
  if (mChd) chd = parseIntSafe(mChd[1])||0
  if (mBby) bby = parseIntSafe(mBby[1])||0
  if (adt==null) return null
  const total = (adt||0)+(chd||0)+(bby||0)
  if (total<1 || total>9) return { error:'limit' }
  return { adt: adt||1, chd: chd||0, bby: bby||0 }
}

function isGreeting(text){ const s=normalizeText(text); return /\b(oi|ola|olá|hey|eae|salve|bom dia|boa tarde|boa noite)\b/.test(s) }
function isConfirmation(text){ const s=normalizeText(text); return /(ok|certo|isso|sim|pode|segue|manda|fechar|confirmo|confirmar)/.test(s) }

function buildLink(state){
  const dep = String(state.dep||"").toUpperCase()
  const des = String(state.des||"").toUpperCase()
  const dpt = String(state.dpt||"")
  const dst = state.dst?String(state.dst):null
  const adt = state.adt||1, chd = state.chd||0, bby = state.bby||0
  const trip = dst? 'rt' : 'ow'
  const url = `https://www.viaza.com.br/busca/passagens/${trip}/${iataToViazaSlug(dep)}/${iataToViazaSlug(des)}?p=${adt}-${chd}-${bby}&dd=${dpt}${dst?`&rd=${dst}`:''}`
  return url
}

function mergeState(sid, patch){ const cur = getSessionState(sid); const next = { ...cur, ...patch }; setSessionState(sid, next); return next }
function complete(s){ return s.dep && s.des && s.dpt && ((s.adt||0)+(s.chd||0)+(s.bby||0)>0) }

const MSG = {
  greet: [
    'Oi! Bora agilizar sua passagem ✈️ Me diz: de onde pra onde?',
    'Olá! Vamos começar sua cotação? Origem e destino, por favor 🙌',
    'E aí! Me conta o trecho: cidade de saída e cidade de chegada 😉'
  ],
  confirmOD: (dep, des)=> pick([
    `Perfeito! 🎯 De ${dep} para ${des}, certo? Qual a data da viagem?`,
    `Ótimo, achei ${dep} → ${des}. Me fala a data pra eu seguir 📅`,
    `${dep} indo pra ${des}, beleza? Qual o dia do voo?`
  ]),
  askDate: pick([
    'Qual a data da ida? Pode ser em formato 12/01 ou 2025-01-12 📅',
    'Me diz o dia da ida (ex.: 15/02) 🗓️'
  ]),
  badDate: pick([
    'Hmm, essa data não rola. Aceito de hoje até 365 dias. Manda outra 📅',
    'Ops, data fora do período. Tente uma entre hoje e 1 ano 😉'
  ]),
  confirmDate: (dpt, dst)=>{ const ida = formatISOToBR(dpt); const volta = dst? formatISOToBR(dst): null; return volta? pick([`Show! Ida ${ida} e volta ${volta} ✨ Quantas pessoas vão?`,`Anotado: ${ida} → ${volta}. Agora, quantas pessoas?`]) : pick([`Anotado: ida ${ida} ✅ Quantas pessoas vão?`,`Boa! Ida ${ida}. Me diz o total de passageiros 👨‍👩‍👧‍👦`]) },
  askPax: pick([
    'Quantas pessoas e a composição? Ex.: 2 adultos e 1 criança',
    'Me fala o total e idades: adultos, crianças (2–12) e bebês (0–2)'
  ]),
  paxLimit: pick([
    'Puts, passamos de 9 😅 A cotação aceita até 9. Quer ajustar o total?',
    'Até 9 passageiros por vez, tá? Me diz quantos você quer cotar'
  ]),
  finalConfirm: (s)=>{ const ida = formatISOToBR(s.dpt); const volta = s.dst? ` e volta ${formatISOToBR(s.dst)}`:''; const comp = [`${s.adt||0} adulto(s)`, (s.chd||0)>0?`${s.chd} criança(s)`:null, (s.bby||0)>0?`${s.bby} bebê(s)`:null].filter(Boolean).join(', '); return pick([
    `Fechamos assim: ${s.dep} → ${s.des}, ida ${ida}${volta}. Passageiros: ${comp}. Posso gerar? ✨`,
    `Resumo rápido: ${s.dep} a ${s.des}, ${ida}${volta}. ${comp}. Confirmo pra criar o link?`
  ]) },
  link: (url)=> pick([
    `Pronto! 🎉 Segue sua cotação → ${url}`,
    `Tudo certo! Aqui está seu link de busca: ${url}`,
    `Maravilha! Abra sua pesquisa: ${url}`
  ])
}

async function handleMessage(sessionId, text){
  let s = getSessionState(sessionId)
  const raw = String(text||"")
  if (!s.step && isGreeting(raw)) return { reply: pick(MSG.greet), state: s }
  const iata = raw.match(/(?:de\s+)?([a-zA-Zãáâàêéíóôõúç\s]+?)\s+(?:para|pra)\s+([a-zA-Zãáâàêéíóôõúç\s]+)(?:\s|$)/i)
  if (!s.dep || !s.des){
    if (iata){
      const depCity = iata[1].trim(); const desCity = iata[2].trim()
      let dep = resolveIATA(depCity); let des = resolveIATA(desCity)
      if (!dep && /sao paulo|são paulo|\bsp\b/i.test(depCity)) dep='SAO'
      if (!des && /sao paulo|são paulo|\bsp\b/i.test(desCity)) des='SAO'
      if (!dep && /rio de janeiro|\brio\b|\brj\b/i.test(depCity)) dep='RIO'
      if (!des && /rio de janeiro|\brio\b|\brj\b/i.test(desCity)) des='RIO'
      if (dep && des){ s = mergeState(sessionId, { dep, des }); return { reply: MSG.confirmOD(dep, des), state: s }
    }
  }
  const dates = parseDates(raw)
  if (!s.dpt && dates){ const { dpt, dst } = dates; if (!within365(dpt) || (dst && !within365(dst))) return { reply: MSG.badDate, state: s }; s = mergeState(sessionId, { dpt, dst, ow: !dst }); return { reply: MSG.confirmDate(dpt, dst), state: s }
  if (!s.dpt && s.dep && s.des){ return { reply: MSG.askDate, state: s } }
  const pax = parsePassengers(raw)
  if (!s.adt && pax){ if (pax.error==='limit') return { reply: MSG.paxLimit, state: s }; s = mergeState(sessionId, { adt: pax.adt, chd: pax.chd, bby: pax.bby }); return { reply: MSG.finalConfirm(s), state: s }
  if (s.dpt && (!s.adt || !s.chd && !s.bby)){ return { reply: MSG.askPax, state: s } }
  if (complete(s) && isConfirmation(raw)){ const url = buildLink(s); const reply = MSG.link(url); sessionState.delete(sessionId); return { reply, state: {} }
  if (!s.dep || !s.des) return { reply: pick(MSG.greet), state: s }
  return { reply: MSG.finalConfirm(s), state: s }
}

app.post('/webhook/evo', async (req,res)=>{
  try{ const { text, number } = req.body || {}; if (!text || !number) return res.json({ ok:true }); await acquireLock(number); try{ const r = await handleMessage(number, text); return res.json({ ok:true }) } finally { releaseLock(number) } } catch(err){ return res.json({ ok:true }) }
})

app.post('/api/chat', async (req,res)=>{
  try{ const { sessionId, message } = req.body || {}; if (!sessionId || !message) return res.status(400).json({ error:'sessionId e message' }); const r = await handleMessage(sessionId, message); return res.json({ reply: r.reply }) } catch(err){ return res.status(500).json({ reply:'Erro interno' }) }
})

app.get('/health', (req,res)=>{ res.json({ status:'ok', timestamp:new Date().toISOString(), 'sessões_ativas': sessions.size }) })

app.listen(PORT, ()=>{ console.log(`\n Servidor iniciado em http://localhost:${PORT}`); console.log(' Rotas disponíveis:'); console.log('    POST /webhook/evo'); console.log('    POST /api/chat'); console.log('    GET  /health\n') })
