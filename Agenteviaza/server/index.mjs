import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import { Agent, run } from '@openai/agents';
import { z } from 'zod';
import { fileURLToPath } from 'url';
import multer from 'multer';

dotenv.config({ path: fileURLToPath(new URL('./.env', import.meta.url)) });

const { OPENAI_API_KEY, PORT = 3002, OPENAI_MODEL, OPENAI_BASE_URL, OPENAI_TRANSCRIBE_MODEL, EVO_API_URL, EVO_API_KEY, EVO_INSTANCE, APP_BASE_URL } = process.env;
const TRANSCRIBE_MODEL = OPENAI_TRANSCRIBE_MODEL || 'gpt-4o-mini-transcribe';

if (!OPENAI_API_KEY) {
  console.warn('Aviso: OPENAI_API_KEY não está definido. Configure em .env ou variável de ambiente.');
}

const app = express();
app.use(cors());
app.use(express.json());
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

const sessions = new Map();
const sessionState = new Map();
const unrecognizedIATA = new Set();
const OFFICIAL_IATA = new Set([
  'BSB','CGH','GIG','SSA','FLN','POA','VCP','REC','CWB','BEL','VIX','SDU','CGB','CGR','FOR','MCP','MGF','GYN','NVT','MAO','NAT','BPS','MCZ','PMW','SLZ','GRU','LDB','PVH','RBR','JOI','UDI','CXJ','IGU','THE','AJU','JPA','PNZ','CNF','BVB','CPV','STM','IOS','JDO','IMP','XAP','MAB','CZS','PPB','CFB','FEN','JTC','MOC','SAO','RIO',
  'JFK','LGA','EWR','MIA','MCO','LAX','SFO','YYZ','YVR','YUL','EZE','AEP','SCL','MVD','MEX','CUN','LIM','BOG','PTY',
  'LHR','LGW','CDG','ORY','AMS','FRA','MUC','BER','MAD','BCN','LIS','OPO','FCO','MXP','VCE','ZRH','GVA',
  'DXB','AUH','DOH','IST','SAW','NRT','HND','SIN','BKK','DMK','DPS','SYD','MEL','CMN','RAK','JNB','CPT'
]);
const IATA_STOPLIST = new Set([
  'dia','hoje','amanha','amanhã','ida','volta','mes','mês','proximo','próximo','de','para','pra','em','ate','até','partindo','retorno','so','somente','só','apenas','com','sem'
]);

const MONTHS_PT = { janeiro:0, jan:0, fevereiro:1, fev:1, março:2, marco:2, mar:2, abril:3, abr:3, maio:4, junho:5, jun:5, julho:6, jul:6, agosto:7, ago:7, setembro:8, set:8, outubro:9, out:9, novembro:10, nov:10, dezembro:11, dez:11 };
function stripDiacritics(s){ return s.normalize('NFD').replace(/[\u0300-\u036f]/g, ''); }
function normalizeText(s){ if(!s) return ''; const noAccents=stripDiacritics(String(s)); return noAccents.toLowerCase().replace(/[^a-z0-9\s]/g,' ').replace(/\s+/g,' ').trim(); }
function levenshtein(a,b){ a=a||''; b=b||''; const m=a.length,n=b.length; if(m===0) return n; if(n===0) return m; const dp=Array.from({length:m+1},()=>new Array(n+1).fill(0)); for(let i=0;i<=m;i++) dp[i][0]=i; for(let j=0;j<=n;j++) dp[0][j]=j; for(let i=1;i<=m;i++){ for(let j=1;j<=n;j++){ const cost=a[i-1]===b[j-1]?0:1; dp[i][j]=Math.min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost); } } return dp[m][n]; }
function thresholdFor(len){ if(len<=3) return 1; if(len<=6) return 2; return 3; }
function parseNumberPt(str){ if(str==null) return undefined; const s=String(str).trim().toLowerCase(); if(/^\d+$/.test(s)) return parseInt(s,10); const map={ zero:0, um:1, uma:1, hum:1, dois:2, duas:2, tres:3, 'três':3, quatro:4, cinco:5, seis:6, sete:7, oito:8, nove:9 }; const key=stripDiacritics(s); return map[key]; }
const IATA_LEXICON = [
  { code: 'CNF', city: 'Belo Horizonte', aliases: ['bh', 'confins', 'pampulha', 'beaga'] },
  { code: 'GRU', city: 'São Paulo', aliases: ['sp', 'guarulhos', 'congonhas', 'soa paulo'] },
  { code: 'SDU', city: 'Rio de Janeiro', aliases: ['rio', 'santos dumont'] },
  { code: 'GIG', city: 'Rio de Janeiro', aliases: ['galeão', 'tom jobim', 'galeo'] },
  { code: 'BSB', city: 'Brasília', aliases: ['brasília', 'brasilia'] },
  { code: 'SSA', city: 'Salvador', aliases: ['salvador'] },
  { code: 'REC', city: 'Recife', aliases: ['recife'] },
  { code: 'FLN', city: 'Florianópolis', aliases: ['florianópolis', 'florianopolis'] },
  { code: 'POA', city: 'Porto Alegre', aliases: ['porto alegre'] },
  { code: 'VCP', city: 'Campinas', aliases: ['campinas', 'viracopos'] },
  { code: 'BEL', city: 'Belém', aliases: ['belém', 'belem'] },
  { code: 'MAO', city: 'Manaus', aliases: ['manaus'] },
  { code: 'NAT', city: 'Natal', aliases: ['natal'] },
  { code: 'FOR', city: 'Fortaleza', aliases: ['fortaleza'] },
  { code: 'CWB', city: 'Curitiba', aliases: ['curitiba'] },
  { code: 'SAO', city: 'São Paulo', aliases: ['são paulo', 'sao paulo', 'sp'] },
  { code: 'RIO', city: 'Rio de Janeiro', aliases: ['rio de janeiro', 'jjd'] },
  { code: 'CGH', city: 'São Paulo', aliases: ['congonhas', 'sp'] },
  { code: 'VIX', city: 'Vitória', aliases: ['vitoria'] },
  { code: 'GYN', city: 'Goiânia', aliases: ['goiania'] },
  { code: 'CGB', city: 'Cuiabá', aliases: ['cuiaba'] },
  { code: 'CGR', city: 'Campo Grande', aliases: ['campo grande'] },
  { code: 'MCZ', city: 'Maceió', aliases: ['maceio'] },
  { code: 'JPA', city: 'João Pessoa', aliases: ['joao pessoa'] },
  { code: 'AJU', city: 'Aracaju', aliases: ['aracaju'] },
  { code: 'THE', city: 'Teresina', aliases: ['teresina'] },
  { code: 'SLZ', city: 'São Luís', aliases: ['sao luis', 'são luís'] },
  { code: 'IOS', city: 'Ilhéus', aliases: ['ilheus'] },
  { code: 'JDO', city: 'Juazeiro do Norte', aliases: ['juazeiro do norte'] },
  { code: 'IMP', city: 'Imperatriz', aliases: ['imperatriz'] },
  { code: 'STM', city: 'Santarém', aliases: ['santarem'] },
  { code: 'MAB', city: 'Marabá', aliases: ['maraba'] },
  { code: 'XAP', city: 'Chapecó', aliases: ['chapeco'] },
  { code: 'JOI', city: 'Joinville', aliases: ['joinville'] },
  { code: 'LDB', city: 'Londrina', aliases: ['londrina'] },
  { code: 'UDI', city: 'Uberlândia', aliases: ['uberlandia'] },
  { code: 'CXJ', city: 'Caxias do Sul', aliases: ['caxias do sul'] },
  { code: 'IGU', city: 'Foz do Iguaçu', aliases: ['foz do iguacu', 'iguacu'] },
  { code: 'BVB', city: 'Boa Vista', aliases: ['boa vista'] },
  { code: 'CPV', city: 'Campina Grande', aliases: ['campina grande'] },
  { code: 'PNZ', city: 'Petrolina', aliases: ['petrolina'] },
  { code: 'MCP', city: 'Macapá', aliases: ['macapa'] },
  { code: 'MGF', city: 'Maringá', aliases: ['maringa'] },
  { code: 'NVT', city: 'Navegantes', aliases: ['navegantes', 'balneario camboriu', 'bc'] },
  { code: 'BPS', city: 'Porto Seguro', aliases: ['porto seguro'] },
  { code: 'PMW', city: 'Palmas', aliases: ['palmas'] },
  { code: 'RBR', city: 'Rio Branco', aliases: ['rio branco'] },
  { code: 'PVH', city: 'Porto Velho', aliases: ['porto velho'] },
  { code: 'MOC', city: 'Montes Claros', aliases: ['montes claros'] }
];
const IATA_INTL = [
  { code: 'JFK', city: 'Nova York', aliases: ['new york','nyc','nova york','jfk'] },
  { code: 'LGA', city: 'Nova York', aliases: ['laguardia','nova york','nyc'] },
  { code: 'EWR', city: 'Nova York', aliases: ['newark','nova york','nyc'] },
  { code: 'MIA', city: 'Miami', aliases: ['miami'] },
  { code: 'MCO', city: 'Orlando', aliases: ['orlando','disney'] },
  { code: 'LAX', city: 'Los Angeles', aliases: ['los angeles','la'] },
  { code: 'SFO', city: 'São Francisco', aliases: ['san francisco','sao francisco'] },
  { code: 'YYZ', city: 'Toronto', aliases: ['toronto'] },
  { code: 'YVR', city: 'Vancouver', aliases: ['vancouver'] },
  { code: 'YUL', city: 'Montreal', aliases: ['montreal'] },
  { code: 'LHR', city: 'Londres', aliases: ['london','londres','heathrow'] },
  { code: 'LGW', city: 'Londres', aliases: ['gatwick','londres'] },
  { code: 'CDG', city: 'Paris', aliases: ['paris','charles de gaulle'] },
  { code: 'ORY', city: 'Paris', aliases: ['paris orly','orly','paris'] },
  { code: 'AMS', city: 'Amsterdã', aliases: ['amsterda','amsterdam'] },
  { code: 'FRA', city: 'Frankfurt', aliases: ['frankfurt'] },
  { code: 'MUC', city: 'Munique', aliases: ['munich','munique'] },
  { code: 'BER', city: 'Berlim', aliases: ['berlin','berlim'] },
  { code: 'MAD', city: 'Madri', aliases: ['madrid','madri'] },
  { code: 'BCN', city: 'Barcelona', aliases: ['barcelona'] },
  { code: 'LIS', city: 'Lisboa', aliases: ['lisboa','lisbon'] },
  { code: 'OPO', city: 'Porto', aliases: ['porto','opo'] },
  { code: 'FCO', city: 'Roma', aliases: ['rome','roma','fiumicino'] },
  { code: 'MXP', city: 'Milão', aliases: ['milan','milao','malpensa'] },
  { code: 'VCE', city: 'Veneza', aliases: ['venice','veneza'] },
  { code: 'ZRH', city: 'Zurique', aliases: ['zurich','zurique'] },
  { code: 'GVA', city: 'Genebra', aliases: ['geneva','genebra'] },
  { code: 'EZE', city: 'Buenos Aires', aliases: ['buenos aires','eziza','ezeiza'] },
  { code: 'AEP', city: 'Buenos Aires', aliases: ['aeroparque','buenos aires'] },
  { code: 'SCL', city: 'Santiago', aliases: ['santiago','chile'] },
  { code: 'MVD', city: 'Montevidéu', aliases: ['montevideo','montevideu'] },
  { code: 'MEX', city: 'Cidade do México', aliases: ['mexico city','cidade do mexico','cdmx'] },
  { code: 'CUN', city: 'Cancún', aliases: ['cancun','cancún'] },
  { code: 'LIM', city: 'Lima', aliases: ['lima'] },
  { code: 'BOG', city: 'Bogotá', aliases: ['bogota','bogotá'] },
  { code: 'PTY', city: 'Cidade do Panamá', aliases: ['panama','cidade do panama','tocumen'] },
  { code: 'DXB', city: 'Dubai', aliases: ['dubai'] },
  { code: 'AUH', city: 'Abu Dhabi', aliases: ['abu dhabi'] },
  { code: 'DOH', city: 'Doha', aliases: ['doha'] },
  { code: 'IST', city: 'Istambul', aliases: ['istanbul','istambul'] },
  { code: 'SAW', city: 'Istambul', aliases: ['sabiha gokcen','sabiha gokçen','istanbul'] },
  { code: 'NRT', city: 'Tóquio', aliases: ['narita','toquio','tokyo'] },
  { code: 'HND', city: 'Tóquio', aliases: ['haneda','toquio','tokyo'] },
  { code: 'SIN', city: 'Singapura', aliases: ['singapore','singapura'] },
  { code: 'BKK', city: 'Bangkok', aliases: ['bangkok','suvarnabhumi'] },
  { code: 'DMK', city: 'Bangkok', aliases: ['don mueang','bangkok'] },
  { code: 'DPS', city: 'Bali', aliases: ['bali','denpasar'] },
  { code: 'SYD', city: 'Sydney', aliases: ['sydney'] },
  { code: 'MEL', city: 'Melbourne', aliases: ['melbourne'] },
  { code: 'CMN', city: 'Casablanca', aliases: ['casablanca'] },
  { code: 'RAK', city: 'Marrakech', aliases: ['marrakech','marraquexe'] },
  { code: 'JNB', city: 'Johannesburgo', aliases: ['johannesburg','johanesburgo'] },
  { code: 'CPT', city: 'Cidade do Cabo', aliases: ['cape town','cidade do cabo'] }
];
IATA_LEXICON.push(...IATA_INTL);
function toISO(d){ const yyyy=d.getFullYear(); const mm=String(d.getMonth()+1).padStart(2,'0'); const dd=String(d.getDate()).padStart(2,'0'); return `${yyyy}-${mm}-${dd}`; }
function adjustYearForMonthDayISO(iso, baseYear){ if(!iso||!/^[0-9]{4}-[0-9]{2}-[0-9]{2}$/.test(iso)) return iso; const [, yyyyStr, mmStr, ddStr] = iso.match(/^(\d{4})-(\d{2})-(\d{2})$/)||[]; const today=new Date(); const mm=Number(mmStr)-1; const dd=Number(ddStr); let year = baseYear ?? today.getFullYear(); const candidate=new Date(year, mm, dd); const midnightToday = new Date(today.getFullYear(), today.getMonth(), today.getDate()); if(candidate<midnightToday){ year+=1; } return toISO(new Date(year, mm, dd)); }
function replaceISODatesWithBR(text){ return String(text||'').replace(/\b(\d{4})-(\d{2})-(\d{2})\b/g, (_m, y, m, d)=>`${d}/${m}/${y}`); }
function formatISOToBR(iso){ const m=String(iso||'').match(/^(\d{4})-(\d{2})-(\d{2})$/); if(!m) return iso; return `${m[3]}/${m[2]}/${m[1]}`; }
function adjustYearsInViazaLink(text){ try{ const m=String(text).match(/https:\/\/www\.viaza\.com\.br\/busca\/passagens\/(ow|rt)\/([a-z]+)\/([a-z]+)\?p=([0-9\-]+)&dd=(\d{4}-\d{2}-\d{2})(?:&rd=(\d{4}-\d{2}-\d{2}))?/); if(!m) return text; const trip=m[1]; const depSlug=m[2]; const desSlug=m[3]; const p=m[4]; const dd=m[5]; const rd=m[6]||null; const today=new Date(); const baseYear=today.getFullYear(); const newDd=adjustYearForMonthDayISO(dd, baseYear); let newRd = rd ? adjustYearForMonthDayISO(rd, baseYear) : null; if(newRd){ const dDpt=new Date(newDd); let dDst=new Date(newRd); if(dDst<dDpt){ const mm=dDst.getMonth(); const ddn=dDst.getDate(); dDst=new Date(dDpt.getFullYear(), mm, ddn); if(dDst<dDpt) dDst=new Date(dDpt.getFullYear()+1, mm, ddn); newRd=toISO(dDst); } } const base='https://www.viaza.com.br/busca/passagens'; const params = trip==='ow' ? `p=${p}&dd=${newDd}` : `p=${p}&dd=${newDd}&rd=${newRd}`; const newUrl = `${base}/${trip}/${depSlug}/${desSlug}?${params}`; return String(text).replace(m[0], newUrl); } catch { return text; } }
function withinWindow(iso, days=360){ if(!iso) return false; const now=new Date(); const today=new Date(now.getFullYear(), now.getMonth(), now.getDate()); const t=new Date(iso); const target=new Date(t.getFullYear(), t.getMonth(), t.getDate()); const diffDays=Math.floor((target-today)/(1000*60*60*24)); return diffDays>=0 && diffDays<=days; }
function parseNaturalDate(input){ if(!input) return undefined; const s=stripDiacritics(String(input).trim().toLowerCase()); if(/^\d{4}-\d{2}-\d{2}$/.test(s)) return s; const dm=s.match(/^(\d{1,2})\/(\d{1,2})(?:\/(\d{4}))?$/); if(dm){ const dd=parseInt(dm[1],10); const mm=parseInt(dm[2],10)-1; if(dm[3]){ const yyyy=parseInt(dm[3],10); const d=new Date(yyyy,mm,dd); return toISO(d); } else { const yyyy=new Date().getFullYear(); let d=new Date(yyyy,mm,dd); const today=new Date(); if(d<today) d=new Date(yyyy+1,mm,dd); return toISO(d); } } const mv=s.match(/^mes que vem dia\s*(\d{1,2})$/); if(mv){ const day=parseInt(mv[1],10); const today=new Date(); const nextMonth=new Date(today.getFullYear(), today.getMonth()+1, day); return toISO(nextMonth); } const fer=s.match(/(\d{1,2})\s*de\s*(setembro| setembro)/); if(fer){ const dd=parseInt(fer[1],10); const mm=8; const today=new Date(); let yyyy=today.getFullYear(); let d=new Date(yyyy,mm,dd); if(d<today) d=new Date(yyyy+1,mm,dd); return toISO(d); } const nat=s.replace(/(^|\s)(\d+)[ºo]?\s/g,'$1$2 ').replace(/de\s+/g,' ').trim(); const parts=nat.split(/\s+/); if(parts.length>=2){ const dd=parseInt(parts[0],10); const monthKey=parts[1]; const mm=MONTHS_PT[monthKey]; if(Number.isInteger(dd) && mm!=null){ const today=new Date(); let yyyy=today.getFullYear(); let d=new Date(yyyy,mm,dd); if(d<today) d=new Date(yyyy+1,mm,dd); return toISO(d); } } return undefined; }

function anyAliasMatch(entry, needle){ const n=normalizeText(needle); const terms=[entry.city, ...(entry.aliases||[])].map(normalizeText); if(terms.some(t=>t===n)) return { code: entry.code, matchType: 'exact' }; const distances=terms.map(t=>levenshtein(n,t)); const best=Math.min(...distances); const thr=thresholdFor(n.length); if(best<=thr) return { code: entry.code, matchType: 'fuzzy' }; return null; }
function resolveIATA(input){ if(!input) return undefined; const raw=String(input).trim(); const s=normalizeText(raw); if(/^[a-z]{4}$/i.test(raw)) return undefined; if(/^[a-z]{3}$/i.test(raw)) return raw.toUpperCase(); for(const entry of IATA_LEXICON){ const m=anyAliasMatch(entry,s); if(m && entry.code!=='SAO' && entry.code!=='RIO'){ return entry.code; } } for(const entry of IATA_LEXICON){ const m=anyAliasMatch(entry,s); if(!m) continue; if(entry.code==='SAO') return 'SAO'; if(entry.code==='RIO') return 'RIO'; if(entry.code==='CNF') return 'CNF'; return entry.code; } console.warn('[IATA] Não reconhecido:', raw); unrecognizedIATA.add(s); return undefined; }
function resolveIATAOptions(input){ if(!input) return []; const s=normalizeText(input); for(const entry of IATA_LEXICON){ const m=anyAliasMatch(entry,s); if(!m) continue; const opts=entry.options||[entry.code]; return Array.from(new Set(opts.map(o=>o.toUpperCase()))); } return []; }

const StateSchema = z.object({ dep: z.string().length(3), des: z.string().length(3), adt: z.number().int().min(0), chd: z.number().int().min(0), bby: z.number().int().min(0), dpt: z.string().regex(/^\d{4}-\d{2}-\d{2}$/), dst: z.string().regex(/^\d{4}-\d{2}-\d{2}$/).nullable(), ec: z.boolean(), ow: z.boolean() })
 .refine(s => (s.adt + s.chd + s.bby) <= 9, { message: 'Máximo 9 passageiros por cotação' })
 .refine(s => s.dep.toUpperCase() !== s.des.toUpperCase(), { message: 'Origem e destino não podem ser iguais' })
 .refine(s => OFFICIAL_IATA.has(s.dep.toUpperCase()) && OFFICIAL_IATA.has(s.des.toUpperCase()), { message: 'IATA inválido fora da lista oficial' });

function iataToViazaSlug(code){ const c=String(code||'').toUpperCase(); if(c==='CNF'||c==='PLU') return 'bhz'; if(c==='SAO') return 'sao'; if(c==='RIO') return 'rio'; return c.toLowerCase(); }
async function buildQuoteLinkStandalone(stateArgs){ const s=StateSchema.parse(stateArgs); const dep=s.dep.toUpperCase(); const des=s.des.toUpperCase(); if(!OFFICIAL_IATA.has(dep)||!OFFICIAL_IATA.has(des)) throw new Error('IATA inválido fora da lista oficial'); if(dep===des) throw new Error('Origem e destino não podem ser iguais'); if(!withinWindow(s.dpt)) throw new Error('Data de ida fora da janela de 360 dias'); let dstVal=s.dst; if(s.ow){ dstVal=null; } else { if(!dstVal) throw new Error('Ida e volta requer dst válida'); if(!withinWindow(dstVal)) throw new Error('Data de volta fora da janela de 360 dias'); if(new Date(dstVal) < new Date(s.dpt)) throw new Error('Volta não pode ser antes da ida'); } const trip=s.ow?'ow':'rt'; const p=`${s.adt}-${s.chd}-${s.bby}`; const url=`https://www.viaza.com.br/busca/passagens/${trip}/${iataToViazaSlug(dep)}/${iataToViazaSlug(des)}?p=${p}&dd=${s.dpt}${dstVal?`&rd=${dstVal}`:''}`; return url; }

function createAgent(){
  const instructions = `Você é o assistente da Viaza (www.viaza.com.br). Entenda pedidos de viagem, colete origem/destino, datas (ida/volta) e passageiros (ADT/CHD/BBY ≤9). Valide IATA, datas futuras em até 360 dias, ida ≤ volta. Não usamos milhas do cliente; emitimos por milhas do nosso banco ou convencional, buscando a melhor tarifa. Quando confirmado, gere o link Viaza no padrão https://www.viaza.com.br/busca/passagens/{ow|rt}/{depSlug}/{desSlug}?p=ADT-CHD-BBY&dd=YYYY-MM-DD[&rd=YYYY-MM-DD], onde depSlug/desSlug são IATA em minúsculo com exceções CNF/PLU→bhz, SAO→sao, RIO→rio. Responda curto, claro e amigável.`;
  const agent = new Agent({ name:'Viaza Assistant', instructions, ...(OPENAI_MODEL ? { model: OPENAI_MODEL } : {}), tools: [] });
  return agent;
}

let agent = createAgent();
const pendingQuotes = new Map();
function isConfirmation(text){ const s=normalizeText(String(text||'')); return /(ok|pode|pode prosseguir|pode cotar|sim|vamos|pross?eguir|manda|segue|confirmo|confirmar|confirmado|confirmada|confirmei|finalizar|fechar|pode enviar|pode emitir|gerar link|ativar link)/i.test(s); }

app.get('/', (req,res)=>{ res.json({ name:'Viaza Agent API', message:'Use POST /api/chat para interagir com o agente.', health:'/api/health', chat:'/api/chat' }); });
app.get('/api/health',(req,res)=>{ res.json({ ok:true }); });

app.get('/api/test-agent', async (req, res) => {
  try {
    console.log('🧪 Testando agent...');
    console.log('Agent:', agent);
    console.log('Agent.name:', agent?.name);
    console.log('Agent.tools:', agent?.tools?.length);
    console.log('🚀 Chamando run()...');
    const result = await run(agent, 'Oi');
    console.log('✅ Sucesso! Result:', result);
    return res.json({ success: true, result });
  } catch (err) {
    console.error('❌ Erro:', err.message);
    console.error('Stack:', err.stack);
    return res.status(500).json({ error: err.message, stack: err.stack });
  }
});

async function processChatMessage(sessionId, message){
  if (!sessionId || typeof message !== 'string' || !message.trim()) {
    return { status: 400, error: 'Parâmetros inválidos: sessionId e message são obrigatórios.' };
  }
  if (!sessions.has(sessionId)) sessions.set(sessionId, []);
  const history = sessions.get(sessionId);
  history.push({ role: 'user', content: String(message) });
  if (isConfirmation(message) && pendingQuotes.has(sessionId)) {
    try {
      const url = await buildQuoteLinkStandalone(pendingQuotes.get(sessionId));
      pendingQuotes.delete(sessionId);
      const reply = `Pronto! Aqui está sua cotação:\n${url}`;
      history.push({ role: 'assistant', content: reply });
      return { reply, note: 'confirmed_quote' };
    } catch (err) {
      pendingQuotes.delete(sessionId);
      const reply = 'Houve erro ao gerar o link após sua confirmação. Vamos ajustar dados (IATA, datas, passageiros) e tentar novamente?';
      history.push({ role: 'assistant', content: reply });
      return { reply, note: 'quote_error_after_confirmation' };
    }
  }
  // Tentar computar estado diretamente da mensagem para confirmação
  const inferred = computeStateFromMessage(message);
  if (inferred) {
    pendingQuotes.set(sessionId, inferred);
    const reply = summaryForState(inferred);
    history.push({ role: 'assistant', content: reply });
    return { reply, note: 'summary_for_confirmation' };
  }
  if (!OPENAI_API_KEY) return { status: 500, error: 'Backend sem OPENAI_API_KEY configurada.' };
  try {
    const prompt = `Cliente: ${String(message)}\nAssistente:`;
    const result = await run(agent, prompt);
    let reply = result?.finalOutput ?? 'Não consegui gerar resposta.';
    reply = replaceISODatesWithBR(adjustYearsInViazaLink(reply));
    history.push({ role: 'assistant', content: reply });
    return { reply };
  } catch (err) {
    return { status: 500, error: 'Instabilidade no agente. Tente novamente.' };
  }
}

app.post('/api/chat', async (req, res) => {
  const { sessionId, message } = req.body || {};
  if (!sessionId || !message) {
    return res.status(400).json({ error: 'Parâmetros obrigatórios: sessionId e message' });
  }

  try {
    if (!sessions.has(sessionId)) { sessions.set(sessionId, []); }
    const history = sessions.get(sessionId);
    history.push({ role: 'user', content: String(message) });

    if (!OPENAI_API_KEY) {
      return res.status(500).json({ reply: 'Backend sem OPENAI_API_KEY configurada.' });
    }

    function getSessionState(sid){ return sessionState.get(sid) || {}; }
    function isStateComplete(s){
      if (!s) return false;
      const total = (s.adt||0)+(s.chd||0)+(s.bby||0);
      const dep = s.dep && String(s.dep).toUpperCase();
      const des = s.des && String(s.des).toUpperCase();
      const ow = !!s.ow;
      const hasDstOk = ow ? true : !!s.dst;
      return !!dep && !!des && dep!==des && OFFICIAL_IATA.has(dep) && OFFICIAL_IATA.has(des) && !!s.dpt && total>0 && hasDstOk;
    }
    function mergeStateData(sid, partial){ const cur=getSessionState(sid); const merged={ ...cur, ...partial }; sessionState.set(sid, merged); return merged; }
    function formatStateMessage(s){ return summaryForState(s); }

    let state = getSessionState(sessionId);
    console.log('📋 Estado atual:', state);

    if (isStateComplete(state) && isConfirmation(message)) {
      console.log('✅ Estado completo e confirmado! Gerando link...');
      try {
        const url = await buildQuoteLinkStandalone(state);
        sessionState.delete(sessionId);
        console.log('🧹 Memória da sessão limpa!');
        const reply = `🎉 Pronto! Aqui está sua cotação:\n${url}\n\nDeseja outra cotação?`;
        history.push({ role: 'assistant', content: reply });
        return res.json({ reply });
      } catch (err) {
        console.error('❌ Erro ao gerar link:', err.message);
        sessionState.delete(sessionId);
        console.log('🧹 Memória limpa após erro!');
        const reply = 'Houve erro ao gerar o link. Vamos começar uma nova cotação? Informe origem, destino, data e passageiros.';
        history.push({ role: 'assistant', content: reply });
        return res.json({ reply });
      }
    }

    console.log('🔍 Extraindo dados da mensagem...');
    const inferred = computeStateFromMessage(message);
    if (inferred) {
      console.log('✅ Novos dados encontrados:', inferred);
      if (Object.keys(state).length > 0 && (state.dep !== inferred.dep || state.des !== inferred.des)) {
        console.log('🔄 Novo pedido detectado! Limpando memória anterior...');
        sessionState.delete(sessionId);
        state = {};
        console.log('🧹 Memória limpa para novo pedido!');
      }
      const merged = mergeStateData(sessionId, { dep: inferred.dep, des: inferred.des, dpt: inferred.dpt, dst: inferred.dst, adt: inferred.adt, chd: inferred.chd, bby: inferred.bby, ec: inferred.ec, ow: inferred.ow });
      if (isStateComplete(merged)) {
        const summary = formatStateMessage(merged);
        const reply = `${summary}\n\n✅ Dados completos! Confirma?`;
        history.push({ role: 'assistant', content: reply });
        return res.json({ reply });
      } else {
        const summary = formatStateMessage(merged);
        const missing = [];
        if (!merged.dep || !merged.des) missing.push('**origem e destino**');
        if (!merged.dpt) missing.push('**data de ida**');
        if (!merged.dst && !merged.ow) missing.push('**data de volta**');
        if ((merged.adt||0)+(merged.chd||0)+(merged.bby||0)===0) missing.push('**número de passageiros**');
        const reply = `${summary}\n\n📝 Ainda preciso de: ${missing.join(', ')}`;
        history.push({ role: 'assistant', content: reply });
        return res.json({ reply });
      }
    }

    console.log('📊 Preenchendo dados parciais...');
    let hasNewData = false;
    const dates = extractDatesFromText(message);
    if (dates && dates.dpt) { mergeStateData(sessionId, { dpt: dates.dpt, dst: dates.dst, ow: dates.ow }); hasNewData = true; console.log('✅ Data armazenada'); }
    const pax = extractPassengersFromText(message);
    if (pax && ((pax.adt||0)+(pax.chd||0)+(pax.bby||0)>0)) { mergeStateData(sessionId, { adt: pax.adt, chd: pax.chd, bby: pax.bby }); hasNewData = true; console.log('✅ Passageiros armazenados'); }
    state = getSessionState(sessionId);
    if (isStateComplete(state)) {
      const summary = formatStateMessage(state);
      const reply = `${summary}\n\n✅ Dados completos! Confirma?`;
      history.push({ role: 'assistant', content: reply });
      return res.json({ reply });
    }

    if (hasNewData) {
      console.log('📋 Mostrando estado e pedindo resto...');
      const summary = formatStateMessage(state);
      const missing = [];
      if (!state.dep || !state.des) missing.push('**origem e destino**');
      if (!state.dpt) missing.push('**data de ida**');
      if (!state.dst && !state.ow) missing.push('**data de volta**');
      if ((state.adt||0)+(state.chd||0)+(state.bby||0)===0) missing.push('**número de passageiros**');
      const reply = `${summary}\n\n📝 Ainda preciso de: ${missing.join(', ')}`;
      history.push({ role: 'assistant', content: reply });
      return res.json({ reply });
    }

    console.log('🚀 Nenhum dado extraído. Chamando agent...');
    try {
      const result = await run(agent, message);
      let reply = null;
      if (result?.state?.modelResponses?.[0]?.output?.[0]?.content?.[0]?.text) { reply = result.state.modelResponses[0].output[0].content[0].text; }
      else if (result?.finalOutput) { reply = result.finalOutput; }
      else if (typeof result === 'string') { reply = result; }
      else { reply = 'Não consegui processar sua mensagem.'; }
      reply = replaceISODatesWithBR(adjustYearsInViazaLink(reply));
      history.push({ role: 'assistant', content: reply });
      return res.json({ reply });
    } catch (err) {
      console.error('❌ ERRO NO AGENT:', err.message);
      return res.status(500).json({ reply: 'Desculpe, tive um problema. Pode repetir sua solicitação?' });
    }

  } catch (err) {
    console.error('❌ Erro geral:', err);
    return res.status(500).json({ reply: 'Erro interno. Tente novamente.' });
  }
});

function evoDigits(n){ return String(n||'').replace(/\D+/g,''); }
async function evoSendText(number,text){ if(!EVO_API_URL||!EVO_API_KEY||!EVO_INSTANCE) return false; const base=String(EVO_API_URL).replace(/\/+$/,''); const url=`${base}/message/sendText/${EVO_INSTANCE}`; const body={ number:evoDigits(number), textMessage:{ text:String(text||'') } }; const r=await fetch(url,{ method:'POST', headers:{ 'Content-Type':'application/json', apikey:EVO_API_KEY }, body: JSON.stringify(body) }); return r.ok; }
function parseEvoPayload(body){ let text=null; let number=null; const m=body?.messages?.[0]; text=m?.message?.conversation||m?.message?.extendedTextMessage?.text||m?.message?.textMessage?.text||body?.textMessage?.text||body?.message?.textMessage?.text||body?.message?.conversation||body?.message?.extendedTextMessage?.text||body?.text; const jid=m?.key?.remoteJid||body?.key?.remoteJid||null; if(jid) number=evoDigits(String(jid).split('@')[0]); if(!number) number=body?.number||body?.sender?.phone||body?.phone||null; number=number?evoDigits(number):null; return { text, number }; }
app.post('/webhook/evo', async (req,res)=>{
  try {
    const { text, number } = parseEvoPayload(req.body || {});
    if (!text || !number) return res.json({ ok: true });
    const sid = number;
    if (!sessions.has(sid)) sessions.set(sid, []);
    const history = sessions.get(sid);
    history.push({ role: 'user', content: String(text) });

    if (!OPENAI_API_KEY) {
      await evoSendText(number, 'Erro: API key não configurada');
      return res.json({ ok: true });
    }

    const inferred = computeStateFromMessage(text);
    let reply = '';

    if (isConfirmation(text) && pendingQuotes.has(sid)) {
      try {
        const url = await buildQuoteLinkStandalone(pendingQuotes.get(sid));
        pendingQuotes.delete(sid);
        reply = `Pronto! Aqui está sua cotação:\n${url}`;
        history.push({ role: 'assistant', content: reply });
      } catch (err) {
        pendingQuotes.delete(sid);
        reply = 'Houve erro ao gerar o link. Tente novamente com os dados completos.';
        history.push({ role: 'assistant', content: reply });
      }
    } else if (inferred) {
      pendingQuotes.set(sid, inferred);
      const summary = summaryForState(inferred);
      reply = `${summary} Confirma?`;
      history.push({ role: 'assistant', content: reply });
    } else {
      try {
        const result = await run(agent, text);
        reply = result?.state?.modelResponses?.[0]?.output?.[0]?.content?.[0]?.text || result?.finalOutput || 'Desculpe, não consegui processar.';
        reply = replaceISODatesWithBR(adjustYearsInViazaLink(reply));
        history.push({ role: 'assistant', content: reply });
      } catch (err) {
        reply = 'Desculpe, tive um problema. Tente novamente.';
      }
    }

    if (reply) await evoSendText(number, reply);
    return res.json({ ok: true });
  } catch {
    return res.json({ ok: true });
  }
});
app.post('/evo/send-text', async (req,res)=>{ try{ const number=req.body?.number||req.query?.number; const text=req.body?.text||req.query?.text||'Teste Viaza: integração Evolution ativa.'; if(!number) return res.status(400).send('number'); const ok=await evoSendText(number,text); return res.json({ ok, number: evoDigits(number), text }); } catch(err){ return res.status(500).send(String(err?.message||err)); } });

app.listen(Number(PORT), ()=>{ console.log(`Servidor iniciado em http://localhost:${PORT}`); });
// Extração de dados a partir da mensagem para resumo/confirmacao
function extractIATAFromText(text){
  const raw = String(text || '').trim();
  let cleaned = raw
    .replace(/^(oi|ola|olá|hey|ola tudo bem|agora|agora quero|mas|pois|e|ja|já|também|tbm|eae|e ai)\s+/i, '')
    .replace(/^(quero|quer|passagem|passagens|uma|um|vou|queria|gostaria|preciso|busco|desejo|gostaria)\s+/i, '')
    .trim();
  const norm = normalizeText(cleaned);

  console.log('  🔍 extractIATAFromText:', { raw, cleaned, norm });

  function isForbiddenToken(tok){
    const t = normalizeText(tok);
    return IATA_STOPLIST.has(t);
  }

  const patterns = [
    /de\s+([a-záàâãéèêíïóôõöúçñ\s]+?)\s+(?:para|pra)\s+([a-záàâãéèêíïóôõöúçñ\s]+?)(?:\s+(?:dia|data|ida|volta|com|sem)|$)/i,
    /(?:saindo\s+)?de\s+([a-záàâãéèêíïóôõöúçñ\s]+?)\s+(?:para|pra)\s+([a-záàâãéèêíïóôõöúçñ\s]+?)(?:\s|,|$)/i,
    /([a-záàâãéèêíïóôõöúçñ\s]+?)\s+(?:para|pra)\s+([a-záàâãéèêíïóôõöúçñ\s]+?)(?:\s+(?:dia|data|ida|volta)|$)/i,
  ];

  for (const pattern of patterns) {
    const match = cleaned.match(pattern);
    if (match) {
      let left = match[1].trim();
      let right = match[2].trim();
      left = left.replace(/^(quero|quer|passagem|passagens|ida|volta|uma|um|vou|queria|gostaria|preciso|busco|desejo)\s+/i, '').replace(/\s+(passagem|passagens)$/i, '').trim();
      right = right.replace(/\s+(dia|data|com|sem|partindo|retorno|ida|volta|em|no|na).*$/i, '').trim();
      console.log('  📍 Padrão encontrado:', { left, right });
      if (left && right && !isForbiddenToken(left) && !isForbiddenToken(right)) {
        const dep = resolveIATA(left);
        const des = resolveIATA(right);
        console.log('  ✅ Resolveu:', { dep, des });
        if (dep && des) {
          console.log('  🎯 RETORNANDO:', { dep, des });
          return { dep, des };
        }
      }
    }
  }

  const codes = Array.from(cleaned.matchAll(/\b([A-Za-z]{3})\b/g))
    .map(x => x[1].toUpperCase())
    .filter(c => !IATA_STOPLIST.has(normalizeText(c)));
  console.log('  📍 Códigos IATA:', codes);
  if (codes.length >= 2) {
    console.log('  🎯 RETORNANDO códigos:', [codes[0], codes[1]]);
    return { dep: codes[0], des: codes[1] };
  }
  if (codes.length === 1) {
    console.log('  📍 Um código, procurando alias...');
    const textWithoutCode = cleaned.replace(new RegExp(`\\b${codes[0]}\\b`, 'i'), '');
    for (const entry of IATA_LEXICON) {
      const match = anyAliasMatch(entry, normalizeText(textWithoutCode));
      if (match && entry.code !== codes[0]) {
        console.log('  🎯 RETORNANDO misto:', [codes[0], entry.code]);
        return { dep: codes[0], des: entry.code };
      }
    }
  }

  const found = [];
  for (const entry of IATA_LEXICON) {
    const match = anyAliasMatch(entry, norm);
    if (match) {
      found.push({ code: entry.code, matchType: match.matchType });
      if (found.length >= 2) break;
    }
  }
  console.log('  📍 Aliases:', found);
  if (found.length >= 2) {
    const exacts = found.filter(f => f.matchType === 'exact').map(f => f.code);
    if (exacts.length >= 2) {
      console.log('  🎯 RETORNANDO exatos:', exacts);
      return { dep: exacts[0], des: exacts[1] };
    }
    console.log('  🎯 RETORNANDO aliases:', [found[0].code, found[1].code]);
    return { dep: found[0].code, des: found[1].code };
  }

  console.log('  ❌ Não encontrou IATA');
  return null;
}
function extractDatesFromText(text){ const raw=stripDiacritics(String(text||'').toLowerCase()); const candidates=[]; const isoMatches=raw.match(/\b\d{4}-\d{2}-\d{2}\b/g)||[]; candidates.push(...isoMatches); const dmMatches=raw.match(/\b\d{1,2}\/\d{1,2}(?:\/\d{4})?\b/g)||[]; candidates.push(...dmMatches); const monthKeys=Object.keys(MONTHS_PT).join('|'); const natMatches=Array.from(raw.matchAll(new RegExp(`(\\d{1,2})\\s*(?:de\\s*)?(${monthKeys})`,'g'))).map(m=>`${m[1]} ${m[2]}`); candidates.push(...natMatches); const parsed=[]; for(const c of candidates){ const iso=parseNaturalDate(c); if(iso) parsed.push(iso); if(parsed.length>=2) break; } const dpt=parsed[0]||null; const dst=parsed[1]||null; if(!dpt) return null; if(!withinWindow(dpt)) return null; if(dst){ if(!withinWindow(dst)) return null; if(new Date(dst)<new Date(dpt)) return null; } return { dpt, dst, ow: !dst }; }
function extractPassengersFromText(text){ const s=stripDiacritics(String(text||'').toLowerCase()); const num=(v)=>parseNumberPt(v)??(isNaN(Number(v))?undefined:Number(v)); const mAdt=s.match(/(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove)\s*adult/); const mChd=s.match(/(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove)\s*crianc/); const mBby=s.match(/(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove)\s*beb/); const adt=num(mAdt?.[1])??1; const chd=num(mChd?.[1])??0; const bby=num(mBby?.[1])??0; if(adt+chd+bby>9) return null; return { adt, chd, bby }; }
function computeStateFromMessage(message){
  console.log('🔧 computeStateFromMessage chamada com:', message);

  const places = extractIATAFromText(message);
  console.log('  → places:', places);

  const dates = extractDatesFromText(message);
  console.log('  → dates:', dates);

  const pax = extractPassengersFromText(message);
  console.log('  → pax:', pax);

  if(!places||!dates||!pax){
    console.log('  ❌ Faltam dados!');
    return null;
  }

  if(!places.dep||!places.des){
    console.log('  ❌ Faltam IATA!');
    return null;
  }

  const total=(pax.adt||0)+(pax.chd||0)+(pax.bby||0);
  if(total<0||total>9){
    console.log('  ❌ Total de passageiros inválido:', total);
    return null;
  }

  const ow=!!dates.ow;
  const dst=ow?null:(dates.dst||null);
  if(!ow&&!dst){
    console.log('  ❌ Sem data de volta!');
    return null;
  }

  const dep=String(places.dep).toUpperCase();
  const des=String(places.des).toUpperCase();
  if(!OFFICIAL_IATA.has(dep)||!OFFICIAL_IATA.has(des)){
    console.log('  ❌ IATA não oficial:', { dep, des });
    return null;
  }

  if(dep===des){
    console.log('  ❌ Origem = Destino');
    return null;
  }

  const state = { dep, des, adt:pax.adt, chd:pax.chd, bby:pax.bby, dpt:dates.dpt, dst, ec:true, ow };
  console.log('  ✅ State completo:', state);
  return state;
}
function summaryForState(state){ const dptBr=formatISOToBR(state.dpt); const hasDst=!!state.dst; const dstBr=hasDst?formatISOToBR(state.dst):null; const pax=[]; if(state.adt>0) pax.push(`${state.adt} adulto(s)`); if(state.chd>0) pax.push(`${state.chd} criança(s)`); if(state.bby>0) pax.push(`${state.bby} bebê(s)`); const paxStr=pax.join(', '); const dstPart=hasDst?` e volta ${dstBr}`:''; return `Resumo: origem ${state.dep}, destino ${state.des}, ida ${dptBr}${dstPart}.${paxStr?` Passageiros: ${paxStr}.`:''} Posso cotar?`; }
