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
const unrecognizedIATA = new Set();
const OFFICIAL_IATA = new Set([
  'BSB','CGH','GIG','SSA','FLN','POA','VCP','REC','CWB','BEL','VIX','SDU','CGB','CGR','FOR','MCP','MGF','GYN','NVT','MAO','NAT','BPS','MCZ','PMW','SLZ','GRU','LDB','PVH','RBR','JOI','UDI','CXJ','IGU','THE','AJU','JPA','PNZ','CNF','BVB','CPV','STM','IOS','JDO','IMP','XAP','MAB','CZS','PPB','CFB','FEN','JTC','MOC','SAO','RIO'
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
  { code: 'CNF', city: 'Belo Horizonte', aliases: ['bh', 'confins', 'pampulha'] },
  { code: 'GRU', city: 'São Paulo', aliases: ['sp', 'guarulhos', 'congonhas'] },
  { code: 'SDU', city: 'Rio de Janeiro', aliases: ['rio', 'santos dumont'] },
  { code: 'GIG', city: 'Rio de Janeiro', aliases: ['galeão', 'tom jobim'] },
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
  { code: 'RIO', city: 'Rio de Janeiro', aliases: ['rio de janeiro'] },
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

  const ResolveIATASchema = z.object({ origem: z.string(), destino: z.string() });
  const DateParseSchema = z.object({ ida: z.string(), volta: z.string().optional() });
  const PaxParseSchema = z.object({ total: z.union([z.number(), z.string()]).optional(), composicao: z.string().optional() });

  const ResolveIATAJson = { type:'object', properties:{ origem:{type:'string'}, destino:{type:'string'} }, required:['origem','destino'], additionalProperties:false };
  const DateParseJson = { type:'object', properties:{ ida:{type:'string'}, volta:{type:'string'} }, required:['ida'], additionalProperties:false };
  const PaxParseJson = { type:'object', properties:{ total:{ oneOf:[{type:'number'},{type:'string'}] }, composicao:{type:'string'} }, additionalProperties:false };
  const StateJson = { type:'object', properties:{ dep:{type:'string',minLength:3,maxLength:3}, des:{type:'string',minLength:3,maxLength:3}, adt:{type:'integer',minimum:0}, chd:{type:'integer',minimum:0}, bby:{type:'integer',minimum:0}, dpt:{type:'string',pattern:'^\\d{4}-\\d{2}-\\d{2}$'}, dst:{ anyOf:[ {type:'string',pattern:'^\\d{4}-\\d{2}-\\d{2}$'}, {type:'null'} ] }, ec:{type:'boolean'}, ow:{type:'boolean'} }, required:['dep','des','adt','chd','bby','dpt','ec','ow'], additionalProperties:false };

  async function resolveIATATool(args){ const { origem, destino } = ResolveIATASchema.parse(args); const dep=resolveIATA(origem); const des=resolveIATA(destino); const depOpts=resolveIATAOptions(origem); const desOpts=resolveIATAOptions(destino); const ambiguousDep = dep==='SAO' || dep==='RIO' || (depOpts.length>1); const ambiguousDes = des==='SAO' || des==='RIO' || (desOpts.length>1); return { dep, des, depOptions:depOpts, desOptions:desOpts, ambiguousDep, ambiguousDes }; }
  async function parseDatesTool(args){ const { ida, volta } = DateParseSchema.parse(args); const dpt=parseNaturalDate(ida); const dst=volta?parseNaturalDate(volta):null; if(!dpt) throw new Error('Data de ida inválida'); if(!withinWindow(dpt)) throw new Error('Data de ida fora da janela de 360 dias'); if(dst){ if(!withinWindow(dst)) throw new Error('Data de volta fora da janela de 360 dias'); if(new Date(dst) < new Date(dpt)) throw new Error('Volta não pode ser antes da ida'); } return { dpt, dst, ow: !dst }; }
  async function parsePassengersTool(args){ const { total, composicao } = PaxParseSchema.parse(args); const totalNum = total!=null ? (typeof total==='number'? total : parseNumberPt(total)) : undefined; let chd=0,bby=0; if(composicao){ const s=stripDiacritics(composicao.toLowerCase()); const mChd=s.match(/(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove)\s*crianc/); const mBby=s.match(/(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove)\s*beb/); if(mChd) chd=parseNumberPt(mChd[1])??0; if(mBby) bby=parseNumberPt(mBby[1])??0; } const totalFinal = totalNum ?? (chd + bby + 1); const adt=Math.max(totalFinal - chd - bby, 0); if(adt+chd+bby>9) throw new Error('Máximo 9 passageiros por cotação'); return { adt, chd, bby }; }
  async function buildQuoteLink(stateArgs){ return buildQuoteLinkStandalone(stateArgs); }

  const agent = new Agent({ name:'Viaza Assistant', instructions, ...(OPENAI_MODEL ? { model: OPENAI_MODEL } : {}), tools:[ {type:'function', name:'resolveIATA', parameters:ResolveIATAJson, invoke:resolveIATATool, needsApproval:()=>false}, {type:'function', name:'parseDates', parameters:DateParseJson, invoke:parseDatesTool, needsApproval:()=>false}, {type:'function', name:'parsePassengers', parameters:PaxParseJson, invoke:parsePassengersTool, needsApproval:()=>false}, {type:'function', name:'buildQuoteLink', parameters:StateJson, invoke:buildQuoteLink, needsApproval:()=>false} ] });
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
    return res.status(400).send('Parâmetros obrigatórios');
  }
  try {
    if (!sessions.has(sessionId)) {
      sessions.set(sessionId, []);
    }
    const history = sessions.get(sessionId);
    history.push({ role: 'user', content: message });

    if (!OPENAI_API_KEY) {
      return res.status(500).json({ reply: 'Backend sem OPENAI_API_KEY configurada.' });
    }

    try {
      console.log('🚀 Chamando run() com:', message);
      const result = await run(agent, message);
      console.log('📊 Tipo de result:', typeof result);
      console.log('📊 Keys de result:', Object.keys(result || {}));
      console.log('📊 Result completo:', JSON.stringify(result, null, 2));

      let reply = null;
      if (result?.lastModelResponse?.output?.[0]?.content?.[0]?.text) {
        reply = result.lastModelResponse.output[0].content[0].text;
        console.log('✅ Extraído de: lastModelResponse');
      } else if (result?.state?.lastModelResponse?.output?.[0]?.content?.[0]?.text) {
        reply = result.state.lastModelResponse.output[0].content[0].text;
        console.log('✅ Extraído de: state.lastModelResponse');
      } else if (result?.output_text) {
        reply = result.output_text;
        console.log('✅ Extraído de: output_text');
      } else if (result?.finalOutput) {
        reply = result.finalOutput;
        console.log('✅ Extraído de: finalOutput');
      } else if (typeof result === 'string') {
        reply = result;
        console.log('✅ Result é string');
      } else {
        reply = 'Não consegui extrair resposta.';
        console.log('❌ Não consegui extrair');
      }

      console.log('📤 Respondendo com:', reply);
      history.push({ role: 'assistant', content: reply });
      return res.json({ reply });
    } catch (err) {
      console.error('❌ ERRO NO AGENT:');
      console.error('  Mensagem:', err.message);
      console.error('  Stack:', err.stack);
      return res.status(500).json({ reply: 'Instabilidade no agente. Tente novamente.' });
    }
  } catch (err) {
    console.error('❌ Erro geral:', err);
    return res.status(500).json({ reply: 'Erro interno.' });
  }
});

function evoDigits(n){ return String(n||'').replace(/\D+/g,''); }
async function evoSendText(number,text){ if(!EVO_API_URL||!EVO_API_KEY||!EVO_INSTANCE) return false; const base=String(EVO_API_URL).replace(/\/+$/,''); const url=`${base}/message/sendText/${EVO_INSTANCE}`; const body={ number:evoDigits(number), textMessage:{ text:String(text||'') } }; const r=await fetch(url,{ method:'POST', headers:{ 'Content-Type':'application/json', apikey:EVO_API_KEY }, body: JSON.stringify(body) }); return r.ok; }
function parseEvoPayload(body){ let text=null; let number=null; const m=body?.messages?.[0]; text=m?.message?.conversation||m?.message?.extendedTextMessage?.text||m?.message?.textMessage?.text||body?.textMessage?.text||body?.message?.textMessage?.text||body?.message?.conversation||body?.message?.extendedTextMessage?.text||body?.text; const jid=m?.key?.remoteJid||body?.key?.remoteJid||null; if(jid) number=evoDigits(String(jid).split('@')[0]); if(!number) number=body?.number||body?.sender?.phone||body?.phone||null; number=number?evoDigits(number):null; return { text, number }; }
app.post('/webhook/evo', async (req,res)=>{ try{ const { text, number }=parseEvoPayload(req.body||{}); if(!text||!number) return res.json({ ok:true }); const sid=number; const result=await processChatMessage(sid, String(text)); if(result?.reply) await evoSendText(number, result.reply); return res.json({ ok:true }); } catch { return res.json({ ok:true }); } });
app.post('/evo/send-text', async (req,res)=>{ try{ const number=req.body?.number||req.query?.number; const text=req.body?.text||req.query?.text||'Teste Viaza: integração Evolution ativa.'; if(!number) return res.status(400).send('number'); const ok=await evoSendText(number,text); return res.json({ ok, number: evoDigits(number), text }); } catch(err){ return res.status(500).send(String(err?.message||err)); } });

app.listen(Number(PORT), ()=>{ console.log(`Servidor iniciado em http://localhost:${PORT}`); });
// Extração de dados a partir da mensagem para resumo/confirmacao
function extractIATAFromText(text){ const raw=String(text||'').trim(); const norm=normalizeText(raw); function isForbiddenToken(tok){ const t=normalizeText(tok); return IATA_STOPLIST.has(t); } let m=norm.match(/\bsaindo\s+de\s+(.+?)\s+(?:pra|para)\s+(.+)/i); if(!m) m=norm.match(/\bde\s+(.+?)\s+(?:pra|para|\u2192|\-\>|\-)\s+(.+)/i); if(m){ const left=m[1].trim(); const right=m[2].trim(); if(!isForbiddenToken(left)&&!isForbiddenToken(right)){ const dep=resolveIATA(left); const des=resolveIATA(right); if(dep&&des&&OFFICIAL_IATA.has(dep)&&OFFICIAL_IATA.has(des)) return { dep, des }; } } const codes=Array.from(raw.matchAll(/\b([A-Za-z]{3})\b/g)).map(x=>x[1].toUpperCase()).filter(c=>OFFICIAL_IATA.has(c)&&!IATA_STOPLIST.has(normalizeText(c))); if(codes.length>=2) return { dep: codes[0], des: codes[1] }; const found=[]; for(const entry of IATA_LEXICON){ const match=anyAliasMatch(entry,norm); if(match&&OFFICIAL_IATA.has(entry.code)){ found.push(entry.code); if(found.length>=2) break; } } if(found.length>=2) return { dep: found[0], des: found[1] }; if(found.length===1) return { dep: found[0], des: null }; return null; }
function extractDatesFromText(text){ const raw=stripDiacritics(String(text||'').toLowerCase()); const candidates=[]; const isoMatches=raw.match(/\b\d{4}-\d{2}-\d{2}\b/g)||[]; candidates.push(...isoMatches); const dmMatches=raw.match(/\b\d{1,2}\/\d{1,2}(?:\/\d{4})?\b/g)||[]; candidates.push(...dmMatches); const monthKeys=Object.keys(MONTHS_PT).join('|'); const natMatches=Array.from(raw.matchAll(new RegExp(`(\\d{1,2})\\s*(?:de\\s*)?(${monthKeys})`,'g'))).map(m=>`${m[1]} ${m[2]}`); candidates.push(...natMatches); const parsed=[]; for(const c of candidates){ const iso=parseNaturalDate(c); if(iso) parsed.push(iso); if(parsed.length>=2) break; } const dpt=parsed[0]||null; const dst=parsed[1]||null; if(!dpt) return null; if(!withinWindow(dpt)) return null; if(dst){ if(!withinWindow(dst)) return null; if(new Date(dst)<new Date(dpt)) return null; } return { dpt, dst, ow: !dst }; }
function extractPassengersFromText(text){ const s=stripDiacritics(String(text||'').toLowerCase()); const num=(v)=>parseNumberPt(v)??(isNaN(Number(v))?undefined:Number(v)); const mAdt=s.match(/(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove)\s*adult/); const mChd=s.match(/(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove)\s*crianc/); const mBby=s.match(/(\d+|uma|um|duas|dois|tres|três|quatro|cinco|seis|sete|oito|nove)\s*beb/); const adt=num(mAdt?.[1])??1; const chd=num(mChd?.[1])??0; const bby=num(mBby?.[1])??0; if(adt+chd+bby>9) return null; return { adt, chd, bby }; }
function computeStateFromMessage(message){ const places=extractIATAFromText(message); const dates=extractDatesFromText(message); const pax=extractPassengersFromText(message); if(!places||!dates||!pax) return null; if(!places.dep||!places.des) return null; const total=(pax.adt||0)+(pax.chd||0)+(pax.bby||0); if(total<0||total>9) return null; const ow=!!dates.ow; const dst=ow?null:(dates.dst||null); if(!ow&&!dst) return null; const dep=String(places.dep).toUpperCase(); const des=String(places.des).toUpperCase(); if(!OFFICIAL_IATA.has(dep)||!OFFICIAL_IATA.has(des)) return null; if(dep===des) return null; return { dep, des, adt:pax.adt, chd:pax.chd, bby:pax.bby, dpt:dates.dpt, dst, ec:true, ow }; }
function summaryForState(state){ const dptBr=formatISOToBR(state.dpt); const hasDst=!!state.dst; const dstBr=hasDst?formatISOToBR(state.dst):null; const pax=[]; if(state.adt>0) pax.push(`${state.adt} adulto(s)`); if(state.chd>0) pax.push(`${state.chd} criança(s)`); if(state.bby>0) pax.push(`${state.bby} bebê(s)`); const paxStr=pax.join(', '); const dstPart=hasDst?` e volta ${dstBr}`:''; return `Resumo: origem ${state.dep}, destino ${state.des}, ida ${dptBr}${dstPart}.${paxStr?` Passageiros: ${paxStr}.`:''} Posso cotar?`; }
