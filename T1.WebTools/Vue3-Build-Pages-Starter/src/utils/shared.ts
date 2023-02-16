export const isNullOrUndefined =(value: unknown): value is undefined | null => {
  return value === undefined || value === null
}

export const deepClone = <T>( target: T): T => {
  if (target === null) {
    return target
  }
  if (target instanceof Date) {
    return new Date(target.getTime()) as any
  }

  if (target instanceof Array) {
    return ([...target] as any[]).map((n: any) => deepClone<any>(n)) as any
  }

  if (typeof target === 'object') {
    const cp = { ...(target as Record<string, any>) } as Record<string, any>
    Object.keys(cp).forEach((k) => {
      cp[k] = deepClone<any>(cp[k])
    })
    return cp as T
  }
  return target
}
export const isObject = (obj: unknown): obj is Record<string, unknown> =>
  obj !== null && !!obj && typeof obj === 'object' && !Array.isArray(obj)


const firstTLDs = 'ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|be|bf|bg|bh|bi|bj|bm|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|cl|cm|cn|co|cr|cu|cv|cw|cx|cz|de|dj|dk|dm|do|dz|ec|ee|eg|es|et|eu|fi|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|im|in|io|iq|ir|is|it|je|jo|jp|kg|ki|km|kn|kp|kr|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|na|nc|ne|nf|ng|nl|no|nr|nu|nz|om|pa|pe|pf|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|yt'.split('|')
const secondTLDs = 'com|edu|gov|net|mil|org|nom|sch|caa|res|off|gob|int|tur|ip6|uri|urn|asn|act|nsw|qld|tas|vic|pro|biz|adm|adv|agr|arq|art|ato|bio|bmd|cim|cng|cnt|ecn|eco|emp|eng|esp|etc|eti|far|fnd|fot|fst|g12|ggf|imb|ind|inf|jor|jus|leg|lel|mat|med|mus|not|ntr|odo|ppg|psc|psi|qsl|rec|slg|srv|teo|tmp|trd|vet|zlg|web|ltd|sld|pol|fin|k12|lib|pri|aip|fie|eun|sci|prd|cci|pvt|mod|idv|rel|sex|gen|nic|abr|bas|cal|cam|emr|fvg|laz|lig|lom|mar|mol|pmn|pug|sar|sic|taa|tos|umb|vao|vda|ven|mie|北海道|和歌山|神奈川|鹿児島|ass|rep|tra|per|ngo|soc|grp|plc|its|air|and|bus|can|ddr|jfk|mad|nrw|nyc|ski|spy|tcm|ulm|usa|war|fhs|vgs|dep|eid|fet|fla|flå|gol|hof|hol|sel|vik|cri|iwi|ing|abo|fam|gok|gon|gop|gos|aid|atm|gsm|sos|elk|waw|est|aca|bar|cpa|jur|law|sec|plo|www|bir|cbg|jar|khv|msk|nov|nsk|ptz|rnd|spb|stv|tom|tsk|udm|vrn|cmw|kms|nkz|snz|pub|fhv|red|ens|nat|rns|rnu|bbs|tel|bel|kep|nhs|dni|fed|isa|nsn|gub|e12|tec|орг|обр|упр|alt|nis|jpn|mex|ath|iki|nid|gda|inc|info|asia'.split('|')
export function getDomain(url: string) {
  const parts = url.replace(/^www\./, '').split('.').slice(-3)
  if (parts.length === 3) {
    if (!((firstTLDs.includes(parts[1]) || secondTLDs.includes(parts[1])) && (firstTLDs.includes(parts[2]) || secondTLDs.includes(parts[2]))))
      parts.shift()

  }
  return parts.join('.')
}
export function getHost() {
  return getDomain(window.location.hostname)
}
export function getQuery() {
  return window.location.search
}

export const isMobile = () => {
  const toMatch = [
    /Android/i,
    /webOS/i,
    /iPhone/i,
    /iPad/i,
    /iPod/i,
    /BlackBerry/i,
    /Windows Phone/i,
    /mobile/i,
  ]
  return toMatch.some(( toMatchItem ) => {
    return navigator.userAgent.match(toMatchItem)
  })
}
export const p0 = ( n: number, t: number ) => {
  return String(t).padStart(n, '0')
}

export function isOnApp() {
  const queryString = window.location.search
  const urlParams = new URLSearchParams(queryString)
  const newParams = new URLSearchParams()
  for (const [name, value] of urlParams) {
    newParams.append(name.toLowerCase(), value)
  }
  const value = newParams.get('isonapp')
  if( value == null) {
    return false
  }
  return value.toLocaleLowerCase() === 'true'
}

export function postAppMessage(message: string) {
  if (typeof window.flutterChannel !== 'undefined') {
    window.flutterChannel.postMessage(message)
  }
}