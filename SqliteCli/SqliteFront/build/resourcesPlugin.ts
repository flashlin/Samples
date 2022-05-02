import type { Manifest, Plugin, ResolvedConfig } from "vite"
import { minify } from "terser"
import path, { resolve } from "path"
import { promises as fs , existsSync } from "fs"
import type { DOMWindow } from "jsdom"
import type { BrotliOptions, ZlibOptions } from 'zlib'
import * as zlib from 'zlib'

type CompressionOptions = Partial<ZlibOptions> | Partial<BrotliOptions>
interface ScriptTag {
  id?: string
  src?: string
  isEsModule: boolean
  innerScript?: string
}

interface LinkTag {
  rel: string
  href: string
}

const safari10NoModuleFix = `!function(){var e=document,t=e.createElement("script");if(!("noModule"in t)&&"onbeforeload"in t){var n=!1;e.addEventListener("beforeload",(function(e){if(e.target===t)n=!0;else if(!e.target.hasAttribute("nomodule")||!n)return;e.preventDefault()}),!0),t.type="module",t.src=".",e.head.appendChild(t),t.remove()}}();`
const generateFallBackCode = (fallBackImport: string, legacyId: string) => `!function(){try{new Function("m","return import(m)")}catch(o){console.warn("vite: loading legacy build because dynamic import is unsupported, syntax error above should be ignored");var e=document.getElementById("${legacyId}"),n=document.createElement("script");n.src=e.src,n.onload=function(){${ fallBackImport }},document.body.appendChild(n)}}()`
const generateDynamicImport = (fallbackUrl: string) => `System.import("${ fallbackUrl }")`

function compress(
    content: Buffer,
    algorithm: 'gzip' | 'brotliCompress' | 'deflate' | 'deflateRaw',
    options: CompressionOptions = {},
) {
  return new Promise<Buffer>((resolve, reject) => {
    // @ts-ignore
    zlib[algorithm](content, options, (err, result) =>
        err ? reject(err) : resolve(result),
    )
  })
}
type Algorithm = 'gzip' | 'brotliCompress'
function getCompressionOptions(
    algorithm:Algorithm = 'gzip',
    compressionOptions: CompressionOptions = {},
) {
  const defaultOptions: {
    [key: string]: Record<string, any>
  } = {
    gzip: {
      level: zlib.constants.Z_BEST_COMPRESSION,
    },
    brotliCompress: {
      params: {
        [zlib.constants.BROTLI_PARAM_QUALITY]:
        zlib.constants.BROTLI_MAX_QUALITY,
        [zlib.constants.BROTLI_PARAM_MODE]: zlib.constants.BROTLI_MODE_TEXT,
      },
    },
  }
  return {
    ...defaultOptions[algorithm],
    ...compressionOptions,
  } as CompressionOptions
}

interface ResourcePluginOptions {
  cdn?: string
  fileName?: string
  subDomain?: string
  outputDir?: string
  compressionOption?: CompressionOptions
}

function getExt(algorithm: Algorithm){
  switch (algorithm){
    case 'brotliCompress':
      return '.br'
    case 'gzip':
      return '.gz'
  }
}

async function compressFileAsync(filePath: string, algorithm: Algorithm, compressionOption: Partial<ZlibOptions> | Partial<BrotliOptions>) {
  const options = getCompressionOptions(algorithm, compressionOption)
  let content = await fs.readFile(filePath)

  try {
    content = await compress(content, algorithm , options)
    await fs.writeFile(`${filePath}${getExt(algorithm)}`, content)
  } catch (error) {
    console.log(`compress error: ${filePath}`)
  }
}

const generateResourcePlugin = ({ cdn, fileName = "inject-shared.js", subDomain = 'www' , outputDir , compressionOption = {}}: ResourcePluginOptions): Plugin => {
  let config: ResolvedConfig

  async function augmentManifest(manifestPath: string, outDir: string) {
    const resolveInOutDir = (path: string) => resolve(outDir, path)
    const manifest: Manifest | undefined
      = await fs.readFile(resolveInOutDir(manifestPath), "utf-8").then(JSON.parse, () => undefined)

    if (manifest) {
      const entryFiles = Object.keys(manifest).filter(file => manifest[file].isEntry)
      const fileContent = {
        legacyPolyfills: "",
        legacyEntry: "",
      }
      const legacyPolyfillId = `shared-polyfill-${ new Date().getTime() }`
      const headScriptTags: Array<ScriptTag> = []
      const headLinkTags: Array<LinkTag> = []
      const bodyTags: Array<ScriptTag> = []
      entryFiles.forEach((fileName) => {
        const manifestElement = manifest[fileName]
        if (fileName === "vite/legacy-polyfills") {
          fileContent.legacyPolyfills = manifestElement.file
        }
        else if (fileName.match(/legacy.(html|js|ts)/)) {
          fileContent.legacyEntry = manifestElement.file
        }
        else {
          headScriptTags.push({
            isEsModule: true,
            src: manifestElement.file,
          })

          if (manifestElement.imports) {
            const vendor = manifestElement.imports.map((chunk) => ({
              rel: "modulepreload",
              href: manifest[chunk].file,
            }))
            headLinkTags.push(...vendor)
          }

          if (manifestElement.css) {
            const styles = manifestElement.css.map(css => {
              return {
                rel: "stylesheet",
                href: css,
              }
            })
            headLinkTags.push(...styles)
          }
        }
      })

      if(fileContent.legacyPolyfills){
        headScriptTags.push({
          isEsModule: true,
          innerScript: generateFallBackCode(generateDynamicImport(fileContent.legacyEntry), legacyPolyfillId),
        })

        bodyTags.push({
          isEsModule: false,
          innerScript: safari10NoModuleFix,
        })

        bodyTags.push({
          isEsModule: false,
          src: fileContent.legacyPolyfills,
          id: legacyPolyfillId,
        })

        bodyTags.push({
          isEsModule: false,
          innerScript: generateDynamicImport(fileContent.legacyEntry),
        })

      }

      const functionText = renderJs(headScriptTags, headLinkTags, bodyTags)
      const minifyOutput = await minify(functionText)
      if (minifyOutput.code) {
        const outputFolder = outputDir ? `${ outDir }/${ outputDir }` : outDir
        if(!existsSync(outputFolder)){
            await fs.mkdir(outputFolder)
        }
        const outputPath = path.join( outputFolder, fileName)
        await fs.writeFile(outputPath, minifyOutput.code, "utf8")
        await compressFileAsync(outputPath , 'gzip', compressionOption)
        await compressFileAsync(outputPath , 'brotliCompress', compressionOption)
      }
    }
  }
  function getDomain(url: string) {
    const firstTLDs = 'ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|be|bf|bg|bh|bi|bj|bm|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|cl|cm|cn|co|cr|cu|cv|cw|cx|cz|de|dj|dk|dm|do|dz|ec|ee|eg|es|et|eu|fi|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|im|in|io|iq|ir|is|it|je|jo|jp|kg|ki|km|kn|kp|kr|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|na|nc|ne|nf|ng|nl|no|nr|nu|nz|om|pa|pe|pf|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|sk|sl|sm|sn|so|sr|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|yt'.split('|')
    const secondTLDs = 'com|edu|gov|net|mil|org|nom|sch|caa|res|off|gob|int|tur|ip6|uri|urn|asn|act|nsw|qld|tas|vic|pro|biz|adm|adv|agr|arq|art|ato|bio|bmd|cim|cng|cnt|ecn|eco|emp|eng|esp|etc|eti|far|fnd|fot|fst|g12|ggf|imb|ind|inf|jor|jus|leg|lel|mat|med|mus|not|ntr|odo|ppg|psc|psi|qsl|rec|slg|srv|teo|tmp|trd|vet|zlg|web|ltd|sld|pol|fin|k12|lib|pri|aip|fie|eun|sci|prd|cci|pvt|mod|idv|rel|sex|gen|nic|abr|bas|cal|cam|emr|fvg|laz|lig|lom|mar|mol|pmn|pug|sar|sic|taa|tos|umb|vao|vda|ven|mie|北海道|和歌山|神奈川|鹿児島|ass|rep|tra|per|ngo|soc|grp|plc|its|air|and|bus|can|ddr|jfk|mad|nrw|nyc|ski|spy|tcm|ulm|usa|war|fhs|vgs|dep|eid|fet|fla|flå|gol|hof|hol|sel|vik|cri|iwi|ing|abo|fam|gok|gon|gop|gos|aid|atm|gsm|sos|elk|waw|est|aca|bar|cpa|jur|law|sec|plo|www|bir|cbg|jar|khv|msk|nov|nsk|ptz|rnd|spb|stv|tom|tsk|udm|vrn|cmw|kms|nkz|snz|pub|fhv|red|ens|nat|rns|rnu|bbs|tel|bel|kep|nhs|dni|fed|isa|nsn|gub|e12|tec|орг|обр|упр|alt|nis|jpn|mex|ath|iki|nid|gda|inc|info|asia'.split('|')
    const parts = url.replace(/^www\\./, '').split('.').slice(-3)

    if (parts.length === 3) {
      if (!((firstTLDs.includes(parts[1]) || secondTLDs.includes(parts[1])) && (firstTLDs.includes(parts[2]) || secondTLDs.includes(parts[2])))) {
        parts.shift()
      }
    }

    return parts.join('.')
  }

  function getStringValue(value: string | undefined) {
    return value ? `'${ value }'` : `!1`
  }

  function getBooleanValue(value: boolean) {
    return value ? `!0` : `!1`
  }

  function createScriptFunctionStrings(headScriptTags: Array<ScriptTag>) {
    return headScriptTags.map(({ isEsModule, id, innerScript, src }) => {
      const urlPath = src ? `BASE_URL+"${src}"` : getStringValue(src)
      return `var e=loadScript(${ getBooleanValue(isEsModule) },${ getStringValue(id) },${ urlPath },${ getStringValue(innerScript) })`
    }).join("\n")
  }

  function createLinkFunctionStrings(headLinkTags: Array<LinkTag>) {
    return headLinkTags.map(({ href, rel }) => `var e =loadLink(BASE_URL+'${ href }','${ rel }')`).join("\n")
  }

  function renderJs(headScriptTags: Array<ScriptTag>, headLinkTags: Array<LinkTag>, bodyTags: Array<ScriptTag>) {
    const d = {} as DOMWindow

    function loadScript(isEsModule: string, id: string, src: string, innerScript: string) {
      const scriptElement = d.createElement("script")
      if (isEsModule) scriptElement.type = "module"
      else scriptElement.noModule = true
      if (id) scriptElement.id = id
      if (src) scriptElement.src = src
      if (innerScript) scriptElement.innerHTML = innerScript
      d.head.appendChild(scriptElement)
    }

    function loadLink(href: string, rel: string) {
      const linkElement = d.createElement("link")
      linkElement.href = href
      linkElement.rel = rel
      d.head.appendChild(linkElement)
    }
    const urlSection = cdn
      ? `
        const BASE_URL = '${cdn}/'
      `
      : `
        ${getDomain.toString()}
        const BASE_URL = ${JSON.stringify(`//${subDomain}.{host}/`)}.replace('{host}', getDomain(window.location.hostname))
      `

    return `!function(d){
    ${urlSection}
    ${ loadScript.toString() }
    ${ loadLink.toString() }
    ${ createScriptFunctionStrings(headScriptTags) }
    ${ createLinkFunctionStrings(headLinkTags) }
    ${ createScriptFunctionStrings(bodyTags) } }(window.document)`
  }

  return {
    name: "vite:shared-resource-plugin",
    enforce: "post",
    apply: "build",
    configResolved(_config) {
      config = _config
    },
    async writeBundle({ dir }) {
      if(!config.build.manifest)
        throw new Error(
          '!!! please enable build manifest !!!',
        )
      await augmentManifest("manifest.json", dir!)
    },
  }
}

export default generateResourcePlugin