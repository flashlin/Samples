import { IntellisenseItem } from "@/components/CodeEditorTypes"

interface IntellisenseReq {
    prevTokens: string[]
    prevText: string
    nextTokens: string[]
    nextText: string
}

interface IntellisenseResp {
    items: IntellisenseItem[]
}

function empty(req: IntellisenseReq): IntellisenseResp {
    if( req.prevTokens.length !== 0 ) {
        return {
            items: []
        }
    }
    if( req.nextTokens.length !== 0 ) {
        return {
            items: []
        }
    }
    return {
        items: [
            {
                title: 'FROM',
                context: 'FROM '
            }
        ]
    }
}

