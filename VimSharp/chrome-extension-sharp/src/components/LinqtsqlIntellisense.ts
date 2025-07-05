import { IntellisenseItem } from "./CodeEditorTypes"

interface IntellisenseCursor
{
    prevTokens: string[]
    prevSpaces: boolean
    afterSpaces: boolean
    afterTokens: string[]
    prevText: string
    afterText: string
}

interface IntellisenseCursorResponse
{
    items: IntellisenseItem[]
}

function empty(input: IntellisenseCursor): IntellisenseCursorResponse
{
    if(input.prevTokens.length > 0)
    {
        return {
            items: []
        }
    }

    return {
        items: [
            {
                title: "From",
                context: "From "
            }
        ]
    }
}

function from(input: IntellisenseCursor): IntellisenseCursorResponse
{
    if(input.prevTokens.length == 0)
    {
        return {
            items: []
        }
    }
    if(input.prevTokens[input.prevTokens.length - 1].toLowerCase() != "from")
    {
        return {
            items: []
        }
    }
    // read top 3 table names
    return {
        items: []
    }
}
