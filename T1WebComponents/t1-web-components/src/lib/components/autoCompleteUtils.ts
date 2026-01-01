export const normalizeText = (text: string): string => {
    return text
        .replace(/([A-Z])/g, ' $1')
        .replace(/[_\s-]+/g, '')
        .toLowerCase()
}

export const splitIntoWords = (text: string): string[] => {
    if (/[_\s-]/.test(text)) {
        return []
    }

    const words: string[] = []
    let currentWord = ''

    for (let i = 0; i < text.length; i++) {
        const char = text[i]
        if (/[A-Z]/.test(char)) {
            if (currentWord) {
                words.push(currentWord)
                currentWord = ''
            }
            currentWord += char
        } else if (/[a-z0-9]/.test(char)) {
            currentWord += char
        }
    }

    if (currentWord) {
        words.push(currentWord)
    }

    return words.length > 1 ? words : []
}

export const findWordMatches = (text: string, searchWords: string[]): Array<{ start: number; end: number }> => {
    const normalizedText = normalizeText(text)
    const matches: Array<{ start: number; end: number }> = []

    const charMap: number[] = []
    let normalizedIndex = 0

    for (let i = 0; i < text.length; i++) {
        const char = text[i]
        if (/[a-zA-Z0-9]/.test(char)) {
            charMap[normalizedIndex++] = i
        } else if (/[_\s-]/.test(char)) {
            continue
        } else {
            charMap[normalizedIndex++] = i
        }
    }

    let searchIndex = 0
    for (const word of searchWords) {
        const normalizedWord = normalizeText(word)
        const isUpperCase = /^[A-Z]/.test(word)

        let searchFrom = normalizedText.indexOf(normalizedWord, searchIndex)

        if (isUpperCase && searchFrom >= 0) {
            let bestMatch = -1

            for (let i = 0; i < text.length; i++) {
                const char = text[i]
                if (char === word[0]) {
                    const charNormalizedIndex = charMap.findIndex((pos) => pos === i)
                    if (charNormalizedIndex >= 0 && charNormalizedIndex >= searchIndex) {
                        const potentialMatch = normalizedText.indexOf(normalizedWord, charNormalizedIndex)
                        if (potentialMatch === charNormalizedIndex && (bestMatch === -1 || charNormalizedIndex < bestMatch)) {
                            bestMatch = charNormalizedIndex
                        }
                    }
                }
            }

            if (bestMatch >= 0) {
                searchFrom = bestMatch
            }
        }

        if (searchFrom >= 0) {
            const wordLength = normalizedWord.length
            if (searchFrom + wordLength <= charMap.length) {
                const startIndex = charMap[searchFrom]
                const endIndex = searchFrom + wordLength < charMap.length
                    ? charMap[searchFrom + wordLength - 1] + 1
                    : text.length

                matches.push({ start: startIndex, end: endIndex })
                searchIndex = searchFrom + wordLength
            }
        }
    }

    return matches
}

export const highlightText = (
    searchText: string,
    text: string,
    highlightStart: string = '<span class="text-yellow-400 font-bold">',
    highlightEnd: string = '</span>'
): string => {
    if (!searchText.trim()) {
        return text
    }

    const normalizedText = normalizeText(text)
    const textHasSeparator = /[_\s-]/.test(text)
    const searchWords = splitIntoWords(searchText)

    if (searchWords.length > 1 && !textHasSeparator) {
        const matches = findWordMatches(text, searchWords)

        if (matches.length > 0) {
            let result = ''
            let lastIndex = 0

            for (const match of matches) {
                result += text.substring(lastIndex, match.start)
                result += `${highlightStart}${text.substring(match.start, match.end)}${highlightEnd}`
                lastIndex = match.end
            }

            result += text.substring(lastIndex)
            return result
        }
    }

    const normalizedSearch = normalizeText(searchText)

    if (!normalizedText.includes(normalizedSearch)) {
        return text
    }

    const matchIndex = normalizedText.indexOf(normalizedSearch)
    const matchLength = normalizedSearch.length

    const charMap: number[] = []
    let normalizedIndex = 0

    for (let i = 0; i < text.length; i++) {
        const char = text[i]
        if (/[a-zA-Z0-9]/.test(char)) {
            charMap[normalizedIndex++] = i
        } else if (/[_\s-]/.test(char)) {
            continue
        } else {
            charMap[normalizedIndex++] = i
        }
    }

    if (matchIndex >= 0 && matchIndex + matchLength <= charMap.length) {
        const startIndex = charMap[matchIndex]
        const endIndex = matchIndex + matchLength < charMap.length
            ? charMap[matchIndex + matchLength - 1] + 1
            : text.length

        const beforeMatch = text.substring(0, startIndex)
        const match = text.substring(startIndex, endIndex)
        const afterMatch = text.substring(endIndex)

        return `${beforeMatch}${highlightStart}${match}${highlightEnd}${afterMatch}`
    }

    return text
}
