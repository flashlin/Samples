const fs = require('fs');
const readline = require('readline');
const BasicTokenizer = require('sentence-tokenization').BasicTokenizer;

const tokenizer = new BasicTokenizer();


const rl = readline.createInterface({
   input: fs.createReadStream('myactivity.txt'),
   crlfDelay: Infinity
});

const tokenizedSentences = []


rl.on('line', (line) => {
   trimmed = line.trim().toLowerCase();
   tokens = tokenizer.tokenize(trimmed)
   if (tokens.length > 1) {
       tokenizedSentences.push(tokens);
   }
});

rl.on('close', () => {
   console.log(`tokenizing complete`);
   rl.close();
   rl.removeAllListeners();
   // console.log(tokenizedSentences.slice(0,5))
   main();
});


function countWords(tokenized) {
   let wordCounts = new Map();
   tokenized.forEach((sentence) => {
       sentence.forEach((word) => {
           if (wordCounts.has(word)) {
               wordCounts.set(word, wordCounts.get(word)+1);
           }else{
               wordCounts.set(word, 1);
           }
       });
   });
   return wordCounts;
}

function getVocab(tokenizedSentences, countThreshold){
   closedVocab = [];
   wordCounts = countWords(tokenizedSentences);
   wordCounts.forEach((count, word) => {
      if (count > countThreshold) {
         closedVocab.push(word);
      }
   });
   return closedVocab;

}

function replaceOOV(tokenizedSentences, vocab, oovToken = '<unk>') {
   const vocabSet = new Set(vocab);
   const replacedTokenizedSentences = []
   tokenizedSentences.forEach((sentence) => {
       const replacedSentence = [];
       sentence.forEach((token) => {
           if (vocabSet.has(token)) {
               replacedSentence.push(token)
           }else{
               replacedSentence.push(oovToken)
           }
       });
       replacedTokenizedSentences.push(replacedSentence)
   });
   return replacedTokenizedSentences;
}


function countNGrams(preprossed, n, startToken='<s>', endToken='<e>') {
   let nGrams = new Map();
   preprossed.forEach((sentence) => {
       padded = (new Array(n-1).fill(startToken)).concat(sentence).concat([endToken])
       for (let i = 0; i < padded.length -n + 1; i++) {
           const nGram = padded.slice(i, i+n).toString();
           if (nGrams.has(nGram)) {
               nGrams.set(nGram,nGrams.get(nGram)+1)
           }else{
               nGrams.set(nGram,1);
           }
           
       }
   });
   return nGrams;
}


function estimateProbability(word, prevNGram, nGramCounts, nPlus1GramCounts, vocabSize, k=0.1) {
   const prevNGramToString = prevNGram.toString();
   const prevNGramCount = nGramCounts.get(prevNGramToString) === undefined ? 0: nGramCounts.get(prevNGramToString);
   const denominator = prevNGramCount + k*vocabSize;
   const nPlus1Gram = prevNGram.slice()
   nPlus1Gram.push(word)
   const nPlus1GramToString = nPlus1Gram.toString();
   const nPlus1GramCount = nPlus1GramCounts.get(nPlus1GramToString) ===  undefined ? 0 : nPlus1GramCounts.get(nPlus1GramToString);
   const numerator = nPlus1GramCount + k
   // console.log(prevNGramCount, nPlus1GramCount)
   return numerator/denominator;
}

function eestimateProbabilities(prevNGram, nGramCounts, nPlus1GramCounts, vocabulary, k=0.1) {
   // const prevNGramToString = prevNGram.toString();
   const probabilities = new Map();
   vocabulary.forEach((word) => {
       probability = estimateProbability(word,prevNGram,nGramCounts,nPlus1GramCounts,vocabulary.length,k);
       probabilities.set(word,probability);
   });
   return probabilities;

}

function isAlphaNumeric(str) {
   var code, i, len;
 
   for (i = 0, len = str.length; i < len; i++) {
     code = str.charCodeAt(i);
     if (!(code > 47 && code < 58) && // numeric (0-9)
         !(code > 64 && code < 91) && // upper alpha (A-Z)
         !(code > 96 && code < 123)) { // lower alpha (a-z)
       return false;
     }
   }
   return true;
 };

function getSuggestionWord(prevTokens,n, nGramCounts, nPlus1GramCounts, vocabulary, k=0.1, startsWith='') {
   if (prevTokens.length < n) {
       prevTokens = (new Array(n-prevTokens.length).fill('<s>')).concat(prevTokens)
   }
   const probabilities = eestimateProbabilities(prevTokens.slice(prevTokens.length-n),nGramCounts, nPlus1GramCounts, vocabulary, k);
   let suggestion;
   let maxProb=0;
   probabilities.forEach((value, key) => {
       if (!isAlphaNumeric(key)) {
           return;
       }
       if (key.startsWith(startsWith)) {
           if (value > maxProb) {
               suggestion = key;
               maxProb = value
           }
       }
   });
   return [suggestion, maxProb];
}

function getAllSuggestions(prevTokens, nGramCountsList,vocabulary, k=0.1,startsWith='') {
   const modelsCount = nGramCountsList.length;
   const suggestions=[];
   for (let i = 0; i < modelsCount - 1; i++) {
       const nGramCounts = nGramCountsList[i]
       const nPlus1GramCounts = nGramCountsList[i+1]
       const suggestion = getSuggestionWord(prevTokens,i+1,nGramCounts,nPlus1GramCounts,vocabulary,k,startsWith);
       suggestions.push(suggestion)
       
   }
   return suggestions;
}

function main() {
   
   const vocab = getVocab(tokenizedSentences,1)

   const preprocessedSentences = replaceOOV(tokenizedSentences,vocab)

   const uniGrams = countNGrams(preprocessedSentences,1)
   const biGrams = countNGrams(preprocessedSentences,2)
   const triGrams = countNGrams(preprocessedSentences,3)
   const quadGrams = countNGrams(preprocessedSentences,4)
   const pentaGrams = countNGrams(preprocessedSentences,5)
   const nGramCountsList = [uniGrams, biGrams, triGrams, quadGrams,pentaGrams];
  
   const userInput = 'from customer select'
   const prevTokens = userInput.split(' ')

   console.log(getAllSuggestions(prevTokens, nGramCountsList, vocab, 0.1))
}


