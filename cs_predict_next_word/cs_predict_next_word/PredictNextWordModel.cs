public class PredictNextWordModel
{
    static List<(string, double)> PredictNextWords(Dictionary<string, Dictionary<string, int>> model, string sentence)
    {
        var words = sentence.Split(' ');
        var nGram = model.Keys.First().Split(' ').Length;
        var lastGram = string.Join(" ", words[^nGram..]);

        if (!model.ContainsKey(lastGram))
        {
            return new List<(string, double)> { ("Unknown", 0.0) };
        }

        var possibleWords = model[lastGram];
        var totalFrequency = 0;

        foreach (var word in possibleWords.Keys)
        {
            totalFrequency += possibleWords[word];
        }

        var nextWords = new List<(string, double)>();

        foreach (var word in possibleWords.Keys)
        {
            var probability = (double)possibleWords[word] / totalFrequency;
            nextWords.Add((word, probability));
        }

        nextWords.Sort((x, y) => y.Item2.CompareTo(x.Item2));

        if (nextWords.Count > 5)
        {
            return nextWords.GetRange(0, 5);
        }
        else
        {
            return nextWords;
        }
    }
}