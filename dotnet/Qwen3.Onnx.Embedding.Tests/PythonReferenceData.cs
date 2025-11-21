using System.Text.Json;
using System.Text.Json.Serialization;

namespace Qwen3.Onnx.Embedding.Tests;

public record PythonReferenceEmbedding(
    [property: JsonPropertyName("embedding")] List<float> Embedding,
    [property: JsonPropertyName("dimension")] int Dimension
);

public static class PythonReferenceDataProvider
{
    private static readonly Lock _lock = new();
    private static readonly Dictionary<string, PythonReferenceEmbedding> _cache = [];

    public static Dictionary<string, PythonReferenceEmbedding> GetReferenceEmbeddings()
    {
        lock (_lock)
        {
            if (_cache.Count > 0)
            {
                return _cache;
            }

            var referenceFile = Path.Combine(AppContext.BaseDirectory, "TestData", "reference_embeddings.json");
            var jsonContent = File.ReadAllText(referenceFile);

            var rawData = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(jsonContent)
                ?? throw new InvalidOperationException("Failed to deserialize reference embeddings");

            foreach (var kvp in rawData)
            {
                var element = kvp.Value;
                var embedding = element.GetProperty("embedding").EnumerateArray()
                    .Select(x => (float)x.GetDouble()).ToList();
                var dimension = element.GetProperty("dimension").GetInt32();

                _cache[kvp.Key] = new PythonReferenceEmbedding(embedding, dimension);
            }

            return _cache;
        }
    }

    public static IEnumerable<string> GetTestTexts()
    {
        var data = GetReferenceEmbeddings();
        return data.Keys;
    }
}
