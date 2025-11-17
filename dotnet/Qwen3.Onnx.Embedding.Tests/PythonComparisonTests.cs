using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Qwen3.Onnx.Utils;
using Yuniko.Software.Qwen3Tokenizer;

namespace Qwen3.Onnx.Embedding.Tests;

public class PythonComparisonTests : IDisposable
{
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly InferenceSession _session;

    public PythonComparisonTests()
    {
        _tokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-Embedding-0.6B", isForEmbeddingModel: true);

        var modelPath = RepositoryPaths.GetEmbeddingModelPath();
        var sessionOptions = new SessionOptions();
        _session = new InferenceSession(modelPath, sessionOptions);
    }

    [Theory]
    [ClassData(typeof(TestTextData))]
    public void Embedding_MatchesPythonReference(string text)
    {
        var reference = PythonReferenceDataProvider.GetReferenceEmbeddings()[text];
        var embedding = GetEmbedding(text);

        Assert.Equal(reference.Dimension, embedding.Length);

        var similarity = CalculateCosineSimilarity(embedding, [.. reference.Embedding]);
        Assert.True(
            similarity >= 0.9999,
            $"Cosine similarity {similarity:F6} is below threshold 0.9999 for text: '{text}'"
        );
    }

    private float[] GetEmbedding(string text)
    {
        var onnxInputs = _tokenizer.PrepareForOnnx(text);

        var inputIdsTensor = new DenseTensor<long>(onnxInputs.InputIds, [1, onnxInputs.SequenceLength]);
        var attentionMaskTensor = new DenseTensor<long>(onnxInputs.AttentionMask, [1, onnxInputs.SequenceLength]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        using var results = _session.Run(inputs);
        var allHiddenStates = results[0].AsEnumerable<float>().ToArray();

        var embedding = LastTokenPool(allHiddenStates, onnxInputs.AttentionMask, onnxInputs.SequenceLength);
        var normalizedEmbedding = Normalize(embedding);

        return normalizedEmbedding;
    }

    private static float[] LastTokenPool(float[] hiddenStates, long[] attentionMask, int sequenceLength)
    {
        const int hiddenSize = 1024;
        int lastTokenIndex = -1;

        for (int i = sequenceLength - 1; i >= 0; i--)
        {
            if (attentionMask[i] == 1)
            {
                lastTokenIndex = i;
                break;
            }
        }

        var embedding = new float[hiddenSize];
        Array.Copy(hiddenStates, lastTokenIndex * hiddenSize, embedding, 0, hiddenSize);

        return embedding;
    }

    private static float[] Normalize(float[] vector)
    {
        double sum = 0;
        for (int i = 0; i < vector.Length; i++)
        {
            sum += vector[i] * vector[i];
        }

        float norm = (float)Math.Sqrt(sum);
        var normalized = new float[vector.Length];

        for (int i = 0; i < vector.Length; i++)
        {
            normalized[i] = vector[i] / norm;
        }

        return normalized;
    }

    private static double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA.Length != vectorB.Length)
            throw new ArgumentException("Vectors must be of the same length");

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    public void Dispose()
    {
        _session?.Dispose();

        GC.SuppressFinalize(this);
    }

    private class TestTextData : TheoryData<string>
    {
        public TestTextData()
        {
            foreach (var text in PythonReferenceDataProvider.GetTestTexts())
            {
                Add(text);
            }
        }
    }
}
