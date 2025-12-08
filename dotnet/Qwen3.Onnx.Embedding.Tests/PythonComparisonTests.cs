using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Qwen3.Onnx.Utils;
using Yuniko.Software.Qwen3Tokenizer;

namespace Qwen3.Onnx.Embedding.Tests;

public sealed class PythonComparisonTests : IDisposable
{
    private readonly Qwen3Tokenizer _tokenizer;
    private readonly InferenceSession _cpuSession;
    private readonly InferenceSession? _cudaSession;
    private readonly bool _cudaAvailable;

    public PythonComparisonTests()
    {
        _tokenizer = Qwen3Tokenizer.FromHuggingFace("Qwen/Qwen3-Embedding-0.6B", isForEmbeddingModel: true);

        var modelPath = RepositoryPaths.GetEmbeddingModelPath();

        var cpuSessionOptions = new SessionOptions
        {
            EnableMemoryPattern = true,
            EnableCpuMemArena = true,
            LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
        };
        _cpuSession = new InferenceSession(modelPath, cpuSessionOptions);

        try
        {
            var cudaSessionOptions = new SessionOptions
            {
                EnableMemoryPattern = true,
                EnableCpuMemArena = false,
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
            };
            cudaSessionOptions.AppendExecutionProvider_CUDA(0);
            _cudaSession = new InferenceSession(modelPath, cudaSessionOptions);
            _cudaAvailable = true;
        }
        catch (OnnxRuntimeException)
        {
            _cudaSession = null;
            _cudaAvailable = false;
        }
    }

    [Theory]
    [ClassData(typeof(TestTextData))]
    public void CpuEmbedding_MatchesPythonReference(string text)
    {
        ValidateEmbeddingAgainstReference(_cpuSession, text, "CPU");
    }

    [SkippableTheory]
    [ClassData(typeof(TestTextData))]
    public void CudaEmbedding_MatchesPythonReference(string text)
    {
        Skip.If(!_cudaAvailable, "CUDA is not available on this system");

        Assert.NotNull(_cudaSession);
        ValidateEmbeddingAgainstReference(_cudaSession, text, "CUDA");
    }

    private void ValidateEmbeddingAgainstReference(InferenceSession session, string text, string providerName)
    {
        var reference = PythonReferenceDataProvider.GetReferenceEmbeddings()[text];
        var embedding = GetEmbedding(session, text);

        Assert.Equal(reference.Dimension, embedding.Length);

        var similarity = CalculateCosineSimilarity(embedding, [.. reference.Embedding]);
        Assert.True(
            similarity >= 0.9999,
            $"{providerName}: Cosine similarity {similarity:F10} is below threshold 0.9999 for text: '{text}'"
        );
    }

    private float[] GetEmbedding(InferenceSession session, string text)
    {
        var onnxInputs = _tokenizer.PrepareForOnnx(text);

        var inputIdsTensor = new DenseTensor<long>(onnxInputs.InputIds, [1, onnxInputs.SequenceLength]);
        var attentionMaskTensor = new DenseTensor<long>(onnxInputs.AttentionMask, [1, onnxInputs.SequenceLength]);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
        };

        using var results = session.Run(inputs);
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
        {
            throw new InvalidOperationException("Vectors must be of the same length");
        }

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
        _cudaSession?.Dispose();
        _cpuSession?.Dispose();

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
