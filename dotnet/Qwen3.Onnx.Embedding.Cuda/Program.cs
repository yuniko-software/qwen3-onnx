using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Qwen3.Onnx.Utils;
using Yuniko.Software.Qwen3Tokenizer;

var modelPath = RepositoryPaths.GetEmbeddingModelPath();
const string tokenizerModel = "Qwen/Qwen3-Embedding-0.6B";
const int cudaDeviceId = 0;

Console.WriteLine("Loading tokenizer...");
var tokenizer = await Qwen3Tokenizer.FromHuggingFaceAsync(tokenizerModel, isForEmbeddingModel: true);
Console.WriteLine("Tokenizer loaded");

Console.WriteLine("Loading embedding model with CUDA...");
using var sessionOptions = new SessionOptions
{
    EnableMemoryPattern = true,
    EnableCpuMemArena = false,
    LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING,
};

sessionOptions.AppendExecutionProvider_CUDA(cudaDeviceId);

using var session = new InferenceSession(modelPath, sessionOptions);
Console.WriteLine("Model loaded\n");

var testTexts = new[]
{
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
};

foreach (var text in testTexts)
{
    Console.WriteLine($"Text: {text}");

    var embedding = GetEmbedding(session, tokenizer, text);

    Console.WriteLine($"Embedding dimension: {embedding.Length}");
    Console.WriteLine($"First 10 values: [{string.Join(", ", embedding.Take(10).Select(x => $"{x:F4}"))}]");
    Console.WriteLine();
}

static float[] GetEmbedding(InferenceSession session, Qwen3Tokenizer tokenizer, string text)
{
    var onnxInputs = tokenizer.PrepareForOnnx(text);

    var inputIdsTensor = new DenseTensor<long>(onnxInputs.InputIds, [1, onnxInputs.SequenceLength]);
    var attentionMaskTensor = new DenseTensor<long>(onnxInputs.AttentionMask, [1, onnxInputs.SequenceLength]);

    var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
        NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
    };

    using var results = session.Run(inputs);
    var embedding = results[0].AsEnumerable<float>().ToArray();

    return embedding;
}
